"""Build dataset items from plant recipes (stage 0D).

A recipe, ``eval/plants/<item_id>.yaml``, names an upstream commit and the bugs to plant in lines
that commit added (``find`` -> ``replace``). The item's diff runs from the commit's parent to the
commit plus the bugs. A clean item is the upstream commit unchanged.

For each recipe this writes ``eval/dataset/<item_id>/``:

- ``pr.diff``: ``git diff --no-color --no-ext-diff --no-textconv -M <parent>`` in a temporary
  worktree at the commit holding the planted files;
- ``ground_truth.json``: the ``GroundTruth`` model; each bug's ``line_start``/``line_end`` are the
  first and last new-file lines where the planted file differs from the file at the commit
  (a line diff, ``difflib``);
- ``meta.json``: commit details, size, the plant patch, sandboxed test results and static hits.

Upstream code never runs on the host: only ``git`` and the static tools ``ruff`` and ``bandit``
(D2a) run here; tests run through ``run_in_sandbox`` (D2). Test results hold durations, so they
are reused from the existing ``meta.json`` when their inputs are unchanged (``tests_key``); this
keeps a rebuild byte-identical. ``--rerun-tests`` forces new runs.

Usage:
    uv run python -m eval.build_dataset <item_id> [...] [--rerun-tests]
    uv run python -m eval.build_dataset --all
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eval.check_repos import parse_pytest_summary
from eval.ground_truth import (
    GroundTruth,
    PlantedBug,
    changed_py_file_count,
    default_accepted_kinds,
    parse_diff,
    size_bucket,
)
from eval.repos import (
    DIFF_OPTIONS,
    add_worktree,
    clone_dir,
    diff_file_changes,
    ensure_clone,
    get_repo,
    git_bytes,
    item_files,
    remove_worktree,
    run_repo_tests,
    snapshot_files,
    workspaces_dir,
)
from reviewer.config import repo_root
from reviewer.sandbox import run_in_sandbox
from reviewer.schemas import validate_posix_path

type PlantKind = Literal[
    "off_by_one",
    "missing_none_check",
    "wrong_comparison",
    "unhandled_exception",
    "sql_injection",
    "hardcoded_secret",
    "ineffective_test",
]
type Bucket = Literal["small", "medium", "large"]

RUFF_SELECT = "E,F,B,S"  # pycodestyle errors, pyflakes, bugbear, flake8-bandit
# "assert used" (ruff S101, bandit B101) flags every assert in a test, so a hit says nothing
# about a planted bug; both are left out of the (informational) static hits.
RUFF_IGNORE = "S101"
BANDIT_SKIP = "B101"


def plants_dir() -> Path:
    return repo_root() / "eval" / "plants"


def dataset_dir() -> Path:
    return repo_root() / "eval" / "dataset"


class PlantError(ValueError):
    """A recipe cannot be applied (find not unique, plant changes nothing, ...)."""


# --- Recipes ---------------------------------------------------------------------------------


class BugRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bug_id: str = Field(pattern=r"^b\d+$")
    kind: PlantKind
    file: str
    find: str = Field(min_length=1)
    replace: str
    description: str = Field(min_length=1)
    # Per-bug extra accepted kinds (decision 0005). Each bug that uses them needs a reason and an
    # entry in docs/decisions/0005-per-bug-accepted-kinds.md (check 12). ``other`` is not a
    # PlantKind, so it can never be added.
    extra_accepted_kinds: list[PlantKind] = Field(default_factory=list)
    extra_accepted_reason: str | None = None

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str) -> str:
        return validate_posix_path(v)

    @model_validator(mode="after")
    def _check_extra_kinds(self) -> Self:
        if self.extra_accepted_kinds:
            if self.extra_accepted_reason is None or not self.extra_accepted_reason.strip():
                raise ValueError("extra_accepted_reason is required with extra_accepted_kinds")
            if self.kind in self.extra_accepted_kinds:
                raise ValueError("extra_accepted_kinds must not repeat the bug's own kind")
            if len(set(self.extra_accepted_kinds)) != len(self.extra_accepted_kinds):
                raise ValueError("extra_accepted_kinds must not contain duplicates")
        elif self.extra_accepted_reason is not None:
            raise ValueError("extra_accepted_reason is only allowed with extra_accepted_kinds")
        return self


def accepted_kinds_for(bug: BugRecipe) -> list[str]:
    """``default_accepted_kinds(kind)`` followed by the bug's extra kinds, without duplicates and
    in that order (decision 0005)."""
    return list(dict.fromkeys([*default_accepted_kinds(bug.kind), *bug.extra_accepted_kinds]))


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"^item_\d{3}$")
    repo: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_bucket: Bucket
    is_clean: bool
    bugs: list[BugRecipe]

    @model_validator(mode="after")
    def _check_bugs(self) -> Self:
        if self.is_clean != (not self.bugs):
            raise ValueError("is_clean must be true exactly when bugs is empty")
        if len(self.bugs) > 3:
            raise ValueError("at most 3 bugs per item")
        ids = [b.bug_id for b in self.bugs]
        if len(set(ids)) != len(ids):
            raise ValueError("bug_id values must be unique")
        return self


def recipe_path(item_id: str) -> Path:
    return plants_dir() / f"{item_id}.yaml"


def load_recipe(item_id: str, path: Path | None = None) -> Recipe:
    path = path or recipe_path(item_id)
    recipe = Recipe.model_validate(yaml.safe_load(path.read_bytes().decode("utf-8")))
    if recipe.item_id != path.stem:
        raise PlantError(f"{path}: item_id {recipe.item_id!r} does not match the file name")
    return recipe


def all_item_ids() -> list[str]:
    return sorted(p.stem for p in plants_dir().glob("item_*.yaml"))


# --- Find/replace and line ranges (pure) ------------------------------------------------------


def split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only (as git does), keeping line endings."""
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def line_of(text: str, offset: int) -> int:
    """1-based line number of the character at ``offset``."""
    return text.count("\n", 0, offset) + 1


@dataclass(frozen=True)
class FindSpan:
    """Where a bug's ``find`` text sits in the file at the commit (1-based, inclusive)."""

    bug_id: str
    file: str
    start_offset: int
    end_offset: int  # exclusive
    first_line: int
    last_line: int


def locate_find(text: str, find: str, bug_id: str, file: str) -> FindSpan:
    count = text.count(find)
    if count != 1:
        raise PlantError(f"{bug_id}: find text occurs {count} times in {file} (must be once)")
    start = text.index(find)
    end = start + len(find)
    return FindSpan(bug_id, file, start, end, line_of(text, start), line_of(text, end - 1))


@dataclass
class PlantResult:
    original: dict[str, str]  # path -> text at the commit (only files with bugs)
    planted: dict[str, str]  # path -> planted text (only files with bugs)
    spans: dict[str, FindSpan]  # bug_id -> where its find text is at the commit
    ranges: dict[str, tuple[int, int]]  # bug_id -> (line_start, line_end) in the planted file


def apply_plants(files: dict[str, bytes], bugs: Sequence[BugRecipe]) -> PlantResult:
    """Apply each bug's ``find`` -> ``replace`` in order to ``files`` (the commit's changed files).

    Each ``find`` must occur exactly once in the file at the commit and exactly once in the
    text produced by the earlier replacements, and bugs in one file must not overlap.
    """
    original: dict[str, str] = {}
    current: dict[str, str] = {}
    spans: dict[str, FindSpan] = {}
    for bug in bugs:
        if bug.file not in files:
            raise PlantError(f"{bug.bug_id}: {bug.file} is not a file the commit adds or changes")
        if bug.file not in original:
            try:
                original[bug.file] = files[bug.file].decode("utf-8")
            except UnicodeDecodeError as e:
                raise PlantError(f"{bug.bug_id}: {bug.file} is not UTF-8: {e}") from e
            current[bug.file] = original[bug.file]
        if bug.find == bug.replace:
            raise PlantError(f"{bug.bug_id}: replace is identical to find")
        span = locate_find(original[bug.file], bug.find, bug.bug_id, bug.file)
        for other in spans.values():
            if other.file == bug.file and (
                span.start_offset < other.end_offset and other.start_offset < span.end_offset
            ):
                raise PlantError(f"{bug.bug_id}: find text overlaps {other.bug_id}'s")
        spans[bug.bug_id] = span
        text = current[bug.file]
        n = text.count(bug.find)
        if n != 1:
            raise PlantError(
                f"{bug.bug_id}: find text occurs {n} times in {bug.file} after earlier plants"
            )
        current[bug.file] = text.replace(bug.find, bug.replace, 1)

    ranges: dict[str, tuple[int, int]] = {}
    for path in original:
        file_spans = [s for s in spans.values() if s.file == path]
        ranges.update(planted_line_ranges(original[path], current[path], file_spans))
    return PlantResult(original, current, spans, ranges)


def diff_regions(original: str, planted: str) -> list[tuple[int, int, int, int]]:
    """Non-equal regions of a line diff: (i1, i2, j1, j2), 0-based half-open line indices."""
    matcher = difflib.SequenceMatcher(None, split_lines(original), split_lines(planted), False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def planted_line_ranges(
    original: str, planted: str, spans: Sequence[FindSpan]
) -> dict[str, tuple[int, int]]:
    """Each bug's (line_start, line_end) in the planted file: the first and last new-file lines
    of the line-diff regions that fall inside that bug's ``find`` lines.

    A region that only deletes lines has no new-file line; it is located at the new-file line
    that follows the deletion (the previous line at the end of the file), as ``changed_lines``
    in ``eval/score.py`` does for removed lines. Every region must belong to exactly one bug and
    every bug must own at least one region.
    """
    n_new = len(split_lines(planted))
    owned: dict[str, list[tuple[int, int]]] = {s.bug_id: [] for s in spans}
    for i1, i2, j1, j2 in diff_regions(original, planted):
        owners = []
        for s in spans:
            lo, hi = s.first_line - 1, s.last_line  # 0-based half-open
            inside = (i1 < hi and i2 > lo) if i2 > i1 else (lo <= i1 <= hi)
            if inside:
                owners.append(s.bug_id)
        if len(owners) != 1:
            raise PlantError(
                f"line-diff region at original lines {i1 + 1}-{i2} belongs to "
                f"{len(owners)} bugs ({owners}); plants must stay inside their find text"
            )
        if j2 > j1:
            owned[owners[0]].append((j1 + 1, j2))
        else:
            line = j1 + 1 if j1 < n_new else max(n_new, 1)
            owned[owners[0]].append((line, line))
    out: dict[str, tuple[int, int]] = {}
    for bug_id, regions in owned.items():
        if not regions:
            raise PlantError(f"{bug_id}: the plant does not change any line")
        out[bug_id] = (min(r[0] for r in regions), max(r[1] for r in regions))
    return out


# --- Git helpers -----------------------------------------------------------------------------


def parent_of(repo: Path, commit: str) -> str:
    parents = git_bytes(repo, "rev-list", "--parents", "-n", "1", commit).decode().split()[1:]
    if len(parents) != 1:
        raise PlantError(f"{commit}: expected exactly one parent, got {len(parents)}")
    return parents[0]


def commit_message(repo: Path, commit: str) -> tuple[str, str]:
    """(subject, body) of the commit message; the body may be empty."""
    msg = git_bytes(repo, "log", "-1", "--format=%B", commit).decode("utf-8")
    lines = msg.replace("\r\n", "\n").split("\n")
    subject = lines[0].strip()
    body = "\n".join(lines[1:]).strip("\n").rstrip()
    return subject, body


def upstream_diff(repo: Path, parent: str, commit: str) -> str:
    return git_bytes(repo, "diff", *DIFF_OPTIONS, parent, commit).decode("utf-8")


def changed_files_at(repo: Path, parent: str, commit: str) -> dict[str, bytes]:
    """Files the commit adds or modifies (rename targets included), read at ``commit`` from git
    objects. Deleted files are left out."""
    written, _ = diff_file_changes(upstream_diff(repo, parent, commit))
    return {p: git_bytes(repo, "cat-file", "blob", f"{commit}:{p}") for p in sorted(written)}


def new_worktree(repo: Path, commit: str, label: str, root: Path | None = None) -> Path:
    """A fresh worktree with a unique name under ``root`` (default ``eval/workspaces/``)."""
    root = root or workspaces_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=root))
    path.rmdir()  # git worktree add wants to create it
    return add_worktree(repo, path, commit)


def write_files(worktree: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        (worktree / PurePosixPath(path)).write_bytes(text.encode("utf-8"))


# --- Static analysis (host, read-only, D2a) ---------------------------------------------------


def _tool(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def ruff_lines(path: Path) -> list[tuple[int, int, str]]:
    """(first line, last line, code) for each ruff diagnostic in ``path``."""
    res = subprocess.run(
        [
            *_tool("ruff"),
            "check",
            "--isolated",
            "--no-cache",
            "--select",
            RUFF_SELECT,
            "--ignore",
            RUFF_IGNORE,
            "--output-format",
            "json",
            "--exit-zero",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    out = []
    for d in json.loads(res.stdout or b"[]"):
        start = d["location"]["row"]
        end = (d.get("end_location") or d["location"])["row"]
        out.append((start, end, d.get("code") or "syntax"))
    return out


def bandit_lines(path: Path) -> list[tuple[int, int, str]]:
    """(first line, last line, test id) for each bandit issue in ``path``."""
    res = subprocess.run(
        [*_tool("bandit"), "-q", "-s", BANDIT_SKIP, "-f", "json", str(path)],
        capture_output=True,
        check=False,
    )
    data = json.loads(res.stdout or b"{}")
    if data.get("errors"):
        raise RuntimeError(f"bandit failed on {path}: {data['errors']}")
    out = []
    for r in data.get("results", []):
        rng = r.get("line_range") or [r["line_number"]]
        out.append((min(rng), max(rng), r["test_id"]))
    return out


def static_hits(file_path: Path, line_start: int, line_end: int) -> dict[str, Any]:
    def hits(diags: list[tuple[int, int, str]]) -> list[str]:
        return sorted({c for a, b, c in diags if a <= line_end and b >= line_start})

    ruff = hits(ruff_lines(file_path))
    bandit = hits(bandit_lines(file_path))
    return {"flagged": bool(ruff or bandit), "ruff": ruff, "bandit": bandit}


# --- Tests in the sandbox (D2) ----------------------------------------------------------------

_FAILED_LINE = re.compile(r"^(FAILED|ERROR) (\S+)", re.MULTILINE)


def test_summary(stdout: str, exit_code: int | None, timed_out: bool) -> dict[str, Any]:
    c = parse_pytest_summary(stdout)
    m = re.search(r"\bin ([\d.]+)s\b", c.summary)
    return {
        "passed": c.passed,
        "failed": c.failed,
        "errors": c.errors,
        "skipped": c.skipped,
        "duration": float(m.group(1)) if m else None,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "summary": c.summary,
        "failing": sorted({f"{k} {n}" for k, n in _FAILED_LINE.findall(stdout)}),
    }


def run_tests(repo_name: str, files: dict[str, bytes], extra: Sequence[str] = ()) -> dict:
    spec = get_repo(repo_name)
    if extra:
        res = run_in_sandbox(spec.image, files, [*spec.test_command, *extra], spec.test_timeout_s)
    else:
        res = run_repo_tests(repo_name, files)
    return test_summary(res.stdout, res.exit_code, res.timed_out)


def enclosing_test(source: str, file: str, line: int) -> str | None:
    """Node id (``file::[Class::]test_name``) of the test function that contains ``line``."""
    tree = ast.parse(source)
    best: tuple[int, str] | None = None

    def visit(node: ast.AST, prefix: list[str]) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, [*prefix, child.name])
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                end = child.end_lineno or child.lineno
                if child.name.startswith("test") and child.lineno <= line <= end:
                    depth = len(prefix)
                    if best is None or depth >= best[0]:
                        best = (depth, "::".join([file, *prefix, child.name]))

    visit(tree, [])
    return best[1] if best else None


# --- Building --------------------------------------------------------------------------------


def candidate_entry(repo_name: str, commit: str) -> tuple[str, dict] | None:
    """(bucket, entry) for ``commit`` in ``eval/candidates/<repo>.json``, or None."""
    data = json.loads((repo_root() / "eval" / "candidates" / f"{repo_name}.json").read_bytes())
    for bucket, entries in data["buckets"].items():
        for e in entries:
            if e["sha"] == commit:
                return bucket, e
    return None


def lines_changed(diff_text: str) -> int:
    return sum(pf.added + pf.removed for pf in parse_diff(diff_text))


def ground_truth_for(recipe: Recipe, slug: str, parent: str, ranges: dict) -> GroundTruth:
    return GroundTruth(
        item_id=recipe.item_id,
        source_repo=slug,
        base_commit=parent,
        is_clean=recipe.is_clean,
        bugs=[
            PlantedBug(
                bug_id=b.bug_id,
                file=b.file,
                line_start=ranges[b.bug_id][0],
                line_end=ranges[b.bug_id][1],
                kind=b.kind,
                accepted_kinds=accepted_kinds_for(b),
                description=b.description,
            )
            for b in recipe.bugs
        ],
    )


@dataclass
class Built:
    recipe: Recipe
    repo: Path
    parent: str
    commit_files: dict[str, bytes]
    plant: PlantResult
    pr_diff: str
    plant_patch: str
    static: dict[str, dict[str, Any]]


def build_in_repo(recipe: Recipe, repo: Path, *, workspace_root: Path | None = None) -> Built:
    """Apply the recipe in a temporary worktree of ``repo`` and return the diffs (no tests)."""
    parent = parent_of(repo, recipe.commit)
    commit_files = changed_files_at(repo, parent, recipe.commit)
    plant = apply_plants(commit_files, recipe.bugs)
    wt = new_worktree(repo, recipe.commit, f"build-{recipe.item_id}", workspace_root)
    try:
        write_files(wt, plant.planted)
        pr_diff = git_bytes(wt, "diff", *DIFF_OPTIONS, parent).decode("utf-8")
        plant_patch = git_bytes(wt, "diff", *DIFF_OPTIONS).decode("utf-8")
        static = {
            b.bug_id: static_hits(wt / PurePosixPath(b.file), *plant.ranges[b.bug_id])
            for b in recipe.bugs
        }
    finally:
        remove_worktree(repo, wt)
        shutil.rmtree(wt, ignore_errors=True)
    return Built(recipe, repo, parent, commit_files, plant, pr_diff, plant_patch, static)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SCORING_ONLY_FIELDS = {"extra_accepted_kinds", "extra_accepted_reason"}


def tests_key(recipe: Recipe, pr_diff: str) -> str:
    spec = get_repo(recipe.repo)
    payload = {
        "image": spec.image,
        "test_command": spec.test_command,
        "commit": recipe.commit,
        "pr_diff": _sha256(pr_diff.encode("utf-8")),
        # Accepted kinds do not change what runs in the sandbox, so they are left out of the key;
        # adding or editing them never re-runs the tests.
        "bugs": [b.model_dump(exclude=_SCORING_ONLY_FIELDS) for b in recipe.bugs],
    }
    return _sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))


def run_item_tests(built: Built) -> dict[str, Any]:
    """Sandboxed runs: original commit, planted item, each bug alone, and for each
    ``ineffective_test`` bug its own test on the planted item."""
    recipe = built.recipe
    name = recipe.repo
    at_commit = snapshot_files(name, recipe.commit)
    out: dict[str, Any] = {
        "tests_original": run_tests(name, at_commit),
        "tests_planted": run_tests(name, item_files(recipe.item_id)),
        "bug_tests": {},
    }
    for bug in recipe.bugs:
        alone = apply_plants(built.commit_files, [bug])
        files = dict(at_commit)
        files.update({p: t.encode("utf-8") for p, t in alone.planted.items()})
        entry: dict[str, Any] = {"alone": run_tests(name, files)}
        if bug.kind == "ineffective_test":
            node = enclosing_test(
                built.plant.planted[bug.file], bug.file, built.plant.ranges[bug.bug_id][0]
            )
            entry["own_test"] = node
            entry["own_test_planted"] = (
                run_tests(name, item_files(recipe.item_id), [node]) if node else None
            )
        out["bug_tests"][bug.bug_id] = entry
    return out


def dumps_json(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build_item(item_id: str, *, rerun_tests: bool = False) -> Path:
    recipe = load_recipe(item_id)
    spec = get_repo(recipe.repo)
    repo = clone_dir(recipe.repo)
    if not (repo / ".git").exists():
        ensure_clone(recipe.repo)
    built = build_in_repo(recipe, repo)
    subject, body = commit_message(repo, recipe.commit)
    found = candidate_entry(recipe.repo, recipe.commit)

    out_dir = dataset_dir() / item_id
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = ground_truth_for(recipe, spec.slug, built.parent, built.plant.ranges)
    (out_dir / "pr.diff").write_bytes(built.pr_diff.encode("utf-8"))
    (out_dir / "ground_truth.json").write_bytes((gt.model_dump_json(indent=2) + "\n").encode())

    meta_path = out_dir / "meta.json"
    key = tests_key(recipe, built.pr_diff)
    old = json.loads(meta_path.read_bytes()) if meta_path.is_file() else {}
    if not rerun_tests and old.get("tests_key") == key:
        tests = {k: old[k] for k in ("tests_original", "tests_planted", "bug_tests")}
    else:
        tests = run_item_tests(built)

    meta = {
        "item_id": item_id,
        "repo": recipe.repo,
        "upstream_commit": recipe.commit,
        "parent": built.parent,
        "pr_title": subject,
        "pr_body": body,
        "bucket": size_bucket(built.pr_diff),
        "py_files_changed": changed_py_file_count(built.pr_diff),
        "lines_changed": lines_changed(built.pr_diff),
        "sweep": found[1]["sweep"] if found else None,
        "plant_patch": built.plant_patch,
        "tests_key": key,
        **tests,
        "static_hits": built.static,
    }
    meta_path.write_bytes(dumps_json(meta))
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.build_dataset", description=__doc__)
    parser.add_argument("items", nargs="*", help="item ids (eval/plants/<item_id>.yaml)")
    parser.add_argument("--all", action="store_true", help="build every recipe")
    parser.add_argument("--rerun-tests", action="store_true", help="ignore cached test results")
    args = parser.parse_args(argv)
    items = all_item_ids() if args.all else args.items
    if not items:
        parser.error("give item ids or --all")
    failed = False
    for item_id in items:
        try:
            out = build_item(item_id, rerun_tests=args.rerun_tests)
        except (PlantError, RuntimeError, OSError, ValueError) as e:
            print(f"{item_id}: FAILED: {e}", file=sys.stderr)
            failed = True
            continue
        meta = json.loads((out / "meta.json").read_bytes())
        print(
            f"{item_id}: {meta['repo']} {meta['upstream_commit'][:12]} {meta['bucket']} "
            f"py_files={meta['py_files_changed']} lines={meta['lines_changed']} "
            f"original[{meta['tests_original']['summary']}] "
            f"planted[{meta['tests_planted']['summary']}]"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
