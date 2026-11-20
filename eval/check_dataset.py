"""Checks for built dataset items (stage 0D). Every check must pass for every item.

Usage:
    uv run python -m eval.check_dataset [item_id ...] [--no-tests]

Prints one PASS/FAIL line per check per item and exits 1 if any check fails. ``--no-tests``
skips check 9 (sandboxed tests) and reports it as SKIP, which still exits 1: a full check needs
Docker.

The checks, numbered as in the stage 0D brief:

1. schema: ``load_item`` succeeds and the recipe matches the ground truth and ``meta.json``;
   each bug's ``accepted_kinds`` is ``default_accepted_kinds(kind)`` plus the recipe's
   ``extra_accepted_kinds`` (no duplicates, never ``other``).
2. eligibility: the commit is indexed in the expected bucket, not before ``eligible_since``, not a
   sweep.
3. bucket: ``size_bucket(pr.diff)`` is the expected bucket (item_003: 4 or 5 ``.py`` files).
4. diff: ``pr.diff`` applies to a fresh worktree at ``base_commit`` and gives exactly the planted
   files; a clean item's ``pr.diff`` is byte-identical to the upstream diff.
5. added_lines: every ``find`` line was added by the upstream commit, and every ground-truth line
   is an added line of ``pr.diff``.
6. plant_real: each plant changes the file, and the ground-truth range is exactly the changed
   lines (recomputed).
7. separation: bugs in one file are at least 10 lines apart; item_003 and item_004 use 2+ files.
8. oracle: a perfect findings set scores strict recall and precision 1.0. Moved to start exactly at
   ``line_end + 4`` (same length) it has no strict hit, and moved to end exactly at
   ``line_start - 4`` it has none either.
9. tests: original commit 0 failed / 0 errors; planted version 0 errors (collection, import or
   setup); results agree with ``meta.json``; each ``ineffective_test`` bug's own test passes.
10. giveaways: no hint words or new comments in ``replace`` text.
11. secrets: no provider-format token in any ``replace`` text; a ``hardcoded_secret`` plant adds a
    string literal.
12. extra_kinds: every bug with ``extra_accepted_kinds`` has a row in
    ``docs/decisions/0005-per-bug-accepted-kinds.md`` naming the item, the bug and each extra kind.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import tokenize
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from eval.build_dataset import (
    PlantError,
    PlantResult,
    Recipe,
    accepted_kinds_for,
    all_item_ids,
    apply_plants,
    candidate_entry,
    changed_files_at,
    dataset_dir,
    diff_regions,
    enclosing_test,
    load_recipe,
    new_worktree,
    parent_of,
    planted_line_ranges,
    run_tests,
    split_lines,
    upstream_diff,
)
from eval.ground_truth import (
    GroundTruth,
    changed_py_file_count,
    default_accepted_kinds,
    load_item,
    parse_diff,
    size_bucket,
)
from eval.repos import (
    apply_diff,
    clone_dir,
    diff_file_changes,
    get_repo,
    item_files,
    remove_worktree,
    snapshot_files,
)
from eval.score import changed_lines, score_config
from reviewer.config import repo_root
from reviewer.schemas import Finding

MIN_BUG_GAP = 10  # lines between bugs in the same file
MIN_FILES = {"item_003": 2, "item_004": 2}
PY_FILE_RANGE = {"item_003": (4, 5)}  # stage 4's timing test needs 4 or 5 .py files
ORACLE_SHIFT = 4  # one more than LINE_TOLERANCE in eval/score.py
EXTRA_KINDS_RECORD = PurePosixPath("docs/decisions/0005-per-bug-accepted-kinds.md")

GIVEAWAY_WORDS = (
    "bug", "planted", "inject", "vuln", "fixme", "todo", "xxx", "hack", "oops", "wrong",
    "insecure",
)  # fmt: skip

# Provider token formats that GitHub push protection / secret scanning recognizes.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "stripe": re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}"),
    "stripe_prefix": re.compile(r"\b(?:sk_live_|sk_test_|rk_live_|rk_test_|rk_)"),
    "slack": re.compile(r"\bxox[abposr]-|hooks\.slack\.com/"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"),
    "openai": re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"),
    "anthropic": re.compile(r"\bsk-ant-"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "gitlab": re.compile(r"\bglpat-"),
    "npm": re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    "pypi": re.compile(r"\bpypi-AgE"),
    "huggingface": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "sendgrid": re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"),
    "twilio": re.compile(r"\bSK[0-9a-fA-F]{32}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}"),
}
_STRING_LITERAL = re.compile(r"""(?:[rRbBuUfF]{0,2})("[^"\n]{8,}"|'[^'\n]{8,}')""")


@dataclass
class CheckResult:
    number: int
    name: str
    ok: bool | None  # None: skipped
    detail: str = ""

    def line(self, item_id: str) -> str:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[self.ok]
        return f"{item_id}  {self.number:>2} {self.name:<12} {status}  {self.detail}".rstrip()


# --- Pure checks (unit-tested on synthetic repositories) --------------------------------------


def added_lines(diff_text: str) -> dict[str, set[int]]:
    """Target line numbers of added lines, per non-deleted file."""
    out: dict[str, set[int]] = {}
    for pf in parse_diff(diff_text):
        if pf.is_removed_file:
            continue
        lines = out.setdefault(pf.path, set())
        for hunk in pf:
            lines.update(ln.target_line_no for ln in hunk if ln.is_added)
    return out


def check_plants_on_added_lines(
    recipe: Recipe, plant: PlantResult, upstream: str, pr_diff: str, gt: GroundTruth
) -> list[str]:
    """Check 5. Returns problems (empty when the check passes)."""
    problems = []
    up_added = added_lines(upstream)
    for bug in recipe.bugs:
        span = plant.spans[bug.bug_id]
        need = set(range(span.first_line, span.last_line + 1))
        missing = sorted(need - up_added.get(bug.file, set()))
        if missing:
            problems.append(f"{bug.bug_id}: find lines {missing} were not added by the commit")
    pr_added = added_lines(pr_diff)
    pr_changed = changed_lines(pr_diff)
    for b in gt.bugs:
        rng = set(range(b.line_start, b.line_end + 1))
        not_added = sorted(rng - pr_added.get(b.file, set()))
        not_changed = sorted(rng - pr_changed.get(b.file, set()))
        if not_added or not_changed:
            problems.append(
                f"{b.bug_id}: lines {sorted(set(not_added) | set(not_changed))} of "
                f"{b.file} are not added lines in pr.diff"
            )
    return problems


def check_plant_real(recipe: Recipe, commit_files: dict[str, bytes], gt: GroundTruth) -> list[str]:
    """Check 6: each plant changes its file, and the ground-truth range equals the recomputed
    range of differing lines; the planted lines differ from the original lines."""
    problems = []
    try:
        plant = apply_plants(commit_files, recipe.bugs)
    except PlantError as e:
        return [str(e)]
    gt_bugs = {b.bug_id: b for b in gt.bugs}
    for bug in recipe.bugs:
        # Each bug alone must change its file, and its range must not depend on the others.
        try:
            alone = apply_plants(commit_files, [bug])
        except PlantError as e:
            problems.append(str(e))
            continue
        if alone.planted[bug.file] == alone.original[bug.file]:
            problems.append(f"{bug.bug_id}: planted file equals the original")
        start, end = plant.ranges[bug.bug_id]
        g = gt_bugs.get(bug.bug_id)
        if g is None or (g.line_start, g.line_end) != (start, end):
            got = None if g is None else (g.line_start, g.line_end)
            problems.append(f"{bug.bug_id}: ground truth range {got} != recomputed {(start, end)}")
            continue
        # The bug alone: its differing regions must hold different text and span exactly the
        # ground-truth range (so the range does not depend on the other bugs either).
        regions = diff_regions(alone.original[bug.file], alone.planted[bug.file])
        old_lines = split_lines(alone.original[bug.file])
        new_lines = split_lines(alone.planted[bug.file])
        if any(old_lines[i1:i2] == new_lines[j1:j2] for i1, i2, j1, j2 in regions):
            problems.append(f"{bug.bug_id}: a changed region has identical text")
        alone_range = alone.ranges[bug.bug_id]
        shift = start - alone_range[0]
        if (alone_range[0] + shift, alone_range[1] + shift) != (start, end):
            problems.append(f"{bug.bug_id}: range alone {alone_range} is not {(start, end)}")
    # The recomputation itself: a fresh line diff per file must give the same ranges.
    for path, text in plant.planted.items():
        spans = [s for s in plant.spans.values() if s.file == path]
        if planted_line_ranges(plant.original[path], text, spans) != {
            s.bug_id: plant.ranges[s.bug_id] for s in spans
        }:
            problems.append(f"{path}: line ranges are not reproducible")
    return problems


def _comments(source: str) -> Counter[str] | None:
    try:
        toks = tokenize.generate_tokens(io.StringIO(source).readline)
        return Counter(t.string.strip() for t in toks if t.type == tokenize.COMMENT)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None


def check_giveaways(recipe: Recipe, plant: PlantResult | None = None) -> list[str]:
    """Check 10: hint words in ``replace`` (unless already in ``find``) and new comments."""
    problems = []
    for bug in recipe.bugs:
        rep, fnd = bug.replace.lower(), bug.find.lower()
        words = [w for w in GIVEAWAY_WORDS if w in rep and w not in fnd]
        if words:
            problems.append(f"{bug.bug_id}: replace contains {words}")
        find_comment_lines = {
            ln.strip() for ln in bug.find.splitlines() if ln.strip().startswith("#")
        }
        new = [
            ln.strip()
            for ln in bug.replace.splitlines()
            if ln.strip().startswith("#") and ln.strip() not in find_comment_lines
        ]
        if new:
            problems.append(f"{bug.bug_id}: replace adds comment line(s) {new}")
        if "#" in bug.replace and bug.replace.count("#") > bug.find.count("#"):
            problems.append(f"{bug.bug_id}: replace adds a '#' (possible inline comment)")
    if plant is not None:
        for path, planted in plant.planted.items():
            if PurePosixPath(path).suffix != ".py":
                continue
            before, after = _comments(plant.original[path]), _comments(planted)
            if before is None or after is None:
                problems.append(f"{path}: cannot tokenize to compare comments")
            elif after - before:
                problems.append(f"{path}: planted file has new comments {list(after - before)}")
    return problems


def check_secrets(recipe: Recipe) -> list[str]:
    """Check 11: no provider-format token in any ``replace`` (that its ``find`` did not already
    have); a ``hardcoded_secret`` plant must add a string literal of 8+ characters."""
    problems = []
    for bug in recipe.bugs:
        for name, pat in SECRET_PATTERNS.items():
            new = set(pat.findall(bug.replace)) - set(pat.findall(bug.find))
            if new:
                problems.append(f"{bug.bug_id}: replace matches the {name} token format")
        if bug.kind == "hardcoded_secret":
            lits = set(_STRING_LITERAL.findall(bug.replace)) - set(
                _STRING_LITERAL.findall(bug.find)
            )
            if not lits:
                problems.append(f"{bug.bug_id}: hardcoded_secret plant adds no string literal")
    return problems


def check_separation(gt: GroundTruth) -> list[str]:
    """Check 7."""
    problems = []
    by_file: dict[str, list] = {}
    for b in gt.bugs:
        by_file.setdefault(b.file, []).append(b)
    for path, bugs in by_file.items():
        bugs.sort(key=lambda b: b.line_start)
        for a, b in zip(bugs, bugs[1:], strict=False):
            gap = b.line_start - a.line_end
            if gap < MIN_BUG_GAP:
                problems.append(
                    f"{a.bug_id} and {b.bug_id} in {path} are {gap} lines apart (< {MIN_BUG_GAP})"
                )
    need = MIN_FILES.get(gt.item_id, 0)
    if len(by_file) < need:
        problems.append(f"bugs are in {len(by_file)} file(s), need at least {need}")
    return problems


type Placement = Literal["exact", "after", "before"]


def oracle_range(line_start: int, line_end: int, placement: Placement) -> tuple[int, int] | None:
    """The oracle finding's range for a bug at ``line_start``..``line_end``.

    ``exact``: the bug's range. ``after``: the same length, starting exactly at
    ``line_end + ORACLE_SHIFT``. ``before``: the same length, ending exactly at
    ``line_start - ORACLE_SHIFT``; the start is clamped to line 1, and None is returned when even
    the end would be before line 1 (no finding can be placed there).
    """
    length = line_end - line_start
    if placement == "exact":
        return line_start, line_end
    if placement == "after":
        start = line_end + ORACLE_SHIFT
        return start, start + length
    end = line_start - ORACLE_SHIFT
    if end < 1:
        return None
    return max(1, end - length), end


def oracle_findings(
    gt: GroundTruth, planted_texts: dict[str, str], placement: Placement = "exact"
) -> list:
    out = []
    for b in gt.bugs:
        rng = oracle_range(b.line_start, b.line_end, placement)
        if rng is None:
            continue
        lines = split_lines(planted_texts[b.file])
        evidence = "".join(lines[b.line_start - 1 : b.line_end]) or lines[b.line_start - 1]
        out.append(
            Finding(
                file=b.file,
                line_start=rng[0],
                line_end=rng[1],
                category=b.kind,
                severity="high",
                confidence=1.0,
                title=f"oracle {b.bug_id}",
                explanation=b.description,
                evidence=evidence,
                agent="oracle",
                model="oracle",
            ).model_dump()
        )
    return out


def oracle_score(item_dir: Path, gt: GroundTruth, findings: list) -> dict[str, Any]:
    """Score ``findings`` for one item with ``eval.score.score_config`` in a temp folder."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = Path(tmp) / "dataset"
        shutil.copytree(item_dir, ds / gt.item_id)
        res = Path(tmp) / "results" / "oracle"
        res.mkdir(parents=True)
        payload = {"item_id": gt.item_id, "config": "oracle", "findings": findings}
        (res / f"{gt.item_id}.json").write_bytes(json.dumps(payload).encode("utf-8"))
        return score_config(ds, res)


def check_oracle(item_dir: Path, gt: GroundTruth, planted_texts: dict[str, str]) -> list[str]:
    """Check 8."""
    if gt.is_clean:
        report = oracle_score(item_dir, gt, [])
        s = report["aggregates"]["strict"]["all"]
        return [] if s["fp"] == 0 and s["tp"] == 0 else [f"clean item scored {s}"]
    problems = []
    report = oracle_score(item_dir, gt, oracle_findings(gt, planted_texts))
    s = report["aggregates"]["strict"]["all"]
    if s["recall"] != 1.0 or s["precision"] != 1.0:
        problems.append(f"oracle recall={s['recall']} precision={s['precision']} (need 1.0)")
    for placement, where in (
        ("after", f"to start at line_end + {ORACLE_SHIFT}"),
        ("before", f"to end at line_start - {ORACLE_SHIFT}"),
    ):
        moved = oracle_score(item_dir, gt, oracle_findings(gt, planted_texts, placement))
        hits = moved["aggregates"]["strict"]["all"]["tp"]
        if hits:
            problems.append(f"oracle moved {where} still has {hits} strict hit(s)")
    return problems


def check_accepted_kinds(recipe: Recipe, gt: GroundTruth) -> list[str]:
    """Part of check 1: ``accepted_kinds`` = default kinds + the recipe's extra kinds."""
    problems = []
    recipe_bugs = {b.bug_id: b for b in recipe.bugs}
    for b in gt.bugs:
        rb = recipe_bugs.get(b.bug_id)
        if rb is None:
            continue  # reported by the bug list comparison
        want = list(dict.fromkeys([*default_accepted_kinds(rb.kind), *rb.extra_accepted_kinds]))
        if "other" in want or b.accepted_kinds != want:
            problems.append(f"{b.bug_id}: accepted_kinds {b.accepted_kinds} != expected {want}")
        if accepted_kinds_for(rb) != want:
            problems.append(f"{b.bug_id}: build_dataset.accepted_kinds_for disagrees with {want}")
    return problems


def documented_extra_kinds(record_text: str) -> dict[tuple[str, str], set[str]]:
    """Rows of the per-bug table in decision 0005: ``| item_id | bug_id | kinds | reason |``,
    where ``kinds`` is a comma-separated list (backticks allowed)."""
    out: dict[tuple[str, str], set[str]] = {}
    for line in record_text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip().strip("`").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not re.fullmatch(r"item_\d{3}", cells[0]):
            continue
        kinds = {k.strip().strip("`").strip() for k in cells[2].split(",")}
        out.setdefault((cells[0], cells[1]), set()).update(k for k in kinds if k)
    return out


def check_extra_kinds_documented(recipe: Recipe, record_text: str | None) -> list[str]:
    """Check 12: each bug with extra accepted kinds has a row in decision 0005 that names every
    one of its extra kinds."""
    bugs = [b for b in recipe.bugs if b.extra_accepted_kinds]
    if not bugs:
        return []
    if record_text is None:
        return [f"{EXTRA_KINDS_RECORD} is missing"]
    rows = documented_extra_kinds(record_text)
    problems = []
    for b in bugs:
        documented = rows.get((recipe.item_id, b.bug_id))
        if documented is None:
            problems.append(f"{b.bug_id}: extra kinds not listed in {EXTRA_KINDS_RECORD}")
            continue
        missing = [k for k in b.extra_accepted_kinds if k not in documented]
        if missing:
            problems.append(f"{b.bug_id}: extra kinds {missing} not listed in {EXTRA_KINDS_RECORD}")
    return problems


def read_extra_kinds_record() -> str | None:
    path = repo_root() / EXTRA_KINDS_RECORD
    return path.read_bytes().decode("utf-8") if path.is_file() else None


# --- The item check ---------------------------------------------------------------------------


def _counts(t: dict) -> tuple:
    return (t["passed"], t["failed"], t["errors"], t["skipped"])


class ItemChecker:
    def __init__(self, item_id: str, run_sandbox: bool = True) -> None:
        self.item_id = item_id
        self.run_sandbox = run_sandbox
        self.item_dir = dataset_dir() / item_id
        self.results: list[CheckResult] = []

    def add(self, number: int, name: str, fn: Callable[[], list[str] | None]) -> None:
        try:
            problems = fn()
        except Exception as e:  # a crashing check is a failing check
            self.results.append(CheckResult(number, name, False, f"{type(e).__name__}: {e}"))
            return
        if problems is None:
            self.results.append(CheckResult(number, name, None, "skipped"))
        else:
            self.results.append(CheckResult(number, name, not problems, "; ".join(problems)))

    def run(self) -> list[CheckResult]:
        self.recipe = load_recipe(self.item_id)
        self.spec = get_repo(self.recipe.repo)
        self.repo = clone_dir(self.recipe.repo)
        self.parent = parent_of(self.repo, self.recipe.commit)
        self.commit_files = changed_files_at(self.repo, self.parent, self.recipe.commit)
        self.upstream = upstream_diff(self.repo, self.parent, self.recipe.commit)
        self.plant = apply_plants(self.commit_files, self.recipe.bugs)
        self.gt, self.pr_diff = load_item(self.item_dir)
        self.meta = json.loads((self.item_dir / "meta.json").read_bytes())

        self.add(1, "schema", self.c1_schema)
        self.add(2, "eligibility", self.c2_eligibility)
        self.add(3, "bucket", self.c3_bucket)
        self.add(4, "diff", self.c4_diff)
        self.add(
            5,
            "added_lines",
            lambda: check_plants_on_added_lines(
                self.recipe, self.plant, self.upstream, self.pr_diff, self.gt
            ),
        )
        self.add(6, "plant_real", lambda: check_plant_real(self.recipe, self.commit_files, self.gt))
        self.add(7, "separation", lambda: check_separation(self.gt))
        self.add(8, "oracle", lambda: check_oracle(self.item_dir, self.gt, self.planted_texts()))
        self.add(9, "tests", self.c9_tests)
        self.add(10, "giveaways", lambda: check_giveaways(self.recipe, self.plant))
        self.add(11, "secrets", lambda: check_secrets(self.recipe))
        self.add(
            12,
            "extra_kinds",
            lambda: check_extra_kinds_documented(self.recipe, read_extra_kinds_record()),
        )
        return self.results

    def planted_texts(self) -> dict[str, str]:
        texts = {p: b.decode("utf-8", "replace") for p, b in self.commit_files.items()}
        texts.update(self.plant.planted)
        return texts

    def c1_schema(self) -> list[str]:
        r, gt, m = self.recipe, self.gt, self.meta
        p = []
        if gt.item_id != r.item_id or m.get("item_id") != r.item_id:
            p.append("item_id differs between recipe, ground truth and meta")
        if gt.source_repo != self.spec.slug:
            p.append(f"source_repo {gt.source_repo!r} != {self.spec.slug!r}")
        if gt.base_commit != self.parent or m.get("parent") != self.parent:
            p.append(f"base_commit/parent is not the commit's parent {self.parent}")
        if m.get("upstream_commit") != r.commit or m.get("repo") != r.repo:
            p.append("meta upstream_commit/repo differ from the recipe")
        if gt.is_clean != r.is_clean:
            p.append("is_clean differs")
        want = [(b.bug_id, b.kind, b.file, b.description) for b in r.bugs]
        got = [(b.bug_id, b.kind, b.file, b.description) for b in gt.bugs]
        if want != got:
            p.append(f"bugs differ: recipe {want} vs ground truth {got}")
        p.extend(check_accepted_kinds(r, gt))
        return p

    def c2_eligibility(self) -> list[str]:
        found = candidate_entry(self.recipe.repo, self.recipe.commit)
        if found is None:
            return [f"{self.recipe.commit[:12]} is not in eval/candidates/{self.recipe.repo}.json"]
        bucket, entry = found
        p = []
        if bucket != self.recipe.expected_bucket:
            p.append(f"indexed in bucket {bucket}, expected {self.recipe.expected_bucket}")
        since = self.spec.eligible_since
        if since is not None and date.fromisoformat(entry["date"][:10]) < since:
            p.append(f"commit date {entry['date'][:10]} is before eligible_since {since}")
        if entry["sweep"]:
            p.append("commit is tagged sweep")
        if entry["parent"] != self.parent:
            p.append("indexed parent differs")
        return p

    def c3_bucket(self) -> list[str]:
        p = []
        bucket = size_bucket(self.pr_diff)
        if bucket != self.recipe.expected_bucket:
            p.append(f"size_bucket(pr.diff) = {bucket}, expected {self.recipe.expected_bucket}")
        n = changed_py_file_count(self.pr_diff)
        if self.item_id in PY_FILE_RANGE:
            lo, hi = PY_FILE_RANGE[self.item_id]
            if not lo <= n <= hi:
                p.append(f"{n} changed .py files, need {lo} to {hi}")
        if self.meta.get("bucket") != bucket or self.meta.get("py_files_changed") != n:
            p.append("meta bucket/py_files_changed disagree with pr.diff")
        return p

    def c4_diff(self) -> list[str]:
        p = []
        wt = new_worktree(self.repo, self.gt.base_commit, f"check-{self.item_id}")
        try:
            diff_bytes = self.pr_diff.encode("utf-8")
            apply_diff(wt, diff_bytes, check_only=True)
            apply_diff(wt, diff_bytes)
            written, removed = diff_file_changes(self.pr_diff)
            expected = dict(self.commit_files)
            expected.update({k: v.encode("utf-8") for k, v in self.plant.planted.items()})
            if written != set(expected):
                p.append(f"pr.diff writes {sorted(written ^ set(expected))} unexpectedly")
            for path, content in sorted(expected.items()):
                f = wt / PurePosixPath(path)
                if not f.is_file() or f.read_bytes() != content:
                    p.append(f"{path}: applied result differs from the planted file")
            for path in sorted(removed):
                if (wt / PurePosixPath(path)).exists():
                    p.append(f"{path}: should be removed by pr.diff")
        finally:
            remove_worktree(self.repo, wt)
            shutil.rmtree(wt, ignore_errors=True)
        if self.recipe.is_clean and self.pr_diff != self.upstream:
            p.append("clean item's pr.diff is not byte-identical to the upstream diff")
        if not self.recipe.is_clean and self.pr_diff == self.upstream:
            p.append("planted item's pr.diff equals the upstream diff")
        return p

    def c9_tests(self) -> list[str] | None:
        if not self.run_sandbox:
            return None
        p = []
        name = self.recipe.repo
        orig = run_tests(name, snapshot_files(name, self.recipe.commit))
        planted = run_tests(name, item_files(self.item_id))
        if orig["timed_out"] or orig["failed"] or orig["errors"] or not orig["passed"]:
            p.append(f"original: {orig['summary'] or 'no pytest summary'}")
        if orig["exit_code"] != 0:
            p.append(f"original exit code {orig['exit_code']}")
        if planted["timed_out"] or planted["errors"] or not planted["passed"]:
            p.append(f"planted: {planted['summary'] or 'no pytest summary'}")
        if planted["exit_code"] not in (0, 1):  # 2+: interrupted, usage or internal error
            p.append(f"planted exit code {planted['exit_code']} (collection or import error?)")
        if self.recipe.is_clean and (planted["failed"] or planted["exit_code"] != 0):
            p.append(f"clean item fails tests: {planted['summary']}")
        for label, fresh in (("tests_original", orig), ("tests_planted", planted)):
            if _counts(fresh) != _counts(self.meta[label]):
                p.append(
                    f"{label} in meta.json {_counts(self.meta[label])} != now {_counts(fresh)}"
                )
        for bug in self.recipe.bugs:
            if bug.kind != "ineffective_test":
                continue
            node = enclosing_test(
                self.plant.planted[bug.file], bug.file, self.gt_bug(bug).line_start
            )
            if node is None:
                p.append(f"{bug.bug_id}: no test function encloses the plant")
                continue
            own = run_tests(name, item_files(self.item_id), [node])
            if own["exit_code"] != 0 or own["failed"] or own["errors"] or not own["passed"]:
                p.append(f"{bug.bug_id}: own test {node} does not pass: {own['summary']}")
        return p

    def gt_bug(self, bug):
        return next(b for b in self.gt.bugs if b.bug_id == bug.bug_id)


def check_item(item_id: str, run_sandbox: bool = True) -> list[CheckResult]:
    try:
        return ItemChecker(item_id, run_sandbox).run()
    except Exception as e:
        return [CheckResult(0, "load", False, f"{type(e).__name__}: {e}")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.check_dataset", description=__doc__)
    parser.add_argument("items", nargs="*", help="item ids (default: every recipe)")
    parser.add_argument("--no-tests", action="store_true", help="skip check 9 (counts as SKIP)")
    args = parser.parse_args(argv)
    items = args.items or all_item_ids()
    ok = True
    for item_id in items:
        for r in check_item(item_id, run_sandbox=not args.no_tests):
            print(r.line(item_id), flush=True)
            ok = ok and r.ok is True
    print("ALL PASS" if ok else "SOME CHECKS FAILED OR SKIPPED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
