"""Dataset source repositories (stage 0C): ``eval/repos.yaml``, clones and sandboxed test runs.

Upstream repository code never runs on the host. Here only ``git`` runs on the host (clone, fetch,
``ls-tree``, ``cat-file``); tests run in the repository's sandbox image via ``run_in_sandbox`` (D2).
Clones live in ``eval/repos/<name>/`` (gitignored) with ``core.autocrlf false`` (D3).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from reviewer.config import repo_root
from reviewer.sandbox import SandboxResult, run_in_sandbox
from reviewer.schemas import Category
from reviewer.tools.context import SandboxSpec

log = logging.getLogger(__name__)

REPOS_YAML = Path(__file__).resolve().parent / "repos.yaml"
MAX_SNAPSHOT_FILE_BYTES = 1_000_000

License = Literal["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0"]
_SHA = r"^[0-9a-f]{40}$"
_URL = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\.git$")


# Local sandbox images with no upstream repository: ``url: null``, ``license: n/a``,
# ``image_commit: null`` and the image ``mapr-sandbox-<name>:local`` are allowed only for these
# names (stage 2A: the synthetic dev fixtures' image, decision 0008).
LOCAL_REPOS = frozenset({"dev_fixture"})


class RepoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(pattern=_URL.pattern)  # None only for LOCAL_REPOS
    license: License | Literal["n/a"]  # "n/a" only for LOCAL_REPOS
    default_branch: str | None  # None only for LOCAL_REPOS
    # the commit whose dependencies the image installs; None only for LOCAL_REPOS
    image_commit: str | None = Field(pattern=_SHA)
    image: str  # mapr-sandbox-<name>:<first 12 chars of image_commit> (or :local)
    pythonpath: list[str]  # paths under /work added to PYTHONPATH in the image
    test_command: list[str] = Field(min_length=1)
    test_timeout_s: PositiveInt
    # Kinds that are natural to plant here (owner's choice, stage 0C); the other D8 kinds are
    # plantable in every repository.
    recommended_plants: list[Category] = Field(default_factory=list)
    # Oldest commit date the image supports (stage 0C task 6); None means the whole index.
    eligible_since: date | None = None

    @property
    def is_local(self) -> bool:
        return self.url is None

    @property
    def slug(self) -> str | None:
        """``owner/name``, as in ``GroundTruth.source_repo``; None for a local image."""
        if self.url is None:
            return None
        m = _URL.match(self.url)
        assert m is not None
        return f"{m.group(1)}/{m.group(2)}"


class ReposFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repos: dict[str, RepoSpec]

    @model_validator(mode="after")
    def _check_images(self) -> Self:
        for name, spec in self.repos.items():
            local = (
                spec.url is None,
                spec.license == "n/a",
                spec.image_commit is None,
                spec.default_branch is None,
            )
            if any(local):
                if name not in LOCAL_REPOS:
                    raise ValueError(
                        f"{name}: url, license n/a, image_commit and default_branch may be "
                        f"null only for the local images {sorted(LOCAL_REPOS)}"
                    )
                if not all(local):
                    raise ValueError(
                        f"{name}: a local image needs url, image_commit and default_branch null "
                        "and license n/a"
                    )
                expected = f"mapr-sandbox-{name}:local"
            else:
                assert spec.image_commit is not None
                expected = f"mapr-sandbox-{name}:{spec.image_commit[:12]}"
            if spec.image != expected:
                raise ValueError(f"{name}: image must be {expected!r}, got {spec.image!r}")
            if "other" in spec.recommended_plants:
                raise ValueError(f"{name}: recommended_plants must not contain 'other'")
        return self


def load_repos(path: Path = REPOS_YAML) -> dict[str, RepoSpec]:
    data = yaml.safe_load(path.read_bytes().decode("utf-8"))
    return ReposFile.model_validate(data).repos


@cache
def _default_repos() -> dict[str, RepoSpec]:
    return load_repos()


def dataset_repos() -> list[str]:
    """Names of the dataset source repositories (every entry but the local images)."""
    return [name for name, spec in _default_repos().items() if not spec.is_local]


def get_repo(name: str) -> RepoSpec:
    repos = _default_repos()
    if name not in repos:
        raise KeyError(f"unknown repository {name!r}; known: {sorted(repos)}")
    return repos[name]


def clone_dir(name: str) -> Path:
    return repo_root() / "eval" / "repos" / name


def _git(repo: Path | None, *args: str, input: bytes | None = None) -> bytes:
    cmd = ["git", *(["-C", str(repo)] if repo else []), *args]
    return subprocess.run(cmd, input=input, capture_output=True, check=True).stdout


def ensure_clone(name: str) -> Path:
    """Full clone of ``name`` with ``core.autocrlf false`` (set before any checkout), or a fetch.

    New clones are made with ``--no-checkout``: nothing reads the working tree, and some upstream
    paths are not valid on Windows.
    """
    spec = get_repo(name)
    if spec.url is None:
        raise ValueError(f"{name} is a local image with no repository to clone")
    path = clone_dir(name)
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(None, "clone", "--no-checkout", spec.url, str(path))
        _git(path, "config", "core.autocrlf", "false")
        return path
    _git(path, "config", "core.autocrlf", "false")
    promisor = subprocess.run(
        ["git", "-C", str(path), "config", "--get", "remote.origin.promisor"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if promisor == "true":
        raise RuntimeError(f"{path} is a partial clone; delete it and run ensure_clone again")
    _git(path, "fetch", "--prune", "origin")
    return path


@dataclass
class Snapshot:
    files: dict[str, bytes]
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def parse_ls_tree(text: bytes) -> list[tuple[str, str, str, int | None, str]]:
    """Parse ``git ls-tree -r -l -z``: (mode, type, oid, size, path)."""
    out = []
    for rec in text.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, oid, size = meta.decode().split()
        out.append((mode, typ, oid, None if size == "-" else int(size), path.decode("utf-8")))
    return out


def take_snapshot(name: str, commit: str) -> Snapshot:
    """Every tracked file at ``commit``, read from git objects (no checkout)."""
    repo = clone_dir(name)
    entries = parse_ls_tree(_git(repo, "ls-tree", "-r", "-l", "-z", commit))
    snap = Snapshot(files={})
    wanted: list[tuple[str, str]] = []
    for mode, typ, oid, size, path in entries:
        if typ != "blob":
            snap.skipped.append((path, f"{typ} (submodule)"))
        elif mode == "120000":
            snap.skipped.append((path, "symlink"))
        elif size is not None and size > MAX_SNAPSHOT_FILE_BYTES:
            snap.skipped.append((path, f"{size} bytes > {MAX_SNAPSHOT_FILE_BYTES}"))
        else:
            wanted.append((oid, path))
    out = _git(repo, "cat-file", "--batch", input="".join(f"{o}\n" for o, _ in wanted).encode())
    pos = 0
    for oid, path in wanted:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].split()
        if len(header) != 3 or header[0].decode() != oid:
            raise RuntimeError(f"unexpected cat-file header for {path}: {out[pos:nl]!r}")
        size = int(header[2])
        snap.files[str(PurePosixPath(path))] = out[nl + 1 : nl + 1 + size]
        pos = nl + 1 + size + 1  # content is followed by a newline
    return snap


def snapshot_files(name: str, commit: str) -> dict[str, bytes]:
    """Every tracked file at ``commit`` as ``{posix_path: bytes}``; files over 1 MB, symlinks and
    submodules are skipped and logged (use ``take_snapshot`` to get the list)."""
    snap = take_snapshot(name, commit)
    for path, reason in snap.skipped:
        log.warning("%s@%s: skipped %s (%s)", name, commit[:12], path, reason)
    return snap.files


def sandbox_spec(name: str) -> SandboxSpec:
    """The ``run_tests`` sandbox for the ``eval/repos.yaml`` entry ``name`` (stage 2A)."""
    spec = get_repo(name)
    return SandboxSpec(
        image=spec.image,
        test_command=tuple(spec.test_command),
        pythonpath=tuple(spec.pythonpath),
        timeout_s=spec.test_timeout_s,
    )


def run_repo_tests(name: str, files: dict[str, bytes]) -> SandboxResult:
    """Run the repository's ``test_command`` on ``files`` in its sandbox image (D2)."""
    spec = get_repo(name)
    return run_in_sandbox(spec.image, files, spec.test_command, spec.test_timeout_s)


# --- Worktrees and dataset items (stage 0D) ---------------------------------------------------

DIFF_OPTIONS = ("--no-color", "--no-ext-diff", "--no-textconv", "-M")


def workspaces_dir() -> Path:
    """``eval/workspaces/`` (gitignored): worktrees for building, checking and reviewing items."""
    return repo_root() / "eval" / "workspaces"


def git_bytes(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    """Run ``git -C repo <args>`` and return stdout; raise with stderr on failure."""
    res = subprocess.run(
        ["git", "-C", str(repo), *args], input=input, capture_output=True, check=False
    )
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args[:3])} failed ({res.returncode}): {err}")
    return res.stdout


def add_worktree(repo: Path, path: Path, commit: str) -> Path:
    """Detached worktree of ``repo`` at ``commit`` with ``core.autocrlf false`` (D3).

    Checking out only writes files; no upstream code runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    git_bytes(repo, "worktree", "prune")
    git_bytes(
        repo,
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.eol=lf",
        "worktree",
        "add",
        "--detach",
        str(path),
        commit,
    )
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    """Remove a worktree made by ``add_worktree`` (and prune stale entries)."""
    if path.exists():
        git_bytes(repo, "worktree", "remove", "--force", str(path))
    git_bytes(repo, "worktree", "prune")


def _registered_worktrees(repo: Path) -> set[Path]:
    out = git_bytes(repo, "worktree", "list", "--porcelain", "-z").decode("utf-8")
    return {
        Path(rec[len("worktree ") :]).resolve()
        for rec in out.split("\0")
        if rec.startswith("worktree ")
    }


def apply_diff(worktree: Path, diff: bytes, *, check_only: bool = False) -> None:
    """``git apply`` (or ``git apply --check``) ``diff`` inside ``worktree``."""
    args = ["-c", "core.autocrlf=false", "apply", "--whitespace=nowarn"]
    if check_only:
        args.append("--check")
    git_bytes(worktree, *args, "-", input=diff)


def repo_name_for_slug(slug: str) -> str:
    """The ``eval/repos.yaml`` key whose URL is ``owner/name`` == ``slug``."""
    for name, spec in _default_repos().items():
        if spec.slug is not None and spec.slug == slug:
            return name
    raise KeyError(f"no repository in {REPOS_YAML.name} for {slug!r}")


def _load_item(item_id: str):
    from eval.ground_truth import load_item  # local import: ground_truth is independent

    return load_item(repo_root() / "eval" / "dataset" / item_id)


def materialize(item_id: str) -> Path:
    """Create ``eval/workspaces/<item_id>/``: a worktree at ``base_commit`` with ``pr.diff``
    applied, and return its path. An existing worktree for the item is replaced.

    Only ``git`` touches the clone; the files are for reading (stage 1 review tools) and for
    copying into the sandbox (``item_files``).
    """
    gt, diff_text = _load_item(item_id)
    name = repo_name_for_slug(gt.source_repo)
    repo = clone_dir(name)
    if not (repo / ".git").exists():
        ensure_clone(name)
    path = workspaces_dir() / item_id
    if path.exists():
        if path.resolve() not in _registered_worktrees(repo):
            raise RuntimeError(f"{path} exists and is not a worktree of {repo}; remove it first")
        remove_worktree(repo, path)
    add_worktree(repo, path, gt.base_commit)
    apply_diff(path, diff_text.encode("utf-8"))
    return path


def diff_file_changes(diff_text: str) -> tuple[set[str], set[str]]:
    """(paths written, paths removed) by a git diff, as POSIX strings. A rename removes its
    source path and writes its target path."""
    from eval.ground_truth import parse_diff

    written: set[str] = set()
    removed: set[str] = set()
    for pf in parse_diff(diff_text):
        source = pf.source_file[2:] if pf.source_file.startswith("a/") else pf.source_file
        if pf.is_removed_file:
            removed.add(str(PurePosixPath(source)))
            continue
        written.add(str(PurePosixPath(pf.path)))
        if not pf.is_added_file and source != pf.path:
            removed.add(str(PurePosixPath(source)))
    return written, removed - written


def item_sandbox(item_id: str) -> SandboxSpec:
    """The ``run_tests`` sandbox of a dataset item: its source repository's entry."""
    gt, _ = _load_item(item_id)
    return sandbox_spec(repo_name_for_slug(gt.source_repo))


def item_files(item_id: str) -> dict[str, bytes]:
    """Every file of the item's PR state (``base_commit`` plus ``pr.diff``) for sandbox runs:
    the tracked files at ``base_commit`` with the files ``pr.diff`` touches read from
    ``materialize(item_id)``."""
    gt, diff_text = _load_item(item_id)
    name = repo_name_for_slug(gt.source_repo)
    path = materialize(item_id)
    files = snapshot_files(name, gt.base_commit)
    written, removed = diff_file_changes(diff_text)
    for p in removed:
        files.pop(p, None)
    for p in written:
        files[p] = (path / PurePosixPath(p)).read_bytes()
    return files
