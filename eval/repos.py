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

log = logging.getLogger(__name__)

REPOS_YAML = Path(__file__).resolve().parent / "repos.yaml"
MAX_SNAPSHOT_FILE_BYTES = 1_000_000

License = Literal["MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0"]
_SHA = r"^[0-9a-f]{40}$"
_URL = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\.git$")


class RepoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(pattern=_URL.pattern)
    license: License
    default_branch: str
    image_commit: str = Field(pattern=_SHA)  # the commit whose dependencies the image installs
    image: str  # mapr-sandbox-<name>:<first 12 chars of image_commit>
    pythonpath: list[str]  # paths under /work added to PYTHONPATH in the image
    test_command: list[str] = Field(min_length=1)
    test_timeout_s: PositiveInt
    # Kinds that are natural to plant here (owner's choice, stage 0C); the other D8 kinds are
    # plantable in every repository.
    recommended_plants: list[Category] = Field(default_factory=list)
    # Oldest commit date the image supports (stage 0C task 6); None means the whole index.
    eligible_since: date | None = None

    @property
    def slug(self) -> str:
        """``owner/name``, as in ``GroundTruth.source_repo``."""
        m = _URL.match(self.url)
        assert m is not None
        return f"{m.group(1)}/{m.group(2)}"


class ReposFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repos: dict[str, RepoSpec]

    @model_validator(mode="after")
    def _check_images(self) -> Self:
        for name, spec in self.repos.items():
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


def run_repo_tests(name: str, files: dict[str, bytes]) -> SandboxResult:
    """Run the repository's ``test_command`` on ``files`` in its sandbox image (D2)."""
    spec = get_repo(name)
    return run_in_sandbox(spec.image, files, spec.test_command, spec.test_timeout_s)
