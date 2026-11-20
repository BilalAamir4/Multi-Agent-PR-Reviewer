"""Small synthetic git repositories for the stage 0D dataset tests (no network, no Docker)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from eval.build_dataset import BugRecipe, Built, Recipe, build_in_repo, ground_truth_for

BASE = """\
def total(values):
    return sum(values)
"""

# The commit adds `clamp` and a test for it; lines 4 to 9 of lib.py are added.
LIB_AT_COMMIT = """\
def total(values):
    return sum(values)


def clamp(value, low, high):
    if value is None:
        return None
    if value < low:
        return low
    return min(value, high)
"""

TEST_AT_COMMIT = """\
from lib import clamp


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    """A repo with a base commit and one commit that adds ``clamp`` and its test."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "lib.py").write_bytes(BASE.encode())
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    (repo / "lib.py").write_bytes(LIB_AT_COMMIT.encode())
    (repo / "test_lib.py").write_bytes(TEST_AT_COMMIT.encode())
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Add clamp")
    return repo, git(repo, "rev-parse", "HEAD")


def bug(**kw) -> BugRecipe:
    data = {
        "bug_id": "b1",
        "kind": "off_by_one",
        "file": "lib.py",
        "find": "if value < low:\n",
        "replace": "if value <= low:\n",
        "description": "Boundary is wrong.",
    }
    data.update(kw)
    return BugRecipe.model_validate(data)


def recipe(commit: str, bugs: list[BugRecipe], item_id: str = "item_900") -> Recipe:
    return Recipe(
        item_id=item_id,
        repo="demo",
        commit=commit,
        expected_bucket="small",
        is_clean=not bugs,
        bugs=bugs,
    )


def build(tmp_path: Path, repo: Path, r: Recipe) -> tuple[Built, object]:
    built = build_in_repo(r, repo, workspace_root=tmp_path / "workspaces")
    gt = ground_truth_for(r, "owner/demo", built.parent, built.plant.ranges)
    return built, gt
