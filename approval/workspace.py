"""Check out the pull request's head commit for the review (D2: only ``git`` touches it on the
host; nothing from the PR runs here)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

GitRunner = Callable[[Sequence[str], Path], None]


def run_git(args: Sequence[str], cwd: Path) -> None:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip().splitlines()[-1:]}")


def checkout_head(
    clone_url: str, head_sha: str, folder: Path, diff_text: str, runner: GitRunner = run_git
) -> tuple[Path, Path]:
    """``folder/repo`` is a shallow checkout of ``head_sha``; ``folder/pr.diff`` is the diff.
    Returns ``(workspace, diff_path)``. The clone URL carries no credentials."""
    repo_dir = folder / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        runner(["init", "-q"], repo_dir)
        runner(["config", "core.autocrlf", "false"], repo_dir)
        runner(["config", "core.symlinks", "false"], repo_dir)
        runner(["remote", "add", "origin", clone_url], repo_dir)
        runner(["fetch", "-q", "--depth", "1", "origin", head_sha], repo_dir)
        runner(["checkout", "-q", "--detach", "FETCH_HEAD"], repo_dir)
    diff_path = folder / "pr.diff"
    diff_path.write_bytes(diff_text.encode("utf-8"))
    return repo_dir, diff_path
