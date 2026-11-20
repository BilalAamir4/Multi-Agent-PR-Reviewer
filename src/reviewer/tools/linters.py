"""ruff and bandit on the PR's changed Python files (stage 1A, D2a: host, read-only).

Both ignore the repository's own configuration: ruff runs with ``--isolated``, and bandit only
looks for a ``.bandit`` file by walking its targets, which here are files, never directories.
Neither imports or executes the code it checks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from reviewer.tools.context import ReviewContext, ToolError
from reviewer.tools.files import resolve_repo_path

LINT_TIMEOUT_S = 60
RUFF_SELECT = "E,F,B"
BANDIT_SKIP = "B101"  # "assert used" fires on every test assert


def lintable_files(ctx: ReviewContext) -> list[str]:
    """Changed ``.py`` files that exist after the PR (added, modified or renamed)."""
    return sorted(f.path for f in ctx.files if f.is_python and f.status != "deleted")


def select_files(ctx: ReviewContext, paths: Sequence[str] | None) -> list[str]:
    """``paths`` checked against the changed ``.py`` files (all of them when None)."""
    allowed = lintable_files(ctx)
    if paths is None:
        chosen = allowed
    else:
        chosen = []
        for p in paths:
            clean = p[2:] if p.startswith("./") else p
            if clean not in allowed:
                raise ToolError(
                    f"{p!r} is not a changed Python file; allowed: {', '.join(allowed) or 'none'}"
                )
            chosen.append(clean)
    for p in chosen:
        _, target = resolve_repo_path(ctx, p)
        if not target.is_file():
            raise ToolError(f"{p!r} does not exist in the workspace")
    return list(dict.fromkeys(chosen))


def _relative(ctx: ReviewContext, filename: str) -> str:
    p = Path(filename)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(ctx.workspace).as_posix()
        except ValueError:
            return p.as_posix()
    return str(PurePosixPath(filename.replace("\\", "/")))


def _run(ctx: ReviewContext, name: str, cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            cmd, cwd=ctx.workspace, capture_output=True, shell=False, timeout=LINT_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as e:
        raise ToolError(f"{name} timed out after {LINT_TIMEOUT_S} seconds") from e


def _stderr_tail(res: subprocess.CompletedProcess[bytes]) -> str:
    lines = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else f"exit code {res.returncode}"


def run_ruff(ctx: ReviewContext, paths: Sequence[str] | None = None) -> str:
    """ruff (rules E, F, B) on changed ``.py`` files, as ``path:line: CODE message``."""
    try:
        files = select_files(ctx, paths)
        if not files:
            return "No changed Python files to check."
        cmd = [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--select",
            RUFF_SELECT,
            "--output-format",
            "json",
            "--no-cache",
            "--",
            *files,
        ]
        res = _run(ctx, "ruff", cmd)
        if res.returncode not in (0, 1):
            raise ToolError(f"ruff failed: {_stderr_tail(res)}")
        diags = json.loads(res.stdout or b"[]")
    except ToolError as e:
        return f"ERROR: {e}"
    except ValueError as e:
        return f"ERROR: cannot read ruff output: {e}"
    rows = sorted(
        (
            _relative(ctx, d["filename"]),
            d["location"]["row"],
            d["location"]["column"],
            d.get("code") or "syntax-error",
            d["message"],
        )
        for d in diags
    )
    head = f"ruff ({RUFF_SELECT}) on {len(files)} file(s): "
    if not rows:
        return head + "no issues."
    lines = [head + f"{len(rows)} issue(s)"]
    lines += [f"{path}:{row}: {code} {msg}" for path, row, _, code, msg in rows]
    return "\n".join(lines)


def run_bandit(ctx: ReviewContext, paths: Sequence[str] | None = None) -> str:
    """bandit (all tests but B101) on changed ``.py`` files, as
    ``path:line: TEST_ID severity/confidence: message``."""
    try:
        files = select_files(ctx, paths)
        if not files:
            return "No changed Python files to check."
        cmd = [
            sys.executable,
            "-m",
            "bandit",
            "-f",
            "json",
            "-q",
            "--skip",
            BANDIT_SKIP,
            "--",
            *files,
        ]
        res = _run(ctx, "bandit", cmd)
        if res.returncode not in (0, 1) or not res.stdout.strip():
            raise ToolError(f"bandit failed: {_stderr_tail(res)}")
        data = json.loads(res.stdout)
    except ToolError as e:
        return f"ERROR: {e}"
    except ValueError as e:
        return f"ERROR: cannot read bandit output: {e}"
    rows = sorted(
        (
            _relative(ctx, r["filename"]),
            r["line_number"],
            r["test_id"],
            f"{r['issue_severity']}/{r['issue_confidence']}",
            r["issue_text"],
        )
        for r in data.get("results", [])
    )
    head = f"bandit (skip {BANDIT_SKIP}) on {len(files)} file(s): "
    lines = [head + (f"{len(rows)} issue(s)" if rows else "no issues.")]
    lines += [f"{path}:{line}: {tid} {sc}: {msg}" for path, line, tid, sc, msg in rows]
    for err in data.get("errors", []):
        lines.append(
            f"ERROR: bandit could not check {_relative(ctx, err['filename'])}: {err['reason']}"
        )
    return "\n".join(lines)
