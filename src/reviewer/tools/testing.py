"""``run_tests``: pytest on the pull request's code in the repository's sandbox image (stage 2A;
D2, D2a, D2b, decision 0008).

The only review tool that executes PR code, so it runs only through ``run_in_sandbox``: the
workspace's files are copied into a fresh no-network container, never mounted. Which files: the
workspace's tracked files plus the files the diff writes, read from disk (the PR version), with
files over 1 MB, symlinks and submodules skipped, as ``eval.repos.snapshot_files`` and
``item_files`` do. A workspace that is not the root of a git work tree (the synthetic dev
fixtures) sends every regular file instead, minus caches and ``.git``.

The command is the sandbox spec's ``test_command`` with ``-x`` / ``--exitfirst`` removed, then
``--tb=short -rfE -q`` (each only if not already there), then the validated test IDs.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from docker.errors import ImageNotFound

from reviewer.sandbox import SandboxResult, SandboxUnavailable, get_client, run_in_sandbox
from reviewer.tools.context import ReviewContext, SandboxSpec, ToolError, limit_output
from reviewer.tools.files import resolve_repo_path

log = logging.getLogger(__name__)

MAX_TEST_IDS = 50
MAX_FILE_BYTES = 1_000_000  # as eval.repos.MAX_SNAPSHOT_FILE_BYTES
EXIT_FIRST = frozenset({"-x", "--exitfirst"})
EXTRA_ARGS = ("--tb=short", "-rfE", "-q")
SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", ".tox"}
)
NO_SANDBOX = "no test sandbox is configured for this review"
OUTPUT_TAIL_CHARS = 4000


@dataclass
class ToolNotes:
    """What the tools of one agent run report back to the agent's ``ReviewResult``.

    ``unavailable`` is set when a tool the agent needed could not run (D2b); the review then ends
    ``partial`` instead of ``complete``."""

    notes: list[str] = field(default_factory=list)
    unavailable: bool = False

    def add(self, note: str, *, unavailable: bool = False) -> None:
        if note not in self.notes:
            self.notes.append(note)
        self.unavailable = self.unavailable or unavailable


# --- Command and test IDs --------------------------------------------------------------------


def build_command(spec: SandboxSpec, test_ids: Sequence[str] = ()) -> list[str]:
    """``test_command`` without ``-x`` / ``--exitfirst``, plus ``--tb=short -rfE -q`` (those not
    already present), plus ``test_ids``."""
    cmd = [arg for arg in spec.test_command if arg not in EXIT_FIRST]
    cmd += [arg for arg in EXTRA_ARGS if arg not in cmd]
    return [*cmd, *test_ids]


def validate_test_id(ctx: ReviewContext, test_id: str) -> str:
    """Check one pytest node ID or path and return it with a leading ``./`` removed.

    Refused: empty, starting with ``-``, containing whitespace, ``..`` or a backslash, or whose
    path part (before the first ``::``) is not a file or directory in the workspace (the usual
    path rules of ``read_file`` apply: relative, no ``:``, nothing under ``.git``)."""
    if not isinstance(test_id, str) or not test_id:
        raise ToolError("test IDs must be non-empty strings")
    if test_id.startswith("-"):
        raise ToolError(f"{test_id!r}: test IDs must not start with '-' (options are not allowed)")
    if any(c.isspace() for c in test_id):
        raise ToolError(f"{test_id!r}: test IDs must not contain whitespace")
    if "\\" in test_id:
        raise ToolError(f"{test_id!r}: use forward slashes, not backslashes")
    if ".." in test_id:
        raise ToolError(f"{test_id!r}: '..' is not allowed")
    path, sep, rest = test_id.partition("::")
    if not path:
        raise ToolError(f"{test_id!r}: a test ID starts with a path, for example tests/test_x.py")
    if sep and not rest:
        raise ToolError(f"{test_id!r}: nothing after '::'")
    clean, target = resolve_repo_path(ctx, path)
    if not target.exists():
        raise ToolError(f"{test_id!r}: {clean!r} does not exist in the repository after the PR")
    return clean + sep + rest


def validate_test_ids(ctx: ReviewContext, test_ids: Sequence[str] | None) -> list[str]:
    if not test_ids:
        return []
    if isinstance(test_ids, str):
        raise ToolError("test_ids must be a list of strings")
    if len(test_ids) > MAX_TEST_IDS:
        raise ToolError(f"at most {MAX_TEST_IDS} test IDs per call, got {len(test_ids)}")
    return list(dict.fromkeys(validate_test_id(ctx, t) for t in test_ids))


# --- Files -----------------------------------------------------------------------------------


def _git_tracked(root: Path) -> set[str] | None:
    """Tracked regular files of ``root`` (``git ls-files -s``), or None when ``root`` is not the
    top of a git work tree. Symlinks (120000) and submodules (160000) are left out."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if top.returncode != 0:
        return None
    if Path(top.stdout.decode("utf-8").strip()).resolve() != root.resolve():
        return None
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "-z"], capture_output=True, check=True
    ).stdout
    tracked: set[str] = set()
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode = meta.split()[0].decode()
        name = path.decode("utf-8")
        if mode in ("120000", "160000"):
            log.info("run_tests: skipped %s (mode %s)", name, mode)
            continue
        tracked.add(str(PurePosixPath(name)))
    return tracked


def _walk(root: Path) -> set[str]:
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(root).as_posix()
            found.add(rel)
    return found


def workspace_files(ctx: ReviewContext) -> dict[str, bytes]:
    """The PR version of the workspace as ``{posix_path: bytes}`` (module docstring)."""
    root = ctx.workspace
    tracked = _git_tracked(root)
    if tracked is None:
        paths = _walk(root)
    else:
        paths = tracked | {f.path for f in ctx.files if f.status != "deleted"}
    files: dict[str, bytes] = {}
    for rel in sorted(paths):
        p = root / PurePosixPath(rel)
        if p.is_symlink() or not p.is_file():
            continue  # deleted by the PR, or a symlink
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            log.warning("run_tests: skipped %s (%d bytes > %d)", rel, size, MAX_FILE_BYTES)
            continue
        files[rel] = p.read_bytes()
    return files


# --- Output ----------------------------------------------------------------------------------

_SUMMARY = re.compile(
    r"^=*\s*(?P<text>(?:no tests ran|\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|"
    r"xpassed|warnings?))\b.*? in \d+(?:\.\d+)?s(?: \([0-9:]+\))?)\s*=*$"
)
_SECTION = re.compile(r"^=+ (?P<name>[A-Za-z ]+?) =+$")


def summary_line(stdout: str) -> str | None:
    """pytest's final summary (for example ``1 failed, 3 passed in 0.12s``), or None."""
    found = None
    for line in stdout.splitlines():
        m = _SUMMARY.match(line.strip())
        if m:
            found = m.group("text")
    return found


def failed_ids(stdout: str) -> tuple[list[str], list[str]]:
    """Failed and errored node IDs from the ``-rfE`` short summary (``FAILED id - msg``)."""
    failed: list[str] = []
    errors: list[str] = []
    for line in stdout.splitlines():
        for prefix, into in (("FAILED ", failed), ("ERROR ", errors)):
            if line.startswith(prefix):
                node = line[len(prefix) :].split(" - ", 1)[0].strip()
                if node and node not in into:
                    into.append(node)
    return failed, errors


def tracebacks(stdout: str) -> str:
    """The FAILURES and ERRORS sections of the output (``--tb=short``), without the short
    summary."""
    out: list[str] = []
    keep = False
    for line in stdout.splitlines():
        m = _SECTION.match(line.strip())
        if m:
            keep = m.group("name") in ("FAILURES", "ERRORS")
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) <= OUTPUT_TAIL_CHARS:
        return text
    return (
        f"[... {len(text) - OUTPUT_TAIL_CHARS} earlier characters left out]\n"
        + text[-OUTPUT_TAIL_CHARS:]
    )


def format_result(res: SandboxResult, timeout_s: int) -> str:
    summary = summary_line(res.stdout)
    failed, errors = failed_ids(res.stdout)
    lines: list[str] = []
    if res.timed_out:
        lines.append(f"ERROR: tests timed out after {timeout_s} s")
    else:
        lines.append(f"pytest exit code {res.exit_code}: {summary or 'no summary line'}")
    lines.append("Failed: " + (", ".join(failed) if failed else "none"))
    lines.append("Errors: " + (", ".join(errors) if errors else "none"))
    tb = tracebacks(res.stdout)
    if tb:
        lines += ["", tb]
    if res.timed_out or summary is None or res.exit_code not in (0, 1):
        output = _tail(res.stdout + ("\n" + res.stderr if res.stderr.strip() else ""))
        lines += ["", "Output:", output or "(none)"]
    return "\n".join(lines)


# --- The tool --------------------------------------------------------------------------------


def check_image(image: str) -> None:
    """Raise ``SandboxUnavailable`` when Docker cannot be reached (D2b) and ``ImageNotFound``
    when ``image`` is not built. (``run_in_sandbox`` alone reports a missing image as
    ``SandboxUnavailable``, because the docker client's HTTP errors are ``OSError``s.)"""
    client = get_client()
    try:
        client.images.get(image)
    finally:
        client.close()


def run_tests(
    ctx: ReviewContext, test_ids: Sequence[str] | None = None, notes: ToolNotes | None = None
) -> str:
    """Run pytest in the sandbox and format the result; ``ERROR: ...`` on any failure. Docker
    not running (D2b) or a missing image is recorded in ``notes`` as an unavailable tool."""
    notes = notes if notes is not None else ToolNotes()
    spec = ctx.sandbox
    if spec is None:
        notes.add(f"run_tests unavailable: {NO_SANDBOX}")
        return f"ERROR: {NO_SANDBOX}"
    try:
        ids = validate_test_ids(ctx, test_ids)
        command = build_command(spec, ids)
        files = workspace_files(ctx)
        check_image(spec.image)
        res = run_in_sandbox(spec.image, files, command, spec.timeout_s)
    except ToolError as e:
        return f"ERROR: {e}"
    except SandboxUnavailable as e:
        notes.add(f"sandbox unavailable: {e}", unavailable=True)
        return f"ERROR: {e}"
    except ImageNotFound:
        msg = f"sandbox image {spec.image} is not built"
        notes.add(f"sandbox unavailable: {msg}", unavailable=True)
        return f"ERROR: {msg}"
    return limit_output(format_result(res, spec.timeout_s))
