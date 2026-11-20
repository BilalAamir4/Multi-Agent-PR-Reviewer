"""Reading files from the workspace, with path safety (stage 1A). Read-only."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from reviewer.tools.context import ReviewContext, ToolError

MAX_READ_LINES = 400
MAX_FILE_BYTES = 2_000_000
BINARY_SNIFF_BYTES = 8192
_DRIVE = re.compile(r"^[A-Za-z]:")


def _is_git_dir(segment: str) -> bool:
    # Windows ignores case and trailing dots and spaces in names: ".GIT" and ".git." are ".git".
    return segment.rstrip(". ").lower() == ".git"


def resolve_repo_path(ctx: ReviewContext, path: str) -> tuple[str, Path]:
    """Check a repository-relative POSIX ``path`` and return ``(clean_path, absolute_path)``.

    Rejects empty, absolute or drive paths, backslashes, colons, ``..`` segments, anything in
    ``.git``, and anything whose resolved path (following symlinks) is outside the workspace.
    """
    if not path or not path.strip():
        raise ToolError("path must not be empty")
    if "\\" in path:
        raise ToolError(f"{path!r}: use forward slashes, not backslashes")
    if path.startswith("/") or _DRIVE.match(path) or PureWindowsPath(path).anchor:
        raise ToolError(f"{path!r}: path must be relative to the repository root, not absolute")
    if ":" in path:
        raise ToolError(f"{path!r}: path must not contain ':'")
    clean = path[2:] if path.startswith("./") else path
    parts = PurePosixPath(clean).parts
    if ".." in clean.split("/"):
        raise ToolError(f"{path!r}: '..' segments are not allowed")
    if any(_is_git_dir(p) for p in parts):
        raise ToolError(f"{path!r}: files under .git are not readable")
    root = ctx.workspace.resolve()
    target = (root / PurePosixPath(clean)).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"{path!r} resolves outside the repository (symlink?)")
    if any(_is_git_dir(p) for p in target.relative_to(root).parts):
        raise ToolError(f"{path!r} resolves into .git")
    return str(PurePosixPath(clean)), target


def _read_text(target: Path, path: str) -> str:
    if not target.exists():
        raise ToolError(f"{path!r} does not exist in the repository after the PR")
    if not target.is_file():
        raise ToolError(f"{path!r} is not a file")
    size = target.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ToolError(f"{path!r} is {size} bytes (over {MAX_FILE_BYTES}); use search_code")
    data = target.read_bytes()
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        raise ToolError(f"{path!r} is a binary file")
    return data.decode("utf-8", errors="replace")


def split_file_lines(text: str) -> list[str]:
    """Lines split on ``\\n`` only, as git numbers them (``str.splitlines`` would also split on
    form feeds and other separators and shift the numbers)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln.removesuffix("\r") for ln in lines]


def read_file(
    ctx: ReviewContext, path: str, start_line: int = 1, end_line: int | None = None
) -> str:
    """Lines ``start_line``..``end_line`` of ``path`` after the PR, numbered, at most 400 per
    call. Returns ``ERROR: ...`` for a refused or unreadable path."""
    try:
        clean, target = resolve_repo_path(ctx, path)
        lines = split_file_lines(_read_text(target, clean))
    except ToolError as e:
        return f"ERROR: {e}"
    total = len(lines)
    if start_line < 1:
        return f"ERROR: start_line must be 1 or more, got {start_line}"
    if end_line is not None and end_line < start_line:
        return f"ERROR: end_line ({end_line}) is before start_line ({start_line})"
    if total == 0:
        return f"{clean} is empty."
    if start_line > total:
        return f"ERROR: start_line {start_line} is past the end of {clean} ({total} lines)"
    want_end = min(end_line if end_line is not None else total, total)
    end = min(want_end, start_line + MAX_READ_LINES - 1)
    width = len(str(end))
    out = [f"{clean}: lines {start_line}-{end} of {total}"]
    out.extend(f"{n:>{width}} | {lines[n - 1]}" for n in range(start_line, end + 1))
    if end < want_end:
        out.append(
            f"[clipped: read_file returns at most {MAX_READ_LINES} lines per call; "
            f"call again with start_line={end + 1}]"
        )
    return "\n".join(out)
