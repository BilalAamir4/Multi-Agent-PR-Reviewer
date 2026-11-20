"""The review context, the changed-file list and the numbered diff (stage 1A, D9, D9a)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from unidiff import PatchedFile, PatchSet
from unidiff.errors import UnidiffParseError

MAX_OUTPUT_CHARS = 12_000

type FileStatus = Literal["added", "modified", "deleted", "renamed"]


class ToolError(ValueError):
    """A request the tool refuses or cannot serve. The agent sees ``ERROR: <message>``."""


def error(message: str) -> str:
    return f"ERROR: {message}"


def limit_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Cut ``text`` to ``limit`` characters, ending with a note of how much was left out."""
    if len(text) <= limit:
        return text
    rest = len(text) - limit
    return f"{text[:limit]}\n[truncated: {rest} more characters; narrow the request]"


@dataclass(frozen=True)
class ChangedFile:
    path: str  # POSIX, new name for renames, old name for deletions
    status: FileStatus
    added: int
    removed: int
    is_python: bool
    old_path: str | None = None  # renames only


def _strip_prefix(name: str) -> str:
    return name[2:] if name.startswith(("a/", "b/")) else name


def _status(pf: PatchedFile) -> FileStatus:
    if pf.is_removed_file:
        return "deleted"
    if pf.is_added_file:
        return "added"
    if pf.is_rename:
        return "renamed"
    return "modified"


def parse_patch(diff_text: str) -> PatchSet:
    try:
        return PatchSet(diff_text)
    except UnidiffParseError as e:
        raise ToolError(f"cannot parse the diff: {e}") from e


def _changed_file(pf: PatchedFile) -> ChangedFile:
    status = _status(pf)
    path = str(PurePosixPath(pf.path))
    return ChangedFile(
        path=path,
        status=status,
        added=pf.added,
        removed=pf.removed,
        is_python=PurePosixPath(path).suffix == ".py",
        old_path=_strip_prefix(pf.source_file) if status == "renamed" else None,
    )


@dataclass(frozen=True)
class ReviewContext:
    """What every tool works on: the checkout after the PR and the PR's unified diff."""

    workspace: Path  # absolute, resolved
    diff_text: str
    files: tuple[ChangedFile, ...]

    @classmethod
    def from_text(cls, workspace: Path, diff_text: str) -> ReviewContext:
        workspace = Path(workspace).resolve()
        if not workspace.is_dir():
            raise ToolError(f"workspace {workspace} is not a directory")
        files = tuple(_changed_file(pf) for pf in parse_patch(diff_text))
        return cls(workspace=workspace, diff_text=diff_text, files=files)

    @classmethod
    def from_paths(cls, workspace: Path, diff_path: Path) -> ReviewContext:
        """Read ``diff_path`` as bytes and decode it as UTF-8 (no newline translation)."""
        return cls.from_text(workspace, Path(diff_path).read_bytes().decode("utf-8"))


def changed_files(ctx: ReviewContext) -> list[ChangedFile]:
    return list(ctx.files)


def _find_patch(patch: PatchSet, path: str) -> PatchedFile:
    want = path[2:] if path.startswith("./") else path
    for pf in patch:
        if want in (str(PurePosixPath(pf.path)), _strip_prefix(pf.source_file)):
            return pf
    names = ", ".join(str(PurePosixPath(pf.path)) for pf in patch) or "none"
    raise ToolError(f"{path!r} is not in the diff; changed files: {names}")


def _numbered_file(pf: PatchedFile) -> list[str]:
    cf = _changed_file(pf)
    what = f"renamed from {cf.old_path}" if cf.status == "renamed" else cf.status
    if pf.is_binary_file:
        return [f"=== {cf.path} ({what}, binary: no text diff) ==="]
    out = [f"=== {cf.path} ({what}, +{cf.added} -{cf.removed}) ==="]
    numbers = [ln.target_line_no for hunk in pf for ln in hunk if ln.target_line_no is not None]
    width = max(4, len(str(max(numbers, default=0))))
    blank = " " * width
    for hunk in pf:
        header = (
            f"@@ -{hunk.source_start},{hunk.source_length} "
            f"+{hunk.target_start},{hunk.target_length} @@ {hunk.section_header}"
        )
        out.append(header.rstrip())
        for ln in hunk:
            text = ln.value.rstrip("\n")
            if ln.is_added or ln.is_context:
                out.append(f"{ln.target_line_no:>{width}} {ln.line_type}{text}")
            elif ln.is_removed:
                out.append(f"{blank} -{text}")
            else:  # "\ No newline at end of file"
                out.append(f"{blank} {ln.line_type}{text}".rstrip())
    return out


DIFF_LEGEND = (
    "Numbers are NEW-file line numbers (cite these). '+' added, ' ' unchanged, "
    "'-' removed (no number: not in the new file)."
)


def numbered_diff(ctx: ReviewContext, path: str | None = None) -> str:
    """The unified diff, complete or for one file, with every added and context line prefixed by
    its new-file line number and removed lines by a blank prefix. Raises ``ToolError`` for a path
    that is not in the diff."""
    patch = parse_patch(ctx.diff_text)
    selected = [_find_patch(patch, path)] if path is not None else list(patch)
    if not selected:
        return "The diff is empty."
    lines = [DIFF_LEGEND]
    for pf in selected:
        lines.append("")
        lines.extend(_numbered_file(pf))
    return "\n".join(lines)
