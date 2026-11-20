"""The review tools: plain functions that take the ``ReviewContext`` first, and ``make_tools``,
which binds them to one context as LangChain tools (the model never passes a workspace path).

Every tool returns a string, cut to 12,000 characters, and reports problems as ``ERROR: ...``
strings instead of raising.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool

from reviewer.tools import files, linters, search
from reviewer.tools.context import ReviewContext, ToolError, limit_output, numbered_diff


def _guard(fn: Callable[[], str]) -> str:
    """Run a tool body: cut long output, turn any exception into an ``ERROR:`` string."""
    try:
        return limit_output(fn())
    except ToolError as e:
        return f"ERROR: {e}"
    except Exception as e:  # never raise into the agent loop
        return f"ERROR: {type(e).__name__}: {e}"


def list_changed_files(ctx: ReviewContext) -> str:
    def body() -> str:
        if not ctx.files:
            return "No files changed."
        rows = []
        for f in ctx.files:
            name = f"{f.old_path} -> {f.path}" if f.old_path else f.path
            rows.append(f"{f.status:<8} +{f.added:<5} -{f.removed:<5} {name}")
        return "\n".join(rows)

    return _guard(body)


def read_diff(ctx: ReviewContext, path: str | None = None) -> str:
    return _guard(lambda: numbered_diff(ctx, path))


def read_file(
    ctx: ReviewContext, path: str, start_line: int = 1, end_line: int | None = None
) -> str:
    return _guard(lambda: files.read_file(ctx, path, start_line, end_line))


def search_code(
    ctx: ReviewContext, pattern: str, glob: str | None = None, fixed_string: bool = False
) -> str:
    return _guard(lambda: search.search_code(ctx, pattern, glob, fixed_string))


def run_ruff(ctx: ReviewContext, paths: list[str] | None = None) -> str:
    return _guard(lambda: linters.run_ruff(ctx, paths))


def run_bandit(ctx: ReviewContext, paths: list[str] | None = None) -> str:
    return _guard(lambda: linters.run_bandit(ctx, paths))


def make_tools(ctx: ReviewContext) -> list[BaseTool]:
    """The six review tools bound to ``ctx``. Their names and docstrings are what the model
    reads."""

    def list_changed_files_tool() -> str:
        """List the files this pull request changes, one per line: status (added, modified,
        deleted or renamed), lines added, lines removed, and the path. Takes no arguments.
        Start here, then read the diff."""
        return list_changed_files(ctx)

    def read_diff_tool(path: str | None = None) -> str:
        """Show the pull request's unified diff with line numbers.

        Every added ('+') and unchanged (' ') line starts with its line number in the NEW file
        (after the pull request); removed ('-') lines have no number because they are not in the
        new file. Cite these new-file numbers in findings.

        Args:
            path: Repository-relative file path, as listed by list_changed_files
                (for example "src/pkg/module.py"). Leave empty for the whole diff.

        Output over 12,000 characters is cut; then ask for one file at a time.
        """
        return read_diff(ctx, path)

    def read_file_tool(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file from the repository as it is after the pull request, with line numbers.

        Args:
            path: Repository-relative path with forward slashes, for example "src/pkg/module.py".
                Absolute paths, "..", backslashes and anything under .git are refused.
            start_line: First line to show (1-based). Default 1.
            end_line: Last line to show (inclusive). Default: the end of the file.

        Returns at most 400 lines per call; when more were asked for, the output says where to
        continue. Binary files are refused. Any file in the repository can be read, not only
        the changed ones.
        """
        return read_file(ctx, path, start_line, end_line)

    def search_code_tool(pattern: str, glob: str | None = None, fixed_string: bool = False) -> str:
        """Search the repository (after the pull request) with ripgrep, for example to find the
        callers or the definition of a function.

        Args:
            pattern: A regular expression (Rust regex syntax), or plain text when fixed_string is
                true. Patterns starting with "-" are searched for, not treated as options.
            glob: Optional file filter, for example "*.py" or "src/**/*.py".
                Prefix with "!" to exclude, for example "!tests/**".
            fixed_string: True to match the pattern literally instead of as a regex.

        Returns "path:line: text" lines: at most 20 matches per file and 50 in total; long lines
        are cut at 300 characters. Hidden and git-ignored files are not searched.
        """
        return search_code(ctx, pattern, glob, fixed_string)

    def run_ruff_tool(paths: list[str] | None = None) -> str:
        """Run the ruff linter (rules E: pycodestyle errors, F: pyflakes, B: bugbear) on Python
        files changed by the pull request. It only reads the code, never runs it.

        Args:
            paths: Changed .py files to check, as listed by list_changed_files. Leave empty to
                check every changed .py file. Other files are refused.

        Returns "path:line: CODE message" lines with new-file line numbers. The repository's own
        ruff settings are ignored. A finding here is a hint, not proof of a bug.
        """
        return run_ruff(ctx, paths)

    def run_bandit_tool(paths: list[str] | None = None) -> str:
        """Run the bandit security scanner on Python files changed by the pull request (all
        bandit tests except B101, "assert used"). It only reads the code, never runs it.

        Args:
            paths: Changed .py files to check, as listed by list_changed_files. Leave empty to
                check every changed .py file. Other files are refused.

        Returns "path:line: TEST_ID SEVERITY/CONFIDENCE: message" lines with new-file line
        numbers. A finding here is a hint, not proof of a vulnerability.
        """
        return run_bandit(ctx, paths)

    specs = [
        ("list_changed_files", list_changed_files_tool),
        ("read_diff", read_diff_tool),
        ("read_file", read_file_tool),
        ("search_code", search_code_tool),
        ("run_ruff", run_ruff_tool),
        ("run_bandit", run_bandit_tool),
    ]
    return [
        StructuredTool.from_function(func=fn, name=name, description=inspect.getdoc(fn))
        for name, fn in specs
    ]
