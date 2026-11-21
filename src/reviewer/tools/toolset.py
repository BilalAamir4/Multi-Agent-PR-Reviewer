"""The review tools: plain functions that take the ``ReviewContext`` first, and ``make_tools``,
which binds them to one context as LangChain tools (the model never passes a workspace path).

Every tool returns a string, cut to 12,000 characters, and reports problems as ``ERROR: ...``
strings instead of raising.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool

from reviewer.tools import files, linters, search, testing
from reviewer.tools.context import ReviewContext, ToolError, limit_output, numbered_diff
from reviewer.tools.testing import ToolNotes

# The six stage 1A tools, in the order make_tools() returns them, then the stage 2A extra.
REVIEW_TOOLS = (
    "list_changed_files",
    "read_diff",
    "read_file",
    "search_code",
    "run_ruff",
    "run_bandit",
)
EXTRA_TOOLS = ("run_tests",)
ALL_TOOLS = REVIEW_TOOLS + EXTRA_TOOLS


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


def run_tests(
    ctx: ReviewContext, test_ids: list[str] | None = None, notes: ToolNotes | None = None
) -> str:
    return _guard(lambda: testing.run_tests(ctx, test_ids, notes))


def make_tools(
    ctx: ReviewContext,
    names: Sequence[str] | None = None,
    notes: ToolNotes | None = None,
) -> list[BaseTool]:
    """Review tools bound to ``ctx``: the six stage 1A tools when ``names`` is None, otherwise
    the named ones (from ``ALL_TOOLS``) in ``ALL_TOOLS`` order. ``notes`` collects what
    ``run_tests`` reports back to the agent's result (D2b). Their names and docstrings are what
    the model reads."""
    wanted = list(REVIEW_TOOLS) if names is None else list(names)
    unknown = sorted(set(wanted) - set(ALL_TOOLS))
    if unknown:
        raise ValueError(f"unknown tools {unknown}; known: {list(ALL_TOOLS)}")
    notes = notes if notes is not None else ToolNotes()

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

    def run_tests_tool(test_ids: list[str] | None = None) -> str:
        """Run the repository's pytest tests on the code as it is after the pull request.

        The tests run isolated, in a throwaway container with no network, on a copy of the
        repository; only pytest runs there. Failing tests are not stopped early.

        Args:
            test_ids: Tests to run, as pytest node IDs or paths from the repository root, for
                example ["tests/test_cart.py", "tests/test_cart.py::test_total"]. Find them in
                the diff or with search_code (for example the pattern "def test_" with the glob
                "tests/**"). Choose the tests of the changed code and the changed tests. Leave
                empty to run the whole suite, which can be slow. IDs that start with "-" or
                contain spaces, "..", or backslashes are refused, and each path must exist.

        Returns the pytest summary line (for example "1 failed, 3 passed in 0.20s"), the IDs of
        the failed and errored tests, and their short tracebacks; output over 12,000 characters
        is cut. A test that fails on the pull request is evidence of a defect.

        You can call this tool at most 3 times in a review, so put every test you need in as
        few calls as you can.
        """
        return run_tests(ctx, test_ids, notes)

    specs = [
        ("list_changed_files", list_changed_files_tool),
        ("read_diff", read_diff_tool),
        ("read_file", read_file_tool),
        ("search_code", search_code_tool),
        ("run_ruff", run_ruff_tool),
        ("run_bandit", run_bandit_tool),
        ("run_tests", run_tests_tool),
    ]
    return [
        StructuredTool.from_function(func=fn, name=name, description=inspect.getdoc(fn))
        for name, fn in specs
        if name in wanted
    ]
