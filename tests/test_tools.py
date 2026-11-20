"""src/reviewer/tools: the review context, the numbered diff and the six tools, on a synthetic
git repository in tmp_path. No network and no model calls; ruff, bandit and ripgrep run on the
host because they only read (D2a). ripgrep must be on PATH: the search tests do not skip.

Every expected value is derived from the fixture texts below (the named edits), never from a
tool's own output.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from eval.score import changed_lines
from reviewer.tools import (
    MAX_OUTPUT_CHARS,
    ReviewContext,
    changed_files,
    limit_output,
    list_changed_files,
    make_tools,
    numbered_diff,
    read_diff,
    read_file,
    run_bandit,
    run_ruff,
    search_code,
)
from reviewer.tools import search as search_module

# --- The fixture: base files and the PR's named edits ----------------------------------------

CORE_BASE = "".join(f"def f{i}():\n    return {i}\n\n" for i in range(1, 11))
# pkg/core.py, modified: three named edits.
F2_OLD = "def f2():\n    return 2\n"
F2_NEW = "def f2(x=None):\n    # changed\n    return -1 if x else 2\n"  # replaces F2_OLD
F9_OLD = "def f9():\n    return 9\n\n"  # deleted
RISKY = "def risky(cmd):\n    return eval(cmd)\n"  # appended; bandit B307
assert CORE_BASE.count(F2_OLD) == 1 and CORE_BASE.count(F9_OLD) == 1
CORE_AFTER = CORE_BASE.replace(F2_OLD, F2_NEW).replace(F9_OLD, "") + RISKY
CORE_ADDED = len(F2_NEW.splitlines()) + len(RISKY.splitlines())
CORE_REMOVED = len(F2_OLD.splitlines()) + len(F9_OLD.splitlines())
CORE_TOTAL = len(CORE_AFTER.splitlines())

# pkg/old_name.py -> pkg/new_name.py, renamed with one changed line.
RENAMED_BASE = "".join(f"VALUE_{i} = {i}\n" for i in range(1, 21))
RENAMED_AFTER = RENAMED_BASE.replace("VALUE_7 = 7\n", "VALUE_7 = 70\n")

GONE_BASE = "x = 1\n"  # pkg/gone.py, deleted
ADDED_PY = "import os\n\nTOKEN = 'abc'\n"  # pkg/added.py, added; ruff F401, bandit B105
NOTES_BASE = "# Notes\n\nfirst\n"
NOTES_AFTER = NOTES_BASE + "second\n"  # docs/notes.md, modified

# Changed .py files that exist after the PR: the linters' default file set.
LINTABLE = ["pkg/added.py", "pkg/core.py", "pkg/new_name.py"]


def line_of(text: str, line: str) -> int:
    """1-based number of the only line of ``text`` equal to ``line``."""
    lines = text.splitlines()
    assert lines.count(line) == 1, line
    return lines.index(line) + 1


# Lint hits planted on purpose: (path, line, code). Nothing else in the fixture is flagged by
# ruff E,F,B or by bandit (checked by hand: CORE_BASE is plain defs and returns, RENAMED_* are
# int assignments, the eval() call is a bandit test, not a ruff E/F/B rule).
RUFF_PLANTED = [("pkg/added.py", line_of(ADDED_PY, "import os"), "F401")]
BANDIT_PLANTED = [
    ("pkg/added.py", line_of(ADDED_PY, "TOKEN = 'abc'"), "B105"),
    ("pkg/core.py", line_of(CORE_AFTER, "    return eval(cmd)"), "B307"),
]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    ).stdout


def write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


@pytest.fixture
def ws(tmp_path) -> Path:
    """A repository whose working tree is the state after a PR that modifies, adds, deletes and
    renames files. ``pr.diff`` sits next to it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    write(repo / "pkg" / "core.py", CORE_BASE)
    write(repo / "pkg" / "old_name.py", RENAMED_BASE)
    write(repo / "pkg" / "gone.py", GONE_BASE)
    write(repo / "docs" / "notes.md", NOTES_BASE)
    write(repo / "data.bin", b"\x00\x01\x02binary")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")

    write(repo / "pkg" / "core.py", CORE_AFTER)
    git(repo, "mv", "pkg/old_name.py", "pkg/new_name.py")
    write(repo / "pkg" / "new_name.py", RENAMED_AFTER)
    (repo / "pkg" / "gone.py").unlink()
    write(repo / "pkg" / "added.py", ADDED_PY)
    write(repo / "docs" / "notes.md", NOTES_AFTER)
    write(repo / "pkg" / "untouched.py", "")  # in the workspace but not in the diff
    git(repo, "add", "pkg", "docs")
    git(repo, "rm", "-q", "--cached", "pkg/untouched.py")
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--no-color", "--no-ext-diff", "-M"],
        capture_output=True,
        check=True,
    ).stdout
    write(tmp_path / "pr.diff", diff)
    return repo


@pytest.fixture
def ctx(ws) -> ReviewContext:
    return ReviewContext.from_paths(ws, ws.parent / "pr.diff")


# --- ReviewContext, changed files, numbered diff ---------------------------------------------


def test_changed_files(ctx):
    got = {f.path: (f.status, f.added, f.removed, f.is_python, f.old_path) for f in ctx.files}
    assert got == {
        "docs/notes.md": ("modified", 1, 0, False, None),
        "pkg/added.py": ("added", len(ADDED_PY.splitlines()), 0, True, None),
        "pkg/core.py": ("modified", CORE_ADDED, CORE_REMOVED, True, None),
        "pkg/gone.py": ("deleted", 0, len(GONE_BASE.splitlines()), True, None),
        "pkg/new_name.py": ("renamed", 1, 1, True, "pkg/old_name.py"),
    }
    assert changed_files(ctx) == list(ctx.files)
    assert ctx.workspace.is_absolute()


def test_list_changed_files(ctx):
    def row(status, added, removed, name):
        return f"{status:<8} +{added:<5} -{removed:<5} {name}"

    assert list_changed_files(ctx).splitlines() == [
        row("modified", 1, 0, "docs/notes.md"),
        row("added", len(ADDED_PY.splitlines()), 0, "pkg/added.py"),
        row("modified", CORE_ADDED, CORE_REMOVED, "pkg/core.py"),
        row("deleted", 0, len(GONE_BASE.splitlines()), "pkg/gone.py"),
        row("renamed", 1, 1, "pkg/old_name.py -> pkg/new_name.py"),
    ]


def test_list_changed_files_empty(tmp_path):
    assert list_changed_files(ReviewContext.from_text(tmp_path, "")) == "No files changed."


def numbered(n: int, marker: str, text: str) -> str:
    """A numbered-diff line as documented: number right-aligned in 4 columns (the minimum width;
    the fixture's files are under 100 lines), a space, then the diff marker and the text."""
    return f"{n:>4} {marker}{text}"


def removed(text: str) -> str:
    return f"{'':>4} -{text}"


def test_numbered_diff_one_file(ctx):
    out = numbered_diff(ctx, "pkg/core.py").splitlines()
    assert out[0].startswith("Numbers are NEW-file line numbers")
    assert out[2] == f"=== pkg/core.py (modified, +{CORE_ADDED} -{CORE_REMOVED}) ==="
    for line in F2_NEW.splitlines() + RISKY.splitlines():
        assert numbered(line_of(CORE_AFTER, line), "+", line) in out
    for line in F2_OLD.splitlines():
        assert removed(line) in out
    for line in ("def f9():", "    return 9"):
        assert removed(line) in out
    # Context lines carry their new-file number: def f1() is within 3 lines of the f2 edit.
    assert numbered(line_of(CORE_AFTER, "def f1():"), " ", "def f1():") in out


def test_numbered_diff_renamed_and_deleted(ctx):
    full = read_diff(ctx)
    assert "=== pkg/new_name.py (renamed from pkg/old_name.py, +1 -1) ===" in full
    n = line_of(RENAMED_AFTER, "VALUE_7 = 70")
    # The old name finds it too.
    assert numbered(n, "+", "VALUE_7 = 70") in read_diff(ctx, "pkg/old_name.py").splitlines()
    gone = read_diff(ctx, "pkg/gone.py").splitlines()
    assert f"=== pkg/gone.py (deleted, +0 -{len(GONE_BASE.splitlines())}) ===" in gone
    assert removed("x = 1") in gone


def test_read_diff_unknown_path(ctx):
    out = read_diff(ctx, "pkg/nope.py")
    assert out.startswith("ERROR: 'pkg/nope.py' is not in the diff; changed files: docs/notes.md")


NUMBERED = re.compile(r"^ *(\d+) ([+ ])")


def numbered_lines(text: str) -> dict[str, dict[str, set[int]]]:
    """Per file: new-file numbers shown on added ('+') and context (' ') lines, and the number
    of the first numbered line after each run of removed lines."""
    out: dict[str, dict[str, set[int]]] = {}
    cur = None
    pending_removed = False
    for line in text.splitlines():
        if line.startswith("=== "):
            cur = out.setdefault(line[4:].split(" (")[0], {"+": set(), " ": set(), "after-": set()})
            continue
        if cur is None:
            continue
        if line.startswith("@@"):
            pending_removed = False
            continue
        m = NUMBERED.match(line)
        if m:
            n = int(m.group(1))
            cur[m.group(2)].add(n)
            if pending_removed:
                cur["after-"].add(n)
                pending_removed = False
        elif line.strip().startswith("-"):
            pending_removed = True
    return out


def dataset_diffs() -> list[Path]:
    root = Path(__file__).resolve().parent.parent / "eval" / "dataset"
    return sorted(root.glob("item_*/pr.diff"))


@pytest.mark.parametrize("source", ["fixture", *[p.parent.name for p in dataset_diffs()]])
def test_numbered_diff_matches_score_changed_lines(source, ctx, tmp_path):
    if source == "fixture":
        c = ctx
    else:
        diff = Path(__file__).resolve().parent.parent / "eval/dataset" / source / "pr.diff"
        c = ReviewContext.from_text(tmp_path, diff.read_bytes().decode("utf-8"))
    shown = numbered_lines(numbered_diff(c))
    changed = changed_lines(c.diff_text)
    assert set(changed) == {f.path for f in c.files if f.status != "deleted"}
    for path, lines in changed.items():
        added = shown[path]["+"]
        # The numbered '+' lines are exactly the added-line subset of changed_lines ...
        assert added <= lines
        # ... and every other changed line is where a removal sits: the next numbered line.
        rest = lines - added
        assert rest <= shown[path]["after-"] | shown[path][" "] | added | _hunk_ends(c, path)


def _hunk_ends(c: ReviewContext, path: str) -> set[int]:
    from reviewer.tools.context import parse_patch

    ends = set()
    for pf in parse_patch(c.diff_text):
        if str(pf.path) == path:
            for h in pf:
                ends.add(max(h.target_start, h.target_start + h.target_length - 1))
    return ends


def test_numbered_added_lines_equal_added_lines_exactly(ctx):
    from eval.check_dataset import added_lines

    shown = numbered_lines(numbered_diff(ctx))
    for path, lines in added_lines(ctx.diff_text).items():
        assert shown[path]["+"] == lines
    # And for core.py, independently from the edit texts:
    want = {line_of(CORE_AFTER, ln) for ln in F2_NEW.splitlines() + RISKY.splitlines()}
    assert shown["pkg/core.py"]["+"] == want


def test_context_from_paths_reads_utf8_bytes(tmp_path):
    diff = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-a = 'x'\n+a = 'é→'\n"
    (tmp_path / "pr.diff").write_bytes(diff.encode("utf-8"))
    ctx = ReviewContext.from_paths(tmp_path, tmp_path / "pr.diff")
    assert numbered(1, "+", "a = 'é→'") in numbered_diff(ctx).splitlines()


# --- Truncation ------------------------------------------------------------------------------


def test_limit_output():
    assert limit_output("x" * MAX_OUTPUT_CHARS) == "x" * MAX_OUTPUT_CHARS
    cut = limit_output("x" * (MAX_OUTPUT_CHARS + 250))
    assert cut == "x" * MAX_OUTPUT_CHARS + "\n[truncated: 250 more characters; narrow the request]"


def test_tool_output_is_truncated(tmp_path):
    body = "".join(f"+line {i} {'y' * 60}\n" for i in range(1, 501))
    diff = (
        "diff --git a/big.py b/big.py\nnew file mode 100644\n--- /dev/null\n+++ b/big.py\n"
        f"@@ -0,0 +1,500 @@\n{body}"
    )
    ctx = ReviewContext.from_text(tmp_path, diff)
    out = read_diff(ctx)
    full = numbered_diff(ctx)
    assert len(full) > MAX_OUTPUT_CHARS
    assert out.endswith(
        f"[truncated: {len(full) - MAX_OUTPUT_CHARS} more characters; narrow the request]"
    )
    assert out.startswith(full[:MAX_OUTPUT_CHARS])


# --- read_file -------------------------------------------------------------------------------


def file_view(path: str, text: str, start: int, end: int) -> list[str]:
    """read_file's documented layout: a header, then ``n | line`` with n right-aligned to the
    width of ``end``."""
    lines = text.splitlines()
    width = len(str(end))
    rows = [f"{n:>{width}} | {lines[n - 1]}" for n in range(start, end + 1)]
    return [f"{path}: lines {start}-{end} of {len(lines)}", *rows]


def test_read_file(ctx):
    n = line_of(CORE_AFTER, "def f2(x=None):")
    out = read_file(ctx, "pkg/core.py", n - 1, n + 1)
    assert out.splitlines() == file_view("pkg/core.py", CORE_AFTER, n - 1, n + 1)
    total = len(ADDED_PY.splitlines())
    assert read_file(ctx, "./pkg/added.py").splitlines() == file_view(
        "pkg/added.py", ADDED_PY, 1, total
    )


def test_read_file_numbers_lines_like_git(tmp_path):
    # A form feed must not start a new line (str.splitlines would split there).
    write(tmp_path / "f.py", "a = 1\nb = '\f'\r\nc = 3\n")
    out = read_file(ReviewContext.from_text(tmp_path, ""), "f.py")
    assert out.splitlines()[-1] == "3 | c = 3"


def test_read_file_clips_at_400_lines(tmp_path):
    write(tmp_path / "long.py", "".join(f"x{i} = {i}\n" for i in range(1, 1001)))
    ctx = ReviewContext.from_text(tmp_path, "")
    out = read_file(ctx, "long.py").splitlines()
    assert out[0] == "long.py: lines 1-400 of 1000"
    assert out[-2] == "400 | x400 = 400"
    assert out[-1] == (
        "[clipped: read_file returns at most 400 lines per call; call again with start_line=401]"
    )
    out = read_file(ctx, "long.py", 401, 2000).splitlines()
    assert out[0] == "long.py: lines 401-800 of 1000" and "start_line=801" in out[-1]
    out = read_file(ctx, "long.py", 801).splitlines()
    assert out[0] == "long.py: lines 801-1000 of 1000" and out[-1] == "1000 | x1000 = 1000"
    assert "clipped" not in read_file(ctx, "long.py", 10, 409)


@pytest.mark.parametrize(
    ("start", "end", "msg"),
    [
        (0, None, "ERROR: start_line must be 1 or more, got 0"),
        (5, 4, "ERROR: end_line (4) is before start_line (5)"),
        (
            CORE_TOTAL + 1,
            None,
            f"ERROR: start_line {CORE_TOTAL + 1} is past the end of pkg/core.py "
            f"({CORE_TOTAL} lines)",
        ),
    ],
)
def test_read_file_bad_lines(ctx, start, end, msg):
    assert read_file(ctx, "pkg/core.py", start, end) == msg


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("", "must not be empty"),
        ("/etc/passwd", "not absolute"),
        ("C:/Windows/win.ini", "not absolute"),
        ("C:Windows/win.ini", "not absolute"),
        ("//server/share/x", "not absolute"),
        ("pkg\\core.py", "forward slashes"),
        ("\\\\server\\share", "forward slashes"),
        ("../outside.py", "'..' segments"),
        ("pkg/../../outside.py", "'..' segments"),
        (".git/config", "under .git"),
        (".GIT/config", "under .git"),
        (".git./config", "under .git"),
        ("pkg/.git/HEAD", "under .git"),
        ("pkg/core.py:stream", "':'"),
        ("pkg/missing.py", "does not exist"),
        ("pkg", "is not a file"),
    ],
)
def test_read_file_path_safety(ctx, path, reason):
    out = read_file(ctx, path)
    assert out.startswith("ERROR: ") and reason in out, out


def symlink_or_skip(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"cannot create symlinks here: {e}")


def test_read_file_rejects_symlink_outside(ctx, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    symlink_or_skip(ctx.workspace / "link.txt", secret)
    out = read_file(ctx, "link.txt")
    assert out == "ERROR: 'link.txt' resolves outside the repository (symlink?)"


def test_read_file_rejects_symlinked_dir_outside(ctx, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("x = 1\n")
    symlink_or_skip(ctx.workspace / "ext", outside, is_dir=True)
    assert "resolves outside the repository" in read_file(ctx, "ext/a.py")


def test_read_file_rejects_symlink_into_git(ctx):
    symlink_or_skip(ctx.workspace / "head", ctx.workspace / ".git" / "HEAD")
    assert read_file(ctx, "head") == "ERROR: 'head' resolves into .git"


def test_read_file_binary(ctx):
    assert read_file(ctx, "data.bin") == "ERROR: 'data.bin' is a binary file"


def test_read_file_replaces_invalid_utf8(tmp_path):
    write(tmp_path / "latin.py", b"name = 'caf\xe9'\n")
    out = read_file(ReviewContext.from_text(tmp_path, ""), "latin.py")
    assert out.splitlines()[1] == "1 | name = 'caf\ufffd'"


# --- search_code -----------------------------------------------------------------------------


def hits(path: str, text: str, pattern: str) -> list[str]:
    """``path:line: text`` for every line of ``text`` that ``pattern`` matches."""
    return [
        f"{path}:{i}: {ln}" for i, ln in enumerate(text.splitlines(), 1) if re.search(pattern, ln)
    ]


def test_search_without_ripgrep(ctx, monkeypatch):
    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    assert search_code(ctx, "def") == "ERROR: ripgrep not installed"


def test_search_empty_pattern(ctx):
    assert search_code(ctx, "") == "ERROR: pattern must not be empty"


def test_search_code(ctx):
    pattern = r"def f[23]\("
    want = hits("pkg/core.py", CORE_AFTER, pattern)
    assert len(want) == 2  # def f2(x=None): and def f3():
    assert search_code(ctx, pattern).splitlines() == want
    assert search_code(ctx, "VALUE_7", glob="*.py").splitlines() == hits(
        "pkg/new_name.py", RENAMED_AFTER, "VALUE_7"
    )
    assert search_code(ctx, "first", glob="!docs/**") == "No matches."
    assert search_code(ctx, "nothing-like-this") == "No matches."


def test_search_pattern_starting_with_dash_is_a_pattern(ctx):
    line = "    return -1 if x else 2"
    assert search_code(ctx, "-1 if x") == f"pkg/core.py:{line_of(CORE_AFTER, line)}: {line}"
    # Even a pattern that looks exactly like a flag is searched for, not obeyed.
    write(ctx.workspace / "flags.txt", "--files\n")
    assert search_code(ctx, "--files", fixed_string=True) == "flags.txt:1: --files"


def test_search_fixed_string_and_bad_regex(ctx):
    line = "    return eval(cmd)"
    want = f"pkg/core.py:{line_of(CORE_AFTER, line)}: {line}"
    assert search_code(ctx, "eval(cmd)", fixed_string=True) == want
    assert search_code(ctx, "eval(").startswith("ERROR: ripgrep: ")


def test_search_skips_git_dir_and_caps_matches(ctx):
    assert search_code(ctx, "repositoryformatversion") == "No matches."  # only in .git/config
    for i in range(4):
        write(ctx.workspace / f"many{i}.txt", "needle\n" * 30)
    out = search_code(ctx, "needle").splitlines()
    assert len(out) == 51
    assert out[-1] == "[stopped at 50 matches; narrow the pattern or the glob]"
    per_file = {line.split(":")[0] for line in out[:-1]}
    assert len(per_file) == 3  # 20 per file


def test_search_cuts_long_lines(ctx):
    write(ctx.workspace / "wide.txt", "needle " + "z" * 1000 + "\n")
    out = search_code(ctx, "needle", glob="wide.txt")
    assert out.startswith("wide.txt:1: ") and len(out) < 400


# --- ruff and bandit -------------------------------------------------------------------------


def test_run_ruff(ctx):
    out = run_ruff(ctx).splitlines()
    assert out[0] == f"ruff (E,F,B) on {len(LINTABLE)} file(s): {len(RUFF_PLANTED)} issue(s)"
    assert [tuple(re.match(r"^(\S+):(\d+): (\S+) ", ln).groups()) for ln in out[1:]] == [
        (p, str(n), code) for p, n, code in RUFF_PLANTED
    ]
    assert out[1].endswith("`os` imported but unused")


def test_run_ruff_ignores_repository_config(ctx):
    write(ctx.workspace / "ruff.toml", '[lint]\nignore = ["F401"]\n')
    write(ctx.workspace / "pyproject.toml", '[tool.ruff.lint]\nignore = ["F401"]\n')
    assert "F401" in run_ruff(ctx, ["pkg/added.py"])


def test_run_bandit(ctx):
    out = run_bandit(ctx).splitlines()
    assert (
        out[0] == f"bandit (skip B101) on {len(LINTABLE)} file(s): {len(BANDIT_PLANTED)} issue(s)"
    )
    got = [re.match(r"^(\S+):(\d+): (B\d+) [A-Z]+/[A-Z]+: ", ln).groups() for ln in out[1:]]
    assert got == [(p, str(n), tid) for p, n, tid in BANDIT_PLANTED]


def test_run_bandit_ignores_repository_config(ctx):
    write(ctx.workspace / ".bandit", "[bandit]\nskips: B105,B307\n")
    write(ctx.workspace / "pkg" / ".bandit", "[bandit]\nskips: B105,B307\n")
    assert "B105" in run_bandit(ctx, ["pkg/added.py"])


@pytest.mark.parametrize("tool", [run_ruff, run_bandit])
@pytest.mark.parametrize(
    "path",
    ["pkg/untouched.py", "pkg/gone.py", "docs/notes.md", "pkg/old_name.py", "../x.py", "/tmp/a.py"],
)
def test_linters_only_accept_changed_python_files(ctx, tool, path):
    out = tool(ctx, [path])
    allowed = ", ".join(LINTABLE)
    assert out == f"ERROR: {path!r} is not a changed Python file; allowed: {allowed}"


@pytest.mark.parametrize("tool", [run_ruff, run_bandit])
def test_linters_check_only_the_given_file(ctx, tool):
    out = tool(ctx, ["pkg/new_name.py"])  # no planted hits in the renamed file
    assert out.endswith("on 1 file(s): no issues.")


@pytest.mark.parametrize("tool", [run_ruff, run_bandit])
def test_linters_with_no_python_changes(tmp_path, tool):
    diff = "diff --git a/r.md b/r.md\n--- a/r.md\n+++ b/r.md\n@@ -1 +1 @@\n-a\n+b\n"
    write(tmp_path / "r.md", "b\n")
    assert tool(ReviewContext.from_text(tmp_path, diff)) == "No changed Python files to check."


@pytest.mark.parametrize("tool", [run_ruff, run_bandit])
def test_linters_refuse_a_changed_file_symlinked_outside(ctx, tmp_path, tool):
    outside = tmp_path / "outside.py"
    outside.write_text("import os\n")
    (ctx.workspace / "pkg" / "added.py").unlink()
    symlink_or_skip(ctx.workspace / "pkg" / "added.py", outside)
    assert "resolves outside the repository" in tool(ctx, ["pkg/added.py"])


# --- make_tools ------------------------------------------------------------------------------

EXPECTED_ARGS = {
    "list_changed_files": {},
    "read_diff": {"path": (False, None)},
    "read_file": {"path": (True, None), "start_line": (False, 1), "end_line": (False, None)},
    "search_code": {
        "pattern": (True, None),
        "glob": (False, None),
        "fixed_string": (False, False),
    },
    "run_ruff": {"paths": (False, None)},
    "run_bandit": {"paths": (False, None)},
}


def test_make_tools_names_and_schemas(ctx):
    tools = make_tools(ctx)
    assert [t.name for t in tools] == list(EXPECTED_ARGS)
    for t in tools:
        schema = t.args_schema.model_json_schema()
        required = set(schema.get("required", []))
        got = {
            name: (name in required, prop.get("default"))
            for name, prop in schema.get("properties", {}).items()
        }
        assert got == EXPECTED_ARGS[t.name], t.name
        assert "workspace" not in str(schema)
        assert t.description and not t.description.startswith(" ")
    types = {t.name: t.args for t in tools}
    assert types["run_ruff"]["paths"]["anyOf"][0] == {"items": {"type": "string"}, "type": "array"}
    assert types["read_file"]["start_line"]["type"] == "integer"
    assert types["search_code"]["fixed_string"]["type"] == "boolean"


def test_make_tools_invoke_and_never_raise(ctx):
    tools = {t.name: t for t in make_tools(ctx)}
    assert tools["list_changed_files"].invoke({}).startswith("modified +1")
    added = len(ADDED_PY.splitlines())
    assert f"=== pkg/added.py (added, +{added} -0) ===" in tools["read_diff"].invoke(
        {"path": "pkg/added.py"}
    )
    assert (
        tools["read_file"].invoke({"path": "pkg/added.py", "end_line": 1}).endswith("1 | import os")
    )
    assert tools["read_file"].invoke({"path": "../x"}).startswith("ERROR: ")
    assert tools["run_ruff"].invoke({"paths": ["pkg/added.py"]}).startswith("ruff (E,F,B)")
    assert tools["run_bandit"].invoke({}).startswith("bandit (skip B101)")
    assert tools["search_code"].invoke({"pattern": "TOKEN"}) == (
        f"pkg/added.py:{line_of(ADDED_PY, 'TOKEN = ' + repr('abc'))}: TOKEN = 'abc'"
    )
    assert tools["read_diff"].invoke({"path": "nope"}).startswith("ERROR: ")


# --- A real dataset item ---------------------------------------------------------------------


@pytest.mark.integration
def test_tools_on_materialized_item_001():
    from eval.build_dataset import load_recipe
    from eval.ground_truth import load_item
    from eval.repos import clone_dir, git_bytes, materialize

    repo = clone_dir("marshmallow")
    if not (repo / ".git").exists():
        pytest.skip("marshmallow clone not present (eval/repos/marshmallow)")
    item = Path(__file__).resolve().parent.parent / "eval" / "dataset" / "item_001"
    gt, _ = load_item(item)
    bug = gt.bugs[0]
    # Expected line: the upstream commit's file from git objects, with the recipe's plant applied.
    recipe = load_recipe("item_001")
    plant = recipe.bugs[0]
    upstream = git_bytes(repo, "show", f"{recipe.commit}:{plant.file}").decode("utf-8")
    assert upstream.count(plant.find) == 1
    planted = upstream.replace(plant.find, plant.replace, 1)
    expected = planted.split("\n")[bug.line_start - 1]
    assert expected.strip() == plant.replace.strip()

    ctx = ReviewContext.from_paths(materialize("item_001"), item / "pr.diff")
    tools = {t.name: t for t in make_tools(ctx)}
    assert bug.file in tools["list_changed_files"].invoke({})
    diff = tools["read_diff"].invoke({"path": bug.file})
    pattern = rf"^ *{bug.line_start} \+{re.escape(expected)}$"
    assert re.search(pattern, diff, re.MULTILINE)
    view = tools["read_file"].invoke(
        {"path": bug.file, "start_line": bug.line_start, "end_line": bug.line_start}
    )
    assert view.endswith(f"{bug.line_start} | {expected}")
    assert not tools["run_ruff"].invoke({}).startswith("ERROR")
    assert not tools["run_bandit"].invoke({}).startswith("ERROR")
    found = tools["search_code"].invoke({"pattern": plant.replace.strip(), "fixed_string": True})
    assert f"{bug.file}:{bug.line_start}: {expected}" in found.splitlines()
