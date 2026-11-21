"""``run_tests`` (stage 2A, decision 0008) with ``run_in_sandbox`` and the image check faked: no
Docker call is made. Command construction, test ID validation, file collection, output
formatting and truncation, the D2b path and the missing-image path."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from docker.errors import ImageNotFound

from reviewer.sandbox import SandboxResult, SandboxUnavailable
from reviewer.tools import MAX_OUTPUT_CHARS, ReviewContext, SandboxSpec, ToolNotes, make_tools
from reviewer.tools import testing as rt
from reviewer.tools.toolset import run_tests
from tests.dev_pr import make_fixture_workspace

SPEC = SandboxSpec(
    image="mapr-sandbox-dev_fixture:local",
    test_command=("python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"),
    pythonpath=(".",),
    timeout_s=60,
)

FAIL_OUT = """\
.F...                                                                    [100%]
=================================== FAILURES ===================================
__________________________________ test_clamp __________________________________
tests/test_summary.py:23: in test_clamp
    assert clamp(5, 0, 10) == 5
E   assert 10 == 5
=========================== short test summary info ============================
FAILED tests/test_summary.py::test_clamp - assert 10 == 5
1 failed, 4 passed in 0.01s
"""

ERROR_OUT = """\
E                                                                        [100%]
==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_broken.py _________________
tests/test_broken.py:1: in <module>
    import missing_module
E   ModuleNotFoundError: No module named 'missing_module'
=========================== short test summary info ============================
ERROR tests/test_broken.py - ModuleNotFoundError: No module named 'missing_module'
1 error in 0.05s
"""


@pytest.fixture
def ctx(tmp_path) -> ReviewContext:
    return make_fixture_workspace("dev_tests_pr", tmp_path, SPEC)


class FakeSandbox:
    """Records ``run_in_sandbox`` calls and returns a canned result (or raises)."""

    def __init__(self, result: SandboxResult | None = None, exc: Exception | None = None):
        self.result = result or SandboxResult(0, "5 passed in 0.01s\n", "", False, 0.1)
        self.exc = exc
        self.calls: list[tuple[str, dict[str, bytes], list[str], int]] = []

    def __call__(self, image, files, command, timeout_s=120):
        self.calls.append((image, files, command, timeout_s))
        if self.exc is not None:
            raise self.exc
        return self.result


@pytest.fixture
def sandbox(monkeypatch) -> FakeSandbox:
    fake = FakeSandbox()
    monkeypatch.setattr(rt, "run_in_sandbox", fake)
    monkeypatch.setattr(rt, "check_image", lambda image: None)
    return fake


# --- Command ---------------------------------------------------------------------------------


def test_command_removes_exitfirst_and_appends_ids():
    assert rt.build_command(SPEC, ["tests/test_summary.py::test_clamp"]) == [
        "python",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "-rfE",
        "tests/test_summary.py::test_clamp",
    ]


def test_command_removes_long_exitfirst_and_adds_q_once():
    spec = SandboxSpec("img", ("python", "-m", "pytest", "--exitfirst", "-x"), (".",), 5)
    assert rt.build_command(spec) == ["python", "-m", "pytest", "--tb=short", "-rfE", "-q"]


def test_tool_runs_spec_command_with_timeout_and_image(ctx, sandbox):
    out = run_tests(ctx, ["tests/test_summary.py", "./tests/test_summary.py::test_clamp"])
    assert out.startswith("pytest exit code 0: 5 passed in 0.01s")
    (image, files, command, timeout) = sandbox.calls[0]
    assert image == SPEC.image and timeout == SPEC.timeout_s
    assert "-x" not in command
    assert command[-2:] == ["tests/test_summary.py", "tests/test_summary.py::test_clamp"]


def test_no_ids_and_empty_list_run_the_whole_suite(ctx, sandbox):
    run_tests(ctx, None)
    run_tests(ctx, [])
    assert [c[2][-1] for c in sandbox.calls] == ["-rfE", "-rfE"]


# --- Test IDs --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("test_id", "reason"),
    [
        ("", "non-empty"),
        ("-k", "must not start with '-'"),
        ("--collect-only", "must not start with '-'"),
        ("tests/test_summary.py::test_x[a b]", "whitespace"),
        ("tests/test_summary.py\t", "whitespace"),
        ("tests\\test_summary.py", "backslashes"),
        ("tests/../stats/summary.py", "'..'"),
        ("tests/test_summary.py::..", "'..'"),
        ("tests/missing.py", "does not exist"),
        ("missing_dir::test_x", "does not exist"),
        ("/etc/passwd", "not absolute"),
        ("C:/x.py", "not absolute"),
        (".git/config", ".git"),
        ("::test_clamp", "starts with a path"),
        ("tests/test_summary.py::", "nothing after"),
    ],
)
def test_bad_ids_are_refused(ctx, sandbox, test_id, reason):
    out = run_tests(ctx, [test_id])
    assert out.startswith("ERROR: ") and reason in out, out
    assert sandbox.calls == []


def test_ids_must_be_a_list_and_not_too_many(ctx, sandbox):
    assert "must be a list" in run_tests(ctx, "tests/test_summary.py")  # type: ignore[arg-type]
    many = ["tests/test_summary.py"] * (rt.MAX_TEST_IDS + 1)
    assert "at most 50 test IDs" in run_tests(ctx, many)
    assert sandbox.calls == []


def test_directory_and_node_ids_are_accepted(ctx, sandbox):
    run_tests(ctx, ["tests", "tests/test_summary.py::test_clamp", "tests/test_summary.py"])
    assert sandbox.calls[0][2][-3:] == [
        "tests",
        "tests/test_summary.py::test_clamp",
        "tests/test_summary.py",
    ]


# --- Files -----------------------------------------------------------------------------------


def test_non_git_workspace_sends_every_file_but_caches(ctx, sandbox):
    (ctx.workspace / "__pycache__").mkdir()
    (ctx.workspace / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (ctx.workspace / "big.bin").write_bytes(b"x" * (rt.MAX_FILE_BYTES + 1))
    run_tests(ctx)
    files = sandbox.calls[0][1]
    assert set(files) == {
        "README.md",
        "stats/__init__.py",
        "stats/summary.py",
        "tests/test_summary.py",
    }
    assert files["stats/summary.py"] == (ctx.workspace / "stats/summary.py").read_bytes()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_workspace_sends_tracked_files_plus_diff_written_files(tmp_path, sandbox):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_bytes(b"A = 1\n")
    (repo / "pkg" / "gone.py").write_bytes(b"G = 1\n")
    (repo / "notes.txt").write_bytes(b"tracked\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "x")
    # The PR (applied, not committed): adds pkg/new.py and deletes pkg/gone.py.
    (repo / "pkg" / "new.py").write_bytes(b"N = 1\n")
    (repo / "pkg" / "gone.py").unlink()
    (repo / "scratch.txt").write_bytes(b"untracked and not in the diff\n")
    diff = (
        "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n--- /dev/null\n"
        "+++ b/pkg/new.py\n@@ -0,0 +1 @@\n+N = 1\n"
        "diff --git a/pkg/gone.py b/pkg/gone.py\ndeleted file mode 100644\n--- a/pkg/gone.py\n"
        "+++ /dev/null\n@@ -1 +0,0 @@\n-G = 1\n"
    )
    ctx = ReviewContext.from_text(repo, diff, SPEC)
    run_tests(ctx)
    assert set(sandbox.calls[0][1]) == {"pkg/a.py", "pkg/new.py", "notes.txt"}


# --- Output ----------------------------------------------------------------------------------


def test_failure_output_has_summary_ids_and_tracebacks(ctx, sandbox):
    sandbox.result = SandboxResult(1, FAIL_OUT, "", False, 0.2)
    out = run_tests(ctx, ["tests/test_summary.py"])
    lines = out.splitlines()
    assert lines[0] == "pytest exit code 1: 1 failed, 4 passed in 0.01s"
    assert lines[1] == "Failed: tests/test_summary.py::test_clamp"
    assert lines[2] == "Errors: none"
    assert "tests/test_summary.py:23: in test_clamp" in out
    assert "E   assert 10 == 5" in out
    assert "short test summary info" not in out
    assert "Output:" not in out


def test_collection_error_lists_errored_ids(ctx, sandbox):
    sandbox.result = SandboxResult(2, ERROR_OUT, "", False, 0.2)
    out = run_tests(ctx)
    assert out.splitlines()[0] == "pytest exit code 2: 1 error in 0.05s"
    assert "Errors: tests/test_broken.py" in out
    assert "ModuleNotFoundError" in out
    assert "Output:" in out  # exit code 2: the raw output tail is included too


def test_no_summary_line_includes_output_and_stderr(ctx, sandbox):
    sandbox.result = SandboxResult(4, "", "ERROR: usage: pytest [options]\n", False, 0.1)
    out = run_tests(ctx)
    assert out.startswith("pytest exit code 4: no summary line")
    assert "ERROR: usage: pytest [options]" in out


def test_long_output_is_truncated_with_the_usual_note(ctx, sandbox):
    body = "\n".join(f"E   line {i} " + "x" * 80 for i in range(400))
    long_out = FAIL_OUT.replace("E   assert 10 == 5\n", f"E   assert 10 == 5\n{body}\n")
    sandbox.result = SandboxResult(1, long_out, "", False, 0.2)
    out = run_tests(ctx)
    assert len(out) <= MAX_OUTPUT_CHARS + 100
    assert out.startswith("pytest exit code 1: 1 failed, 4 passed in 0.01s")
    assert out.rstrip().endswith("more characters; narrow the request]")
    assert "[truncated: " in out


def test_timeout_reports_error_and_partial_output(ctx, sandbox):
    sandbox.result = SandboxResult(None, "..F.\n", "", True, 60.0)
    out = run_tests(ctx)
    assert out.startswith("ERROR: tests timed out after 60 s")
    assert "Output:\n..F." in out


# --- No sandbox, D2b, missing image ----------------------------------------------------------


def test_no_sandbox_configured(tmp_path, sandbox):
    ctx = make_fixture_workspace("dev_tests_pr", tmp_path)
    notes = ToolNotes()
    assert run_tests(ctx, None, notes) == "ERROR: no test sandbox is configured for this review"
    assert notes.unavailable is False
    assert notes.notes == ["run_tests unavailable: no test sandbox is configured for this review"]
    assert sandbox.calls == []


def test_docker_not_running_is_d2b(ctx, monkeypatch):
    def down(image):
        raise SandboxUnavailable()

    monkeypatch.setattr(rt, "check_image", down)
    monkeypatch.setattr(rt, "run_in_sandbox", FakeSandbox())
    notes = ToolNotes()
    assert run_tests(ctx, None, notes) == "ERROR: Docker Desktop is not running"
    assert notes.unavailable is True
    assert notes.notes == ["sandbox unavailable: Docker Desktop is not running"]


def test_docker_lost_during_the_run_is_d2b(ctx, monkeypatch):
    monkeypatch.setattr(rt, "check_image", lambda image: None)
    monkeypatch.setattr(rt, "run_in_sandbox", FakeSandbox(exc=SandboxUnavailable()))
    notes = ToolNotes()
    assert run_tests(ctx, None, notes) == "ERROR: Docker Desktop is not running"
    assert notes.unavailable is True


def test_missing_image(ctx, monkeypatch):
    def missing(image):
        raise ImageNotFound("no such image")

    monkeypatch.setattr(rt, "check_image", missing)
    notes = ToolNotes()
    out = run_tests(ctx, None, notes)
    assert out == "ERROR: sandbox image mapr-sandbox-dev_fixture:local is not built"
    assert notes.unavailable is True


def test_unexpected_exception_becomes_error_string(ctx, monkeypatch):
    monkeypatch.setattr(rt, "check_image", lambda image: None)
    monkeypatch.setattr(rt, "run_in_sandbox", FakeSandbox(exc=RuntimeError("boom")))
    assert run_tests(ctx) == "ERROR: RuntimeError: boom"


# --- As a LangChain tool ---------------------------------------------------------------------


def test_make_tools_default_is_unchanged_and_run_tests_is_opt_in(ctx):
    assert [t.name for t in make_tools(ctx)] == [
        "list_changed_files",
        "read_diff",
        "read_file",
        "search_code",
        "run_ruff",
        "run_bandit",
    ]
    tools = make_tools(ctx, ["run_tests", "read_diff"])
    assert [t.name for t in tools] == ["read_diff", "run_tests"]
    desc = tools[1].description
    for words in ("no network", "search_code", "at most 3 times", "12,000 characters"):
        assert words in desc
    with pytest.raises(ValueError, match="unknown tools"):
        make_tools(ctx, ["run_shell"])


def test_tool_invocation_reports_into_notes(ctx, monkeypatch):
    def down(image):
        raise SandboxUnavailable()

    monkeypatch.setattr(rt, "check_image", down)
    notes = ToolNotes()
    (tool,) = make_tools(ctx, ["run_tests"], notes)
    assert tool.invoke({"test_ids": ["tests/test_summary.py"]}) == (
        "ERROR: Docker Desktop is not running"
    )
    assert notes.unavailable is True
