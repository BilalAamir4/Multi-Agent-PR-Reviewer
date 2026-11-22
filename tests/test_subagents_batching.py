"""The spawn decision and the batching of a specialist's files (stage 4A, decision 0012, Task 2).

Table-driven, no model, network or Docker call. A synthetic workspace and diff are written per
case: every file is added with a chosen number of lines, so its estimated tokens are known to the
order of magnitude (``estimate_tokens`` is ``ceil(len / 3)`` over the numbered diff only; whole
file contents are not counted, decision 0012 stage 4C).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from reviewer.agents.subagents import (
    make_batches,
    plan_batches,
    should_spawn_for,
    specialist_area,
)
from reviewer.config import SpawnConfig, load_review_config
from reviewer.tools import ReviewContext

CFG: SpawnConfig = load_review_config().spawn  # 4 files / 30000 tokens, 3 per batch, 5 batches
SMALL = 10  # lines: about 100 tokens of numbered diff
HUGE = 2600  # lines: about 16,000 tokens of diff; two of them are over 30,000
OVER = 3500  # lines: about 21,500 tokens, over batch_token_budget (20,000) on its own


def _added(path: str, lines: int) -> str:
    body = "".join(f"+v{i} = {i}\n" for i in range(lines))
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{lines} @@\n{body}"
    )


def _deleted(path: str, lines: int) -> str:
    body = "".join(f"-v{i} = {i}\n" for i in range(lines))
    return (
        f"diff --git a/{path} b/{path}\ndeleted file mode 100644\n--- a/{path}\n+++ /dev/null\n"
        f"@@ -1,{lines} +0,0 @@\n{body}"
    )


def _modified(path: str) -> str:
    """A one-line change inside a file that is long on disk (the diff stays small)."""
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        "@@ -1,3 +1,3 @@\n v0 = 0\n-v1 = 1\n+v1 = 100\n v2 = 2\n"
    )


def make_ctx_long_files(tmp: Path, files: dict[str, int]):
    """A context whose files have ``files`` ({path: line count}) lines on disk but whose diff only
    changes one line in each."""
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    diff = ""
    for path, n in files.items():
        target = ws / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"v{i} = {i}\n" for i in range(n)), encoding="utf-8")
        diff += _modified(path)
    return ReviewContext.from_text(ws, diff)


def make_ctx(tmp: Path, files: dict[str, int], deleted: dict[str, int] | None = None):
    """A context whose diff adds ``files`` ({path: line count}) and deletes ``deleted``."""
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    diff = ""
    for path, n in files.items():
        target = ws / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"v{i} = {i}\n" for i in range(n)), encoding="utf-8")
        diff += _added(path, n)
    for path, n in (deleted or {}).items():
        diff += _deleted(path, n)
    return ReviewContext.from_text(ws, diff)


def groups_of(tmp, files, cfg=CFG, **kw):
    ctx = make_ctx(tmp, files, **kw)
    return [list(b.files) for b in make_batches(specialist_area("bug_reviewer", ctx), ctx, cfg)]


def py(*names: str, n: int = SMALL) -> dict[str, int]:
    return {name: n for name in names}


# --- Should it spawn? ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "spawn", "triggers"),
    [
        pytest.param(py("a.py", "b.py", "c.py"), False, [], id="3 files: no spawn"),
        pytest.param(py("a.py", "b.py", "c.py", "d.py"), True, ["files"], id="4 files: by count"),
        pytest.param(
            {"a.py": HUGE, "b.py": HUGE}, True, ["tokens"], id="2 files with a huge diff: by tokens"
        ),
        pytest.param(
            {"a.py": HUGE, "b.py": HUGE, "c.py": SMALL, "d.py": SMALL},
            True,
            ["files", "tokens"],
            id="both",
        ),
        pytest.param({"a.py": 1200, "b.py": 10}, False, [], id="big but under the limit"),
        pytest.param(py("a.py", "b.py", "c.py", "README.md", "x.txt"), False, [], id="non-py"),
    ],
)
def test_spawn_decision(tmp_path, files, spawn, triggers):
    ctx = make_ctx(tmp_path, files)
    d = should_spawn_for("bug_reviewer", ctx, CFG)
    assert d.spawn is spawn
    assert list(d.triggers) == triggers
    assert d.py_files == len(d.area) == sum(1 for f in files if f.endswith(".py"))
    assert (d.min_py_files, d.max_specialist_prompt_tokens) == (4, 30000)
    assert (d.estimated_tokens > 30000) == ("tokens" in triggers)


def test_huge_file_contents_with_a_small_diff_do_not_spawn(tmp_path):
    ctx = make_ctx_long_files(tmp_path, {"a.py": 20000, "b.py": 20000})  # ~200k tokens on disk
    d = should_spawn_for("bug_reviewer", ctx, CFG)
    assert not d.spawn and list(d.triggers) == []
    assert 0 < d.estimated_tokens < 1000


def test_the_decision_records_both_values(tmp_path):
    d = should_spawn_for("bug_reviewer", make_ctx(tmp_path, py("a.py", "b.py")), CFG)
    assert (d.py_files, d.min_py_files) == (2, 4)
    assert 0 < d.estimated_tokens < d.max_specialist_prompt_tokens


def test_deleted_and_non_python_files_are_not_in_the_area(tmp_path):
    ctx = make_ctx(
        tmp_path, {"a.py": SMALL, "b.py": SMALL, "README.md": SMALL, "docs/x.rst": SMALL},
        deleted={"old.py": SMALL, "gone/more.py": SMALL},
    )  # fmt: skip
    for name in ("bug_reviewer", "security_reviewer", "test_reviewer"):
        assert specialist_area(name, ctx) == ["a.py", "b.py"]
    assert not should_spawn_for("bug_reviewer", ctx, CFG).spawn  # 2 files, not 4


def test_the_decision_ignores_the_switch(tmp_path):
    """The D6 reuse rule asks what would spawn, whatever ``enabled`` says."""
    ctx = make_ctx(tmp_path, py("a.py", "b.py", "c.py", "d.py"))
    off = CFG.model_copy(update={"enabled": False})
    assert should_spawn_for("bug_reviewer", ctx, off).spawn


# --- Batches ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            py("pkg/foo.py", "tests/test_foo.py", "pkg/bar.py"),
            [["pkg/bar.py", "pkg/foo.py", "tests/test_foo.py"]],
            id="source with its test (test_foo.py); the folder mate joins as it fits",
        ),
        pytest.param(
            py("pkg/foo.py", "tests/foo_test.py"),
            [["pkg/foo.py", "tests/foo_test.py"]],
            id="source with its test (foo_test.py)",
        ),
        pytest.param(
            py("a/foo.py", "b/foo.py", "tests/test_foo.py"),
            [["a/foo.py", "b/foo.py", "tests/test_foo.py"]],
            id="ambiguous stem: two foo.py, the test is not paired (all fit in one batch)",
        ),
        pytest.param(
            py("pkg/a.py", "pkg/b.py", "other/c.py"),
            [["other/c.py", "pkg/a.py", "pkg/b.py"]],
            id="folder grouping, groups in (directory, first path) order",
        ),
        pytest.param(
            py("pkg/foo.py", "tests/test_foo.py", "pkg/x.py", "pkg/y.py", "pkg/z.py"),
            [["pkg/foo.py", "tests/test_foo.py"], ["pkg/x.py", "pkg/y.py", "pkg/z.py"]],
            id="a related pair and a group of three do not share a batch of three",
        ),
        pytest.param(
            py("p/a.py", "p/b.py", "p/c.py", "p/d.py", "p/e.py"),
            [["p/a.py", "p/b.py", "p/c.py"], ["p/d.py", "p/e.py"]],
            id="a group over the file limit is split in path order",
        ),
        pytest.param(
            py("t/test_a.py", "t/test_b.py"),
            [["t/test_a.py", "t/test_b.py"]],
            id="tests with no source are grouped by folder",
        ),
        pytest.param(
            py("pkg/foo.py", "tests/test_foo.py", "tests/foo_test.py"),
            [["pkg/foo.py", "tests/foo_test.py", "tests/test_foo.py"]],
            id="two tests of one source stay with it",
        ),
    ],
)
def test_batches(tmp_path, files, expected):
    assert groups_of(tmp_path, files) == expected


def test_a_single_oversized_file_gets_a_batch_of_its_own(tmp_path):
    ctx = make_ctx(tmp_path, {"p/a.py": SMALL, "p/big.py": OVER, "p/c.py": SMALL})
    batches = make_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    assert [list(b.files) for b in batches] == [["p/a.py"], ["p/big.py"], ["p/c.py"]]
    assert [b.over_budget for b in batches] == [False, True, False]
    assert batches[1].estimated_tokens > CFG.batch_token_budget


def test_the_token_budget_closes_a_batch_before_the_file_limit(tmp_path):
    files = {"p/a.py": HUGE, "p/b.py": HUGE, "p/c.py": SMALL}  # 16k + 16k > 20k
    assert groups_of(tmp_path, files) == [["p/a.py"], ["p/b.py", "p/c.py"]]


def test_large_file_contents_do_not_force_one_file_per_batch(tmp_path):
    ctx = make_ctx_long_files(tmp_path, {"p/a.py": 20000, "p/b.py": 20000, "p/c.py": 20000})
    batches = make_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    assert [list(b.files) for b in batches] == [["p/a.py", "p/b.py", "p/c.py"]]
    assert not batches[0].over_budget


def test_batches_are_numbered_from_one_in_packing_order(tmp_path):
    ctx = make_ctx(tmp_path, py("b/x.py", "a/y.py", "c/z.py"))
    batches = make_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    assert [b.index for b in batches] == [1]
    assert batches[0].files == ("a/y.py", "b/x.py", "c/z.py")  # (directory, first path)
    ctx2 = make_ctx(tmp_path / "two", py(*[f"d{i}/f.py" for i in range(7)]))
    numbered = make_batches(specialist_area("bug_reviewer", ctx2), ctx2, CFG)
    assert [b.index for b in numbered] == list(range(1, len(numbered) + 1))


@pytest.mark.parametrize(
    ("count", "limit", "n_batches"),
    [(16, 4, 4), (26, 6, 5), (15, 3, 5), (17, 4, 5)],
    ids=["16 files", "26 files", "15 files fit at 3", "17 files"],
)
def test_the_cap_raises_the_file_limit_until_the_count_fits(tmp_path, count, limit, n_batches):
    files = py(*[f"d{i:02d}/m.py" for i in range(count)])
    ctx = make_ctx(tmp_path, files)
    plan = plan_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    assert plan.file_limit_used == limit
    assert len(plan.batches) == n_batches <= CFG.max_subagents
    assert plan.budget_exceeded is (limit > CFG.max_files_per_subagent)
    assert not plan.token_budget_ignored
    assert sorted(f for b in plan.batches for f in b.files) == sorted(files)
    assert all(len(b.files) <= limit for b in plan.batches)


def test_when_the_token_budget_alone_keeps_the_count_over_the_cap_it_is_ignored(tmp_path):
    ctx = make_ctx(tmp_path, {f"d{i}/m.py": OVER for i in range(7)})  # 7 files over the budget
    plan = plan_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    assert plan.budget_exceeded and plan.token_budget_ignored
    assert len(plan.batches) <= CFG.max_subagents
    assert sorted(f for b in plan.batches for f in b.files) == sorted(
        f"d{i}/m.py" for i in range(7)
    )


def test_non_python_and_deleted_files_are_never_batched(tmp_path):
    ctx = make_ctx(
        tmp_path,
        {**py("a.py", "b.py", "c.py", "d.py", "README.md"), "x.cfg": 5},
        deleted={"old.py": SMALL},
    )  # fmt: skip
    plan = plan_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
    files = [f for b in plan.batches for f in b.files]
    assert sorted(files) == ["a.py", "b.py", "c.py", "d.py"]


def test_the_same_input_gives_the_same_batches_in_every_run(tmp_path):
    names = [f"{d}/{n}.py" for d in "abcd" for n in ("x", "y", "test_x")]
    results = []
    for seed in range(5):
        shuffled = names[:]
        random.Random(seed).shuffle(shuffled)
        ctx = make_ctx(tmp_path / f"run{seed}", {n: SMALL for n in shuffled})
        plan = plan_batches(specialist_area("bug_reviewer", ctx), ctx, CFG)
        results.append([(b.index, b.files, b.estimated_tokens) for b in plan.batches])
    assert all(r == results[0] for r in results)
    # and twice on one context
    ctx = make_ctx(tmp_path / "again", {n: SMALL for n in names})
    again = [plan_batches(specialist_area("bug_reviewer", ctx), ctx, CFG) for _ in range(2)]
    assert again[0] == again[1]


def test_no_files_no_batches(tmp_path):
    ctx = make_ctx(tmp_path, {"README.md": SMALL})
    assert make_batches(specialist_area("bug_reviewer", ctx), ctx, CFG) == []


def test_the_test_path_rule_is_the_one_from_find_candidates():
    from eval.find_candidates import is_test_path
    from reviewer.agents import subagents

    for p in ("tests/a.py", "pkg/test_a.py", "pkg/a_test.py", "conftest.py", "pkg/testing.py"):
        assert subagents._is_test_path(p) == is_test_path(p)
