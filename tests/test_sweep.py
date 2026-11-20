"""The sweep detector in eval/sweep.py."""

import pytest

from eval.sweep import SWEEP_THRESHOLD, is_trivial, normalise, strip_annotations, sweep_stats


def diff(path: str, removed: list[str], added: list[str], context: list[str] = ()) -> str:
    ctx = list(context)
    old_n, new_n = len(ctx) + len(removed), len(ctx) + len(added)
    body = "".join(f" {c}\n" for c in ctx)
    body += "".join(f"-{r}\n" for r in removed) + "".join(f"+{a}\n" for a in added)
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -1,{old_n} +1,{new_n} @@\n{body}"
    )


def everything(_path: str) -> bool:
    return True


# --- normalise -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("def f(a, b=1):", "def f(a: int, b: dict[str, int] = 1) -> list[int]:"),
        ("    timeout=1.0,", "    timeout: float = 1.0,"),
        ("    *args,", "    *args: t.Any,"),
        ("x = 'a'", 'x = "a"'),
        ("x = 'old message'", "x = f'new message {y}'"),
        ("self.x = 0", "self.x: int = 0"),
        ("x = f(y)", "x = f(y)  # type: ignore[arg-type]"),
        ("x = f(y)", "x = f( y )"),
        (") :", ") -> None:"),
    ],
)
def test_normalise_equates_mechanical_changes(before, after):
    assert normalise(before) == normalise(after)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("if x > 0:", "if x >= 0:"),
        ("return items[1:]", "return items[:-1]"),
        ("d = {a: 1}", "d = {a: 2}"),
        ("name: 5,", "name: 6,"),  # a dict entry, not an annotation
        ("x = y", "x = y or None"),
    ],
)
def test_normalise_keeps_logic_changes(before, after):
    assert normalise(before) != normalise(after)


def test_strip_annotations_leaves_slices_lambdas_and_walrus():
    for code in ("y = x[1:2]", "f = lambda q: q", "if (n := len(a)) > 3: pass"):
        assert strip_annotations(code) == code


# --- is_trivial ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "",
        "    ",
        "import os",
        "from typing import Any",
        "@functools.cache",
        "conn: sqlite3.Connection",
        '"""Docstring only."""',
    ],
)
def test_structurally_trivial_without_removed_lines(line):
    assert is_trivial(line, [], in_import=False, in_docstring=False)


def test_lines_inside_imports_and_docstrings_are_trivial():
    assert is_trivial("    Any,", [], in_import=True, in_docstring=False)
    assert is_trivial("Some prose about x.", [], in_import=False, in_docstring=True)


def test_new_logic_without_removed_lines_is_not_trivial():
    assert not is_trivial("    return None", [], in_import=False, in_docstring=False)


def test_reflowed_line_is_trivial():
    removed = ["result = compute(alpha, beta)"]
    for piece in ("result = compute(", "    alpha,", "    beta,", ")"):
        assert is_trivial(piece, removed, in_import=False, in_docstring=False)


# --- sweep_stats -----------------------------------------------------------------------------


def test_annotation_sweep_detected():
    text = diff(
        "pkg/a.py",
        removed=["def f(a, b=None):", "    x = []"],
        added=["def f(a: int, b: str | None = None) -> list[int]:", "    x: list[int] = []"],
    )
    stats = sweep_stats(text, everything)
    assert (stats.trivial, stats.total) == (2, 2)
    assert stats.is_sweep


def test_logic_change_not_a_sweep():
    text = diff(
        "pkg/a.py",
        removed=["    if n > limit:", "        return items[1:]"],
        added=["    if n >= limit:", "        return items[:-1]"],
    )
    stats = sweep_stats(text, everything)
    assert stats.trivial == 0 and not stats.is_sweep


def test_import_block_and_decorator_count_as_trivial():
    text = diff(
        "pkg/a.py",
        removed=[],
        added=["from typing import (", "    Any,", "    cast,", ")", "@overload"],
        context=["import os"],
    )
    stats = sweep_stats(text, everything)
    assert (stats.trivial, stats.total) == (5, 5)


def test_threshold_boundary():
    trivial = ["import a", "import b", "import c", "import d", "import e", "import f", "import g"]
    logic = ["x = compute()", "y = other()", "z = third()"]
    at = sweep_stats(diff("pkg/a.py", [], trivial + logic), everything)  # 7 of 10
    below = sweep_stats(diff("pkg/a.py", [], trivial[:6] + logic), everything)  # 6 of 9
    assert at.ratio == pytest.approx(SWEEP_THRESHOLD) and at.is_sweep
    assert below.ratio < SWEEP_THRESHOLD and not below.is_sweep


def test_include_filter_excludes_files():
    text = diff("tests/test_a.py", [], ["import os"]) + diff("pkg/a.py", ["x = 1"], ["x = 2"])
    stats = sweep_stats(text, lambda p: not p.startswith("tests/"))
    assert (stats.trivial, stats.total) == (0, 1)


def test_no_added_lines_is_not_a_sweep():
    stats = sweep_stats(diff("pkg/a.py", ["x = 1"], []), everything)
    assert stats.total == 0 and not stats.is_sweep
