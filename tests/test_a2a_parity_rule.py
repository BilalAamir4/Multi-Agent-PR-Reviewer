"""The live-parity pass rule (stage 3, decision 0011), checked without any model call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_a2a_parity import FIXTURES, difference, matching_size, overlaps, parity


def f(file="a.py", category="sql_injection", start=10, end=10):
    return {"file": file, "category": category, "line_start": start, "line_end": end}


def test_overlap_needs_same_file_same_category_and_overlapping_lines():
    assert overlaps(f(), f())
    assert overlaps(f(start=10, end=20), f(start=20, end=25))
    assert not overlaps(f(start=10, end=19), f(start=20, end=25))
    assert not overlaps(f(file="b.py"), f())
    assert not overlaps(f(category="other"), f())


def test_difference_counts_unmatched_findings_on_both_sides():
    assert difference([], []) == 0
    assert difference([f()], [f()]) == 0
    assert difference([f()], []) == 1
    assert difference([f()], [f(file="b.py")]) == 2
    assert difference([f(), f(file="b.py")], [f()]) == 1


def test_each_finding_is_used_once():
    wide = f(start=1, end=100)
    assert matching_size([wide, wide], [f(start=5, end=5)]) == 1
    assert difference([wide, wide], [f(start=5, end=5)]) == 1
    # an augmenting path: the first x must give way so the second x can match
    xs, ys = [f(start=1, end=10), f(start=1, end=1)], [f(start=1, end=1), f(start=5, end=5)]
    assert matching_size(xs, ys) == 2


def test_remote_equal_to_one_local_passes():
    a, b = [f()], [f(), f(file="b.py", category="other")]
    v = parity(a, b, [f()])
    assert v["matches_a_local_exactly"] and v["passed"] and v["local_vs_local"] == 1


def test_remote_may_differ_as_much_as_local_differs_from_itself():
    a, b = [f()], [f(file="b.py")]  # local vs local: 2
    remote = [f(file="c.py")]  # 2 from each local
    v = parity(a, b, remote)
    assert v["local_vs_local"] == 2 and v["nearest"] == 2 and v["passed"]
    assert not v["matches_a_local_exactly"]


def test_remote_that_differs_more_than_local_from_itself_fails():
    a = b = [f()]  # local vs local: 0
    v = parity(a, b, [f(file="x.py")])
    assert v["local_vs_local"] == 0 and v["nearest"] == 2 and not v["passed"]
    assert not parity([], [], [f()])["passed"]
    assert parity([], [], [])["passed"]


def test_the_recorded_live_parity_run_passed_if_there_is_one():
    path = Path(__file__).resolve().parents[1] / "docs" / "a2a-parity.json"
    if not path.is_file():
        pytest.skip("no live parity run recorded yet (scripts/check_a2a_parity.py)")
    results = json.loads(path.read_bytes().decode("utf-8"))
    assert set(results) == set(FIXTURES)
    for name, r in results.items():
        assert "partial" not in r, f"{name}: the run did not finish"
        assert set(r["runs"]) == {"local_1", "local_2", "remote"}, name
        assert r["parity"]["passed"], name
        assert r["runs"]["remote"]["transport"]["name"] == "a2a", name
        assert all(run["status"] == "complete" for run in r["runs"].values()), name
        # the recorded verdict is the rule applied to the recorded findings
        again = parity(*(r["runs"][k]["findings"] for k in ("local_1", "local_2", "remote")))
        assert again == r["parity"], name
