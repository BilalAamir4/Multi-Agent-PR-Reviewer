import json
import shutil
from pathlib import Path

import pytest

from eval.ground_truth import DatasetError, GroundTruth, PlantedBug, load_item
from eval.score import (
    LINE_TOLERANCE,
    aggregate,
    changed_lines,
    is_in_diff,
    main,
    match,
    score_config,
    score_item,
)
from reviewer.schemas import Finding

FIXTURES = Path(__file__).parent / "fixtures"
DATASET = FIXTURES / "dataset"
RESULTS = FIXTURES / "results"

# item_001: bug b1 = pkg/calc.py 11-12 missing_none_check (+ unhandled_exception),
#           bug b2 = pkg/util.py 2 off_by_one.
# Changed lines: pkg/calc.py {11, 12, 42, 59}, pkg/util.py {1, 2, 3}, README.md {2}.
GT_001, DIFF_001 = load_item(DATASET / "item_001")
GT_002, DIFF_002 = load_item(DATASET / "item_002")


def f(file="pkg/calc.py", start=11, end=None, category="missing_none_check") -> Finding:
    return Finding(
        file=file,
        line_start=start,
        line_end=start if end is None else end,
        category=category,
        severity="high",
        confidence=0.8,
        title="t",
        explanation="e",
        evidence="x",
        agent="a",
        model="m",
    )


def b(bug_id, start, end=None, kind="off_by_one", file="pkg/calc.py") -> PlantedBug:
    return PlantedBug(
        bug_id=bug_id,
        file=file,
        line_start=start,
        line_end=start if end is None else end,
        kind=kind,
        accepted_kinds=[kind],
        description="d",
    )


def only_b1() -> GroundTruth:
    return GT_001.model_copy(update={"bugs": [GT_001.bugs[0]]})


def test_tolerance_constant():
    assert LINE_TOLERANCE == 3


def test_changed_lines_fixture():
    changed = changed_lines(DIFF_001)
    assert changed == {
        "pkg/calc.py": {11, 12, 42, 59},
        "pkg/util.py": {1, 2, 3},
        "README.md": {2},
    }


# --- the matching cases ----------------------------------------------------------------------


def test_exact_location_right_kind_hits_both_modes():
    r = score_item(only_b1(), DIFF_001, [f(start=11, end=12, category="missing_none_check")])
    assert r["modes"]["strict"]["tp"] == 1
    assert r["modes"]["location_only"]["tp"] == 1
    assert r["modes"]["strict"]["matched"] == [["b1", "item_001#0"]]


def test_tolerance_boundary_after_bug():
    bug = b("b1", 11, 12)
    assert match([bug], [f(start=15, category="off_by_one")], "strict") == [("b1", 0)]
    assert match([bug], [f(start=16, category="off_by_one")], "strict") == []


def test_tolerance_boundary_before_bug():
    bug = b("b1", 11, 12)
    assert match([bug], [f(start=8, category="off_by_one")], "location_only") == [("b1", 0)]
    assert match([bug], [f(start=7, category="off_by_one")], "location_only") == []


def test_in_diff_tolerance_boundary():
    changed = {"pkg/calc.py": {12}}
    assert is_in_diff(f(start=15), changed)
    assert not is_in_diff(f(start=16), changed)
    assert is_in_diff(f(start=1, end=9), changed)
    assert not is_in_diff(f(start=1, end=8), changed)


def test_wrong_kind_is_strict_miss_location_hit():
    r = score_item(only_b1(), DIFF_001, [f(category="sql_injection")])
    assert r["modes"]["strict"]["tp"] == 0
    assert r["modes"]["strict"]["fp"] == 1
    assert r["modes"]["location_only"]["tp"] == 1
    assert r["modes"]["location_only"]["fp"] == 0


def test_equivalent_kind_is_strict_hit():
    r = score_item(only_b1(), DIFF_001, [f(category="unhandled_exception")])
    assert r["modes"]["strict"]["tp"] == 1


def test_other_never_strict_hit():
    r = score_item(only_b1(), DIFF_001, [f(start=11, end=12, category="other")])
    assert r["modes"]["strict"]["tp"] == 0
    assert r["modes"]["strict"]["fp"] == 1
    assert r["modes"]["location_only"]["tp"] == 1


def test_far_unchanged_line_discarded_not_fp():
    r = score_item(only_b1(), DIFF_001, [f(start=30)])
    assert r["discarded"] == 1
    assert r["discarded_findings"] == ["item_001#0"]
    for mode in ("strict", "location_only"):
        assert r["modes"][mode]["fp"] == 0
        assert r["modes"][mode]["fn"] == 1


def test_file_not_in_diff_discarded():
    r = score_item(only_b1(), DIFF_001, [f(file="pkg/other.py", start=11)])
    assert r["discarded"] == 1
    assert r["modes"]["strict"]["fp"] == 0


def test_deleted_file_not_in_changed_lines():
    assert "pkg/old.py" not in changed_lines(DIFF_001)
    r = score_item(only_b1(), DIFF_001, [f(file="pkg/old.py", start=1)])
    assert r["discarded"] == 1


def test_removed_only_hunk_is_in_diff():
    changed = changed_lines(DIFF_001)
    # Hunk "@@ -40,5 +41,3 @@" removes two lines; they map to the next target line, 42.
    assert 42 in changed["pkg/calc.py"]
    assert is_in_diff(f(start=42), changed)
    # Hunk "@@ -60,3 +59,1 @@" ends with removals; they map to the hunk's last target line.
    assert 59 in changed["pkg/calc.py"]
    assert is_in_diff(f(start=59), changed)
    assert not is_in_diff(f(start=50), changed)


def test_removed_lines_at_hunk_end_use_last_target_line():
    diff = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -20,4 +20,2 @@\n a = 1\n b = 2\n-c = 3\n-d = 4\n"
    )
    assert changed_lines(diff) == {"m.py": {21}}


def test_two_findings_on_one_bug():
    r = score_item(only_b1(), DIFF_001, [f(start=11), f(start=12)])
    s = r["modes"]["strict"]
    assert (s["tp"], s["fp"], s["duplicates"]) == (1, 1, 1)
    assert s["duplicate_findings"] == ["item_001#1"]


@pytest.mark.parametrize("reverse", [False, True])
def test_bipartite_maximum_matching(reverse):
    bugs = [b("b1", 11, 12), b("b2", 17, 18)]
    a = f(start=14, end=15, category="off_by_one")  # overlaps b1 and b2
    bb = f(start=10, end=10, category="off_by_one")  # overlaps b1 only
    findings = [bb, a] if reverse else [a, bb]
    pairs = match(bugs, findings, "strict")
    assert len(pairs) == 2
    idx_a, idx_b = (1, 0) if reverse else (0, 1)
    assert pairs == [("b1", idx_b), ("b2", idx_a)]


def test_clean_item_false_alarms():
    findings = [f(file="pkg/io.py", start=2), f(file="pkg/io.py", start=1, end=2)]
    item = score_item(GT_002, DIFF_002, findings)
    agg = aggregate([item], None)
    assert agg["strict"]["all"]["false_alarms_per_clean_pr"] == 2
    assert agg["strict"]["all"]["recall"] is None
    assert agg["strict"]["all"]["precision"] == 0


# --- score_config and labels -----------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(DATASET, tmp_path / "dataset")
    shutil.copytree(RESULTS, tmp_path / "results")
    return tmp_path


def test_score_config_fixture(workspace):
    report = score_config(workspace / "dataset", workspace / "results" / "demo")
    s = report["aggregates"]["strict"]["all"]
    loc = report["aggregates"]["location_only"]["all"]
    assert (s["tp"], s["fp"], s["fn"], s["discarded"], s["items"]) == (1, 3, 1, 1, 2)
    assert s["recall"] == 0.5
    assert s["precision"] == 0.25
    assert s["false_alarms_per_clean_pr"] == 2
    assert s["adjusted_precision"] is None
    assert s["unlabelled_unmatched"] == 3
    assert (loc["tp"], loc["fp"], loc["fn"]) == (2, 2, 0)
    assert report["aggregates"]["strict"]["large"]["items"] == 0
    assert report["aggregates"]["strict"]["large"]["recall"] is None


def test_adjusted_precision_with_labels(workspace):
    findings_dir = workspace / "results" / "demo"
    labels = {"item_002#0": "real_bug", "item_002#1": "false_alarm"}
    (findings_dir / "manual_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    s = score_config(workspace / "dataset", findings_dir)["aggregates"]["strict"]["all"]
    assert s["precision"] == 0.25
    assert s["adjusted_precision"] == 0.5
    assert s["real_bug_labels"] == 1
    assert s["unlabelled_unmatched"] == 1  # item_001#1 has no label


def test_label_for_unknown_id_is_error(workspace):
    findings_dir = workspace / "results" / "demo"
    (findings_dir / "manual_labels.json").write_text(
        json.dumps({"item_002#7": "real_bug"}), encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="item_002#7"):
        score_config(workspace / "dataset", findings_dir)


def test_missing_findings_file_names_item(workspace):
    (workspace / "results" / "demo" / "item_002.json").unlink()
    with pytest.raises(DatasetError, match="item_002"):
        score_config(workspace / "dataset", workspace / "results" / "demo")


def test_findings_for_unknown_item_is_error(workspace):
    findings_dir = workspace / "results" / "demo"
    data = json.loads((findings_dir / "item_002.json").read_text(encoding="utf-8"))
    data["item_id"] = "item_099"
    (findings_dir / "item_099.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError, match="item_099"):
        score_config(workspace / "dataset", findings_dir)


def test_config_name_mismatch_is_error(workspace):
    findings_dir = workspace / "results" / "demo"
    data = json.loads((findings_dir / "item_002.json").read_text(encoding="utf-8"))
    data["config"] = "other"
    (findings_dir / "item_002.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError, match="config"):
        score_config(workspace / "dataset", findings_dir)


# --- CLI -------------------------------------------------------------------------------------


def _cli(workspace, *extra):
    return main(
        [
            "--config",
            "demo",
            "--dataset",
            str(workspace / "dataset"),
            "--results",
            str(workspace / "results"),
            *extra,
        ]
    )


def test_cli_writes_score_json(workspace, capsys):
    assert _cli(workspace) == 0
    out = workspace / "results" / "demo" / "score.json"
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["config"] == "demo"
    assert b"\r\n" not in out.read_bytes()
    printed = capsys.readouterr().out
    assert "strict" in printed and "location_only" in printed
    # score.json in the findings folder must not break a second run
    assert _cli(workspace) == 0


def test_cli_custom_out(workspace):
    out = workspace / "custom" / "s.json"
    assert _cli(workspace, "--out", str(out)) == 0
    assert out.is_file()


def test_cli_malformed_input_exits_2(workspace, capsys):
    (workspace / "results" / "demo" / "item_001.json").write_text("{not json", encoding="utf-8")
    assert _cli(workspace) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert len(err.strip().splitlines()) == 1
