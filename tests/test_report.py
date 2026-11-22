"""eval/report.py on the synthetic scorer dataset: every number is checked against a hand count."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import report as rp

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def finding(**kw) -> dict:
    base = {
        "file": "pkg/util.py", "line_start": 2, "line_end": 2, "category": "off_by_one",
        "severity": "high", "confidence": 0.9, "title": "t", "explanation": "e",
        "evidence": "return xs[len(xs)]", "agent": "single", "model": "m",
    }  # fmt: skip
    base.update(kw)
    return base


HIT = finding()
STRAY = finding(file="pkg/io.py", line_start=2, line_end=2, category="wrong_comparison")


def result_block(calls=5, tin=1000, tout=100, net=4.0, wait=1.5, wall=0.5, **extra) -> dict:
    return {
        "status": "complete", "model_calls": calls, "tool_calls": 3, "input_tokens": tin,
        "output_tokens": tout, "net_duration_s": net, "wait_s": wait, "wait_s_wall": wall, **extra,
    }  # fmt: skip


def orchestrated(**kw) -> dict:
    child = {"agent": "bug_reviewer", "model_calls": 3, "input_tokens": 600, "output_tokens": 50}
    spawned = {
        "agent": "test_reviewer", "own_model_calls": 1, "own_input_tokens": 100,
        "own_output_tokens": 10, "model_calls": 3, "input_tokens": 300, "output_tokens": 30,
        "children": [{"agent": "test_reviewer/sub1", "model_calls": 2, "input_tokens": 200,
                      "output_tokens": 20}],
    }  # fmt: skip
    return result_block(
        own_model_calls=1,
        own_input_tokens=100,
        own_output_tokens=20,
        children=[child, spawned],
        **kw,
    )


def save(results: Path, config: str, item: str, findings: list[dict], result=None, **top):
    (results / config / "runs").mkdir(parents=True, exist_ok=True)
    (results / config / f"{item}.json").write_text(
        json.dumps({"item_id": item, "config": config, "findings": findings}), encoding="utf-8"
    )
    run = {"result": result or result_block(), "scoring": {"decision": "scored", "cause": "x"}}
    run["run_log"] = [{"reason": "first run", "scoring": run["scoring"]}]
    run.update(top)
    (results / config / "runs" / f"{item}.json").write_text(json.dumps(run), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    dataset, results = tmp_path / "dataset", tmp_path / "results"
    shutil.copytree(FIXTURES / "dataset", dataset)
    sub_hit = {**HIT, "agent": "bug_reviewer/sub1"}
    sub_stray = {**STRAY, "agent": "bug_reviewer/sub2"}
    save(results, "single", "item_001", [HIT])
    save(results, "single", "item_002", [STRAY])
    save(results, "orchestrator", "item_001", [], orchestrated())
    save(results, "orchestrator", "item_002", [], orchestrated())
    save(results, "subagents", "item_001", [sub_hit], orchestrated())
    save(results, "subagents", "item_002", [sub_stray], orchestrated())
    cum = {"model_calls": 9, "tool_calls": 6, "input_tokens": 2000, "output_tokens": 200,
           "duration_s": 20, "wait_s_sum": 3, "wait_s_wall": 2, "net_duration_s": 8.0}  # fmt: skip
    save(results, "verifier", "item_001", [sub_hit], result_block(calls=1), cumulative=cum)
    save(results, "verifier", "item_002", [], result_block(calls=1), cumulative=cum)
    return dataset, results


NOTES = {"item_001/b2": "pkg/calc.py"}


def test_incomplete_configuration_gets_no_row(world):
    dataset, results = world
    text, configs = rp.build_report(results, dataset, NOTES)
    assert [c.name for c in configs] == ["single", "orchestrator", "subagents", "verifier"]
    assert "Not complete (no row): judge." in text


def test_main_table_numbers(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    by = {c.name: c for c in configs}
    single = rp.metric_row(by["single"], dataset, "all")
    # 2 planted bugs, 1 hit; 1 tp + 1 fp (the stray on the clean PR); 1 false alarm / 1 clean PR.
    assert single[2:6] == ["0.50", "0.50 / -", "1.00", "0.50 / 0.50"]
    assert single[6] == "1K in, 100 out"
    assert single[7] == "$0.0004"
    assert single[8] == "4.0 s" and single[9] == "1.5 s"
    orch = rp.metric_row(by["orchestrator"], dataset, "all")
    assert orch[2] == "0.00" and orch[9] == "0.5 s"  # orchestrated rows use wait_s_wall
    ver = rp.metric_row(by["verifier"], dataset, "all")
    assert ver[6] == "2K in, 200 out"  # cumulative cost
    assert ver[8] == "8.0 s" and ver[9] == "2.0 s"
    assert ver[4] == "0.00"  # the verifier dropped the stray on the clean PR


def test_bucket_tables_split_by_size(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    single = next(c for c in configs if c.name == "single")
    small = rp.metric_row(single, dataset, "small")
    assert small[1] == "2"  # both fixture items are small
    assert rp.metric_row(single, dataset, "large")[1] == "0"


def test_per_kind_and_cross_file(world):
    dataset, results = world
    text, configs = rp.build_report(results, dataset, NOTES)
    by = {c.name: c for c in configs}
    truth = {
        i: {
            b["bug_id"]: b["kind"]
            for b in json.loads((dataset / i / "ground_truth.json").read_bytes())["bugs"]
        }
        for i in ("item_001", "item_002")
    }
    kinds = rp.per_kind(by["single"], truth)
    assert kinds["off_by_one"] == (1, 1) and kinds["missing_none_check"] == (0, 1)
    assert rp.per_kind(by["orchestrator"], truth)["off_by_one"] == (0, 1)
    assert "## Recall on the 1 cross-file bugs" in text
    assert "| item_001/b2 | pkg/calc.py | hit | miss | hit | hit |" in text


def test_hits_found_by_sub_agents(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    by = {c.name: c for c in configs}
    hits = rp.subagent_only_hits(by["subagents"], by["orchestrator"])
    assert hits == [("item_001", "b2", "bug_reviewer/sub1", True)]
    assert rp.subagent_only_hits(by["single"], by["orchestrator"]) == []  # agent "single"


def test_agent_breakdown_table(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    rows = {r[0]: r for r in rp.agent_table(next(c for c in configs if c.name == "subagents"))}
    assert rows["orchestrator"][1] == "2"  # 1 own call x 2 reviews
    assert rows["bug_reviewer"][1] == "6" and rows["test_reviewer"][1] == "2"
    assert rows["test_reviewer/sub"][1] == "4" and rows["test_reviewer/sub"][3] == "400"


def test_judge_drops_split_hits_and_false_alarms(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    by = {c.name: c for c in configs}
    d = rp.judge_drops(results, dataset, by["verifier"], by["subagents"])
    assert d == {"input": 2, "kept": 1, "dropped": 1, "hits": 0, "false_alarms": 1, "other": 0,
                 "bugs_lost": 0}  # fmt: skip


def test_a_judge_that_drops_a_hit_loses_recall(world):
    dataset, results = world
    save(results, "verifier", "item_001", [], result_block(calls=1), cumulative={
        "model_calls": 9, "tool_calls": 6, "input_tokens": 2000, "output_tokens": 200,
        "duration_s": 20, "wait_s_sum": 3, "wait_s_wall": 2, "net_duration_s": 8.0,
    })  # fmt: skip
    _, configs = rp.build_report(results, dataset, NOTES)
    by = {c.name: c for c in configs}
    d = rp.judge_drops(results, dataset, by["verifier"], by["subagents"])
    assert d["hits"] == 1 and d["bugs_lost"] == 1 and d["dropped"] == 2


def test_adjusted_precision_appears_with_labels(world):
    dataset, results = world
    (results / "single" / "manual_labels.json").write_text(
        json.dumps({"item_002#0": "real_bug"}), encoding="utf-8"
    )
    _, configs = rp.build_report(results, dataset, NOTES)
    single = next(c for c in configs if c.name == "single")
    assert rp.metric_row(single, dataset, "all")[3] == "0.50 / 1.00"


def test_reused_and_rerun_items_are_listed(world):
    dataset, results = world
    save(results, "subagents", "item_001", [], orchestrated(), reused_from="orchestrator")
    run = json.loads((results / "single" / "runs" / "item_002.json").read_bytes())
    run["run_log"] = [
        {"reason": "first run", "scoring": {"decision": "rerun_required", "cause": "quota_paused"}},
        {"reason": "rerun after quota", "scoring": {"decision": "scored", "cause": "complete"}},
    ]
    (results / "single" / "runs" / "item_002.json").write_text(json.dumps(run), encoding="utf-8")
    text, _ = rp.build_report(results, dataset, NOTES)
    assert "reused from `orchestrator` (1): item_001" in text
    assert "`single` item_002: 2 recorded run(s)" in text and "quota_paused" in text


def test_readme_table_is_replaced_in_place(world):
    dataset, results = world
    _, configs = rp.build_report(results, dataset, NOTES)
    readme = (
        "# T\n\n## Results\n\ntext\n\n"
        "| Configuration | Recall | Precision (strict / adjusted) | False alarms per clean PR | Cost per review | Median time |\n"  # noqa: E501
        "| --- | --- | --- | --- | --- | --- |\n"
        "| old | row |\n| old2 | row |\n\nDetails: x\n"
    )
    new = rp.update_readme(readme, rp.readme_rows(configs, dataset))
    assert "| old |" not in new and "Details: x" in new
    assert (
        "| One agent | 0.50 (location-only 0.50) | 0.50 / - | 1.00 | $0.0004 (1K input, 100 output tokens) | 4.0 s (+1.5 s waiting) |"  # noqa: E501
        in new
    )
    assert "| Plus orchestrator as judge (optional) |  |  |  |  |  |" in new


def test_main_writes_report_and_readme(world, tmp_path, capsys):
    dataset, results = world
    readme = tmp_path / "README.md"
    readme.write_text(
        "| Configuration | Recall | Precision (strict / adjusted) | x | y | z |\n| --- |\n| a |\n",
        encoding="utf-8",
    )
    rc = rp.main(["--results", str(results), "--dataset", str(dataset), "--readme", str(readme)])
    assert rc == 0
    assert (results / "REPORT.md").read_text(encoding="utf-8").startswith("# Evaluation report")
    assert "| One agent |" in readme.read_text(encoding="utf-8")
