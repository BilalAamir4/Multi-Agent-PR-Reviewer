"""eval/plan_runs.py and eval/results_data.py on a synthetic dataset and results folder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval import plan_runs
from eval import results_data as rd


def diff_with(n_py: int) -> str:
    out = []
    for k in range(n_py):
        out.append(
            f"diff --git a/f{k}.py b/f{k}.py\n--- a/f{k}.py\n+++ b/f{k}.py\n@@ -1 +1,2 @@\n a\n+b\n"
        )
    return "".join(out)


@pytest.fixture
def world(tmp_path):
    dataset, results = tmp_path / "dataset", tmp_path / "results"
    sizes = {"item_001": 1, "item_002": 4, "item_003": 8, "item_004": 8, "item_005": 8}
    for item, n in sizes.items():
        d = dataset / item
        d.mkdir(parents=True)
        (d / "pr.diff").write_text(diff_with(n), encoding="utf-8", newline="\n")
        (d / "ground_truth.json").write_text("{}", encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"pr_title": f"title {item}"}), encoding="utf-8")
    return dataset, results, list(sizes)


def write(
    results: Path,
    config: str,
    item: str,
    *,
    calls=10,
    scoring=None,
    reused=None,
    findings=(),
    scored=True,
    started="2026-11-22T20:00:00+00:00",
):
    run = {
        "result": {"model_calls": calls, "status": "complete"},
        "run_log": [{"started_at": started, "model_calls": calls}],
    }
    if scoring:
        run["scoring"] = scoring
    if reused:
        run["reused_from"] = reused
        run["run_log"] = [{"started_at": started, "reused_from": reused}]
    p = results / config / "runs" / f"{item}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run), encoding="utf-8")
    if scored:
        (results / config / f"{item}.json").write_text(
            json.dumps({"item_id": item, "config": config, "findings": list(findings)}),
            encoding="utf-8",
        )


def test_status_values(world):
    dataset, results, ids = world
    write(results, "single", "item_001")
    write(
        results,
        "single",
        "item_002",
        scoring={"decision": "rerun_required", "cause": "x"},
        scored=False,
    )
    write(results, "subagents", "item_001", reused="orchestrator")
    assert rd.item_status(results, "single", "item_001") == ("scored", None)
    assert rd.item_status(results, "single", "item_002") == ("rerun_required", "x")
    assert rd.item_status(results, "single", "item_003") == ("missing", None)
    assert rd.item_status(results, "subagents", "item_001") == ("reused", "orchestrator")


def test_latest_rerun_decision_wins_over_stale_scored_file(world):
    dataset, results, ids = world
    write(results, "single", "item_001", scoring={"decision": "rerun_required", "cause": "boom"})
    assert rd.item_status(results, "single", "item_001") == ("rerun_required", "boom")


def test_estimates_use_defaults_then_observed_means(world):
    dataset, results, ids = world
    assert rd.estimate_calls(results, dataset, "orchestrator", "item_003", ids) == 70
    assert rd.estimate_calls(results, dataset, "orchestrator", "item_001", ids) == 10
    assert rd.estimate_calls(results, dataset, "orchestrator", "item_002", ids) == 30
    write(results, "orchestrator", "item_003", calls=40)
    write(results, "orchestrator", "item_004", calls=50)
    assert rd.estimate_calls(results, dataset, "orchestrator", "item_005", ids) == 70  # only 2
    write(results, "orchestrator", "item_005", calls=60)
    assert rd.observed_mean(results, dataset, "orchestrator", "large", ids) == 50
    assert rd.observed_mean(results, dataset, "orchestrator", "small", ids) is None


def test_subagents_estimate_is_zero_for_items_that_will_be_reused(world):
    dataset, results, ids = world
    assert rd.estimate_calls(results, dataset, "subagents", "item_001", ids) == 0  # 1 py file
    assert rd.estimate_calls(results, dataset, "subagents", "item_002", ids) == 30  # 4 py files
    assert rd.would_spawn_estimate(dataset, "item_002") is True


def test_reused_items_do_not_feed_the_observed_mean(world):
    dataset, results, ids = world
    for i in ("item_003", "item_004"):
        write(results, "subagents", i, calls=60)
    write(results, "subagents", "item_005", calls=999, reused="orchestrator")
    assert rd.observed_mean(results, dataset, "subagents", "large", ids) is None


def test_verifier_and_judge_estimates_follow_the_findings(world):
    dataset, results, ids = world
    f = {"file": "x"}
    write(results, "subagents", "item_003", findings=[f, f])
    assert rd.estimate_calls(results, dataset, "verifier", "item_003", ids) == 6
    assert rd.estimate_calls(results, dataset, "verifier", "item_004", ids) == 0
    assert rd.estimate_calls(results, dataset, "judge", "item_003", ids) == 8
    assert rd.estimate_calls(results, dataset, "judge", "item_004", ids) == 0


def test_calls_today_counts_the_pacific_day_and_skips_reused(world):
    dataset, results, ids = world
    write(results, "single", "item_001", calls=7, started="2026-11-22T20:00:00+00:00")
    write(results, "single", "item_002", calls=5, started="2026-11-20T20:00:00+00:00")
    write(
        results, "subagents", "item_001", reused="orchestrator", started="2026-11-22T21:00:00+00:00"
    )
    assert rd.calls_on_pacific_day(results, "2026-11-22") == 7
    assert rd.calls_on_pacific_day(results, "2026-11-20") == 5
    # 02:00 UTC on the 23rd is still the 22nd in Pacific time.
    write(results, "single", "item_003", calls=3, started="2026-11-23T02:00:00+00:00")
    assert rd.calls_on_pacific_day(results, "2026-11-22") == 10


def test_next_command_follows_the_d6_order(world):
    dataset, results, ids = world
    for i in ids:
        write(results, "single", i)
    text = plan_runs.format_plan(results, dataset, budget=100, day="2026-11-22")
    assert "Next: uv run python -m eval.run --config orchestrator --max-calls" in text
    for i in ids:
        write(results, "orchestrator", i)
        write(results, "subagents", i)
        write(results, "verifier", i)
    text = plan_runs.format_plan(results, dataset, budget=100, day="2026-11-22")
    assert "--config judge" in text and "optional" in text
    for i in ids:
        write(results, "judge", i)
    assert "nothing left to run" in plan_runs.format_plan(results, dataset, 100, "2026-11-22")


def test_plan_shows_rerun_cause_and_budget(world):
    dataset, results, ids = world
    write(
        results,
        "single",
        "item_001",
        scoring={"decision": "rerun_required", "cause": "boom"},
        started="2026-01-01T00:00:00+00:00",
    )
    write(results, "single", "item_002", calls=30, started=datetime.now(UTC).isoformat())
    text = plan_runs.format_plan(results, dataset, budget=100)
    assert "RERUN (boom)" in text
    assert "recorded today (Pacific day): 30; budget 100, left today 70" in text
    assert "--config single --max-calls 70" in text


def test_main_prints_the_plan(world, capsys):
    dataset, results, ids = world
    assert plan_runs.main(["--results", str(results), "--dataset", str(dataset)]) == 0
    assert "Next:" in capsys.readouterr().out
