"""``eval.run`` for stage 4B (decision 0013): ``--config verifier`` (row 4) and ``--config judge``
(row 4b) judge the saved ``subagents`` findings, write the kept ones in the scorer's format, record
the step's cost and the cumulative cost, and apply the scoring policy to the judge step. Fake
results and fake agents only: no model, network, Docker or git call."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import run as runner
from reviewer.agents.judging import JudgeStepResult, VerificationRecord, assign_refs
from reviewer.agents.orchestrator import OrchestratorResult
from reviewer.config import load_review_config
from tests.test_eval_run import finding
from tests.test_scoring_policy import LIMITS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ITEMS = ("item_001", "item_002")
KEEP, DROP = finding(title="keep", line_start=2), finding(title="drop", line_start=8, line_end=8)
REVIEW_COST = {
    "model_calls": 20, "tool_calls": 15, "input_tokens": 50_000, "output_tokens": 800,
    "duration_s": 100.0, "wait_s_sum": 60.0, "wait_s_wall": 40.0, "net_duration_s": 60.0,
}  # fmt: skip


def step(kind, findings, reject_titles=(), status="complete", **kw) -> JudgeStepResult:
    refs = assign_refs(findings)
    kept = [f for f in findings if f.title not in reject_titles]
    records = [
        VerificationRecord(
            finding_ref=r, verdict="rejected" if f.title in reject_titles else "confirmed",
            reason="because",
        )
        for r, f in zip(refs, findings, strict=True)
    ]  # fmt: skip
    dropped = [
        {"draft": {**f.model_dump(mode="json"), "finding_id": r},
         "reason": "verifier_rejected" if kind == "verifier" else "judge_rejected",
         "detail": "because"}
        for r, f in zip(refs, findings, strict=True) if f.title in reject_titles
    ]  # fmt: skip
    calls = 0 if not findings else 4
    return JudgeStepResult(
        status=status, agent=kind, judge="verifier" if kind == "verifier" else "orchestrator",
        prompt_version="abc123def456", limits=LIMITS, evaluation=True, model_id="m",
        findings=kept, dropped=dropped, verifications=records, input_findings=len(findings),
        model_calls=calls, tool_calls=calls // 2, input_tokens=1000 * calls,
        output_tokens=50 * calls, duration_s=8.0 if calls else 0.0, wait_s=3.0 if calls else 0.0,
        wait_s_sum=3.0 if calls else 0.0, wait_s_wall=2.0 if calls else 0.0,
        net_duration_s=6.0 if calls else 0.0, **kw,
    )  # fmt: skip


@pytest.fixture
def env(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "dataset", dataset)
    results = tmp_path / "results"
    seen: list[tuple] = []
    outcome: dict = {}

    def fake_verifier(ctx, title, findings, router, *, config, item_id):
        seen.append(("verifier", item_id, [f.title for f in findings], title))
        return outcome.get(item_id) or step("verifier", findings, ["drop"])

    def fake_judge(ctx, plan, summary, findings, router, *, config, item_id):
        seen.append(("judge", item_id, [f.title for f in findings], plan, summary))
        return outcome.get(item_id) or step("judge", findings, ["drop"])

    monkeypatch.setattr(runner, "run_verifier", fake_verifier)
    monkeypatch.setattr(runner, "run_judge", fake_judge)
    monkeypatch.setattr(runner, "item_inputs", lambda item_id: (object(), f"title {item_id}", "b"))
    monkeypatch.setattr(runner, "QuotaRouter", lambda evaluation: None)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)

    def argv(config, *more):
        return ["--config", config, "--dataset", str(dataset), "--results", str(results), *more]

    return argv, results, seen, outcome


def seed(results: Path, findings_by_item: dict[str, list]) -> None:
    """Saved ``subagents`` results: the scored file and the run file of each item."""
    for item, findings in findings_by_item.items():
        out = results / "subagents"
        saved = OrchestratorResult(
            status="complete", findings=list(findings), prompt_version="abc123def456",
            limits=LIMITS, evaluation=True, summary=f"summary of {item}", **REVIEW_COST,
            plan=[{"specialist": "bug_reviewer", "called": True, "reason": "source"}],
        )  # fmt: skip
        (out / "runs").mkdir(parents=True, exist_ok=True)
        (out / "runs" / f"{item}.json").write_text(
            json.dumps({"result": saved.model_dump(mode="json"), "run_log": []}), encoding="utf-8"
        )
        (out / f"{item}.json").write_text(
            json.dumps({"item_id": item, "config": "subagents",
                        "findings": [f.model_dump(mode="json") for f in findings]}),
            encoding="utf-8",
        )  # fmt: skip


def read(path: Path):
    return json.loads(path.read_bytes())


# --- Missing inputs --------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["verifier", "judge"])
def test_missing_subagents_results_stop_the_run_with_a_clear_error(env, capsys, config):
    argv, results, seen, _ = env
    assert runner.main(argv(config)) == 2
    err = capsys.readouterr().err
    assert "run --config subagents first" in err and "item_001" in err
    assert seen == [] and not (results / config).exists()


def test_an_unscored_subagents_item_counts_as_missing(env, capsys):
    argv, results, seen, _ = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    (results / "subagents" / "item_002.json").unlink()  # for example rerun_required
    assert runner.main(argv("verifier")) == 2
    assert "item_002" in capsys.readouterr().err and seen == []


# --- Row 4: the verifier ---------------------------------------------------------------------


def test_verifier_config_judges_the_saved_findings_and_writes_the_kept_ones(env):
    argv, results, seen, _ = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    assert runner.main(argv("verifier")) == 0
    assert seen == [("verifier", i, ["keep", "drop"], f"title {i}") for i in ITEMS]
    out = results / "verifier"
    scored = read(out / "item_001.json")
    assert scored["config"] == "verifier" and scored["item_id"] == "item_001"
    assert [f["title"] for f in scored["findings"]] == ["keep"]
    assert scored["findings"][0]["agent"] == "single"  # unchanged by the judge
    run = read(out / "runs" / "item_001.json")
    assert run["scoring"] == {"decision": "scored", "cause": "complete"}
    assert run["source"] == "subagents" and run["result"]["judge"] == "verifier"
    log = run["run_log"][-1]
    assert (log["input_findings"], log["kept"], log["dropped"]) == (2, 1, 1)
    assert [v["verdict"] for v in log["verdicts"]] == ["confirmed", "rejected"]
    assert log["model_calls"] == 4 and log["prompt_version"] == "abc123def456"


def test_the_run_files_record_the_step_and_the_cumulative_cost(env):
    argv, results, *_ = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    assert runner.main(argv("verifier")) == 0
    run = read(results / "verifier" / "runs" / "item_001.json")
    assert run["review_cost"] == REVIEW_COST
    cum = run["cumulative"]
    assert cum["model_calls"] == 24 and cum["tool_calls"] == 17
    assert cum["input_tokens"] == 54_000 and cum["output_tokens"] == 1000
    assert cum["duration_s"] == 108.0 and cum["net_duration_s"] == 66.0
    assert cum["wait_s_wall"] == 42.0 and cum["wait_s_sum"] == 63.0
    assert run["result"]["model_calls"] == 4  # the judge step alone
    assert run["run_log"][-1]["cumulative"] == cum


def test_items_with_no_findings_make_no_model_call(env):
    argv, results, seen, outcome = env
    seed(results, {"item_001": [], "item_002": [KEEP, DROP]})
    outcome["item_001"] = step("verifier", [])
    assert runner.main(argv("verifier")) == 0
    run = read(results / "verifier" / "runs" / "item_001.json")
    assert run["result"]["model_calls"] == 0 and run["cumulative"]["model_calls"] == 20
    assert read(results / "verifier" / "item_001.json")["findings"] == []


def test_summary_reports_the_step_and_the_cumulative_row(env):
    argv, results, *_ = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    assert runner.main(argv("verifier")) == 0
    text = (results / "verifier" / "summary.md").read_text("utf-8")
    assert "# Results: `verifier`" in text and "limits.verifier_per_finding" in text
    assert "(`src/reviewer/prompts/verifier.md`" in text
    assert "## Cost and time (the judge step alone)" in text
    assert "## Cumulative cost: the saved `subagents` review plus this step" in text
    assert "Mean model calls per review: 24.0" in text
    assert "| verifier_rejected | 2 |" in text
    assert "| rejected | 2 |" in text and "| confirmed | 2 |" in text
    runs = runner.load_runs(results / "verifier", "verifier")
    assert all(isinstance(r, JudgeStepResult) for r in runs.values())
    assert set(runner.load_cumulative(results / "verifier")) == set(ITEMS)


# --- Row 4b: the orchestrator as judge -------------------------------------------------------


def test_judge_config_passes_the_saved_plan_and_summary(env):
    argv, results, seen, _ = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    assert runner.main(argv("judge")) == 0
    kind, item, titles, plan, summary = seen[0]
    assert (kind, item, titles) == ("judge", "item_001", ["keep", "drop"])
    assert plan[0]["specialist"] == "bug_reviewer" and summary == "summary of item_001"
    out = results / "judge"
    assert [f["title"] for f in read(out / "item_001.json")["findings"]] == ["keep"]
    assert read(out / "item_001.json")["config"] == "judge"
    text = (out / "summary.md").read_text("utf-8")
    assert "limits.judge" in text and "| judge_rejected | 2 |" in text
    assert "(`src/reviewer/prompts/orchestrator.md`" in text


# --- The scoring policy applies to the judge step --------------------------------------------


@pytest.mark.parametrize("config", ["verifier", "judge"])
@pytest.mark.parametrize("cls", ["RetriesExhausted", "AllExhausted"])
def test_infrastructure_errors_mean_the_item_must_be_rerun(env, capsys, config, cls):
    argv, results, seen, outcome = env
    seed(results, {"item_001": [KEEP, DROP], "item_002": [KEEP, DROP]})
    outcome["item_001"] = step(config, [KEEP, DROP], status="error", error_class=cls, error="x")
    assert runner.main(argv(config)) == 1  # non-zero: the item must be run again
    out = results / config
    assert not (out / "item_001.json").exists()  # not scored
    assert (out / "item_002.json").is_file()
    run = read(out / "runs" / "item_001.json")
    assert run["scoring"]["decision"] == "rerun_required" and cls in run["scoring"]["cause"]
    assert "scoring decision: rerun_required" in capsys.readouterr().out
    outcome.clear()  # a re-run picks the item up and scores it
    assert runner.main(argv(config)) == 0
    assert (out / "item_001.json").is_file()
    assert len(read(out / "runs" / "item_001.json")["run_log"]) == 2


def test_quota_pause_in_the_judge_step_stops_the_run(env):
    argv, results, seen, outcome = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    outcome["item_001"] = step("verifier", [KEEP, DROP], status="quota_paused",
                               error_class="QuotaPaused", error="x")  # fmt: skip
    assert runner.main(argv("verifier")) == 3
    assert not (results / "verifier" / "item_001.json").exists() and len(seen) == 1


def test_a_failed_check_that_is_not_infrastructure_is_scored(env):
    argv, results, seen, outcome = env
    seed(results, {i: [KEEP, DROP] for i in ITEMS})
    from reviewer.agents.base import ReviewResult

    bad = ReviewResult(status="limit_reached", agent="verifier", limit="max_model_calls",
                       prompt_version="p", limits=LIMITS, evaluation=True)  # fmt: skip
    outcome["item_001"] = step("verifier", [KEEP, DROP], status="partial", children=[bad])
    assert runner.main(argv("verifier")) == 0
    run = read(results / "verifier" / "runs" / "item_001.json")
    assert run["scoring"]["decision"] == "scored" and "partial" in run["scoring"]["cause"]


def test_judge_limits_and_prompts_are_wired_to_the_config():
    cfg = load_review_config()
    assert cfg.limits.for_agent(runner.JUDGE_LIMITS["verifier"]).max_model_calls == 6
    assert cfg.limits.for_agent(runner.JUDGE_LIMITS["judge"]).max_tool_calls == 12
    assert set(runner.CONFIGS) >= {"verifier", "judge"} and runner.JUDGED == ("verifier", "judge")
