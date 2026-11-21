"""Scoring policy for non-complete reviews (stage 2C, decision 0010), the same for every
configuration. Fake results only: no model, network or Docker call."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import run as runner
from reviewer.agents.base import ReviewResult
from reviewer.agents.orchestrator import OrchestratorResult
from reviewer.config import AgentLimits
from tests.test_eval_run import finding

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIMITS = AgentLimits(max_model_calls=25, max_tool_calls=40)
BUG, SEC, TST = "bug_reviewer", "security_reviewer", "test_reviewer"
SANDBOX = "sandbox unavailable: Docker Desktop is not running"


def res(status, agent="single", **kw) -> ReviewResult:
    return ReviewResult(
        status=status, agent=agent, prompt_version="abc123def456", limits=LIMITS,
        evaluation=True, model_id="gemini-3.1-flash-lite", **kw,
    )  # fmt: skip


def orch(status, *children, **kw) -> OrchestratorResult:
    findings = [finding(agent=c.agent) for c in children if c.status == "complete"]
    return OrchestratorResult(
        status=status, prompt_version="abc123def456", limits=LIMITS, evaluation=True,
        model_id="gemini-3.1-flash-lite", children=list(children), findings=findings, **kw,
    )  # fmt: skip


def err(agent, cls, **kw):
    return res("error", agent, error_class=cls, error="x", **kw)


# (result, decision, cause fragment)
SINGLE = [
    (res("complete"), "scored", "complete"),
    (res("limit_reached", limit="max_tool_calls"), "scored", "limit_reached"),
    (err("single", "NoStructuredResponse"), "scored", "error: NoStructuredResponse"),
    (err("single", "ValidationError"), "scored", "error: ValidationError"),  # a FATAL model error
    (err("single", "PromptTooLarge"), "scored", "error: PromptTooLarge"),
    (res("quota_paused", error_class="QuotaPaused", error="x"), "rerun_required", "quota_paused"),
    (err("single", "RetriesExhausted"), "rerun_required", "RetriesExhausted"),
    (err("single", "AllExhausted"), "rerun_required", "AllExhausted"),
    (res("partial", notes=[SANDBOX]), "rerun_required", "sandbox unavailable"),
    (res("partial"), "rerun_required", "partial: no recorded cause"),
]
ORCH = [
    (orch("complete", res("complete", BUG)), "scored", "complete"),
    (orch("complete"), "scored", "complete"),  # no specialist called
    (
        orch("limit_reached", res("complete", BUG), limit="max_tool_calls"),
        "scored",
        "limit_reached",
    ),
    (
        orch("partial", res("complete", BUG), res("limit_reached", TST, limit="max_model_calls")),
        "scored",
        f"partial: {TST} limit_reached",
    ),
    (
        orch("partial", res("complete", BUG), err(SEC, "NoStructuredResponse")),
        "scored",
        f"partial: {SEC} error",
    ),
    (
        orch("partial", res("limit_reached", BUG), err(TST, "ValidationError")),  # FATAL
        "scored",
        f"{BUG} limit_reached, {TST} error",
    ),
    (orch("error", error_class="NoStructuredResponse"), "scored", "error: NoStructuredResponse"),
    (orch("error", error_class="ValidationError"), "scored", "error: ValidationError"),
    (res("quota_paused", "orchestrator"), "rerun_required", "quota_paused"),
    (
        orch("quota_paused", res("quota_paused", SEC), res("complete", BUG)),
        "rerun_required",
        "quota_paused",
    ),
    (
        orch("error", err(TST, "RetriesExhausted"), error_class="RetriesExhausted"),
        "rerun_required",
        "RetriesExhausted",
    ),
    (
        orch("partial", res("complete", BUG), err(SEC, "AllExhausted")),
        "rerun_required",
        f"AllExhausted ({SEC})",
    ),
    (
        orch("partial", res("complete", BUG), res("partial", TST, notes=[SANDBOX])),
        "rerun_required",
        f"sandbox unavailable ({TST})",
    ),
    # One infrastructure cause among scorable ones: the whole review is re-run.
    (
        orch("partial", res("limit_reached", BUG), res("partial", TST, notes=[SANDBOX])),
        "rerun_required",
        f"sandbox unavailable ({TST})",
    ),
    (
        orch("partial", res("complete", BUG), res("partial", TST)),
        "rerun_required",
        f"{TST} partial with no recorded cause",
    ),
    # Stage 3 (decision 0011): the A2A security service not running, or its call failing.
    (
        orch(
            "partial",
            res("complete", BUG),
            err(SEC, "RemoteUnavailable", notes=["security service unavailable: http://x:2025"]),
        ),
        "rerun_required",
        f"RemoteUnavailable ({SEC})",
    ),
    (
        orch("partial", res("complete", BUG), err(SEC, "RemoteError")),
        "rerun_required",
        f"RemoteError ({SEC})",
    ),
]


@pytest.mark.parametrize(("result", "decision", "cause"), SINGLE + ORCH)
def test_decide_scoring(result, decision, cause):
    got_decision, got_cause = runner.decide_scoring(result)
    assert got_decision == decision
    assert cause in got_cause


# --- End to end through the runner, for both configurations ----------------------------------


def run_with(monkeypatch, tmp_path, config, outcomes):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "dataset", dataset)
    results = tmp_path / "results"
    fake = lambda ctx, title, body, router, item_id=None, config=None: outcomes[item_id]  # noqa: E731
    monkeypatch.setattr(runner, "orchestrate" if config == "orchestrator" else "review", fake)
    monkeypatch.setattr(runner, "item_inputs", lambda item_id: (object(), "t", "b"))
    monkeypatch.setattr(runner, "QuotaRouter", lambda evaluation: None)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)
    argv = ["--config", config, "--dataset", str(dataset), "--results", str(results)]
    return argv, results / config


@pytest.mark.parametrize("config", ["single", "orchestrator"])
def test_scored_cause_writes_findings_and_records_the_decision(monkeypatch, tmp_path, config):
    if config == "single":
        first = res("limit_reached", limit="max_tool_calls")
        second = err("single", "NoStructuredResponse")
    else:
        first = orch(
            "partial", res("complete", BUG), res("limit_reached", TST, limit="max_tool_calls")
        )
        second = orch("error", res("complete", BUG), error_class="NoStructuredResponse")
    argv, out = run_with(monkeypatch, tmp_path, config, {"item_001": first, "item_002": second})
    assert runner.main(argv) == 0
    for item in ("item_001", "item_002"):
        assert (out / f"{item}.json").is_file()
        run = json.loads((out / "runs" / f"{item}.json").read_bytes())
        assert run["scoring"]["decision"] == "scored"
        assert run["run_log"][-1]["scoring"] == run["scoring"]
    # The findings that exist are written; a failed agent contributes nothing.
    n = len(json.loads((out / "item_001.json").read_bytes())["findings"])
    assert n == (0 if config == "single" else 1)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Scoring decisions (decision 0010)" in summary
    assert "Items scored: 2; items needing a re-run: 0" in summary
    assert "Recorded runs: 2, of which `rerun_required` (not scored, re-run): 0" in summary


@pytest.mark.parametrize("config", ["single", "orchestrator"])
@pytest.mark.parametrize("kind", ["retries", "all_exhausted", "sandbox"])
def test_rerun_required_is_recorded_not_scored_and_exits_non_zero(
    monkeypatch, tmp_path, config, kind, capsys
):
    def bad(cfg):
        if kind == "sandbox":
            if cfg == "single":
                return res("partial", notes=[SANDBOX])
            return orch("partial", res("complete", BUG), res("partial", TST, notes=[SANDBOX]))
        cls = "RetriesExhausted" if kind == "retries" else "AllExhausted"
        if cfg == "single":
            return err("single", cls)
        return orch("error", err(SEC, cls), error_class=cls)

    good = res("complete") if config == "single" else orch("complete", res("complete", BUG))
    argv, out = run_with(monkeypatch, tmp_path, config, {"item_001": good, "item_002": bad(config)})
    assert runner.main(argv) == 1
    assert (out / "item_001.json").is_file() and not (out / "item_002.json").exists()
    run = json.loads((out / "runs" / "item_002.json").read_bytes())
    assert run["scoring"]["decision"] == "rerun_required"
    assert run["scoring"]["cause"]
    assert not (out / "summary.md").exists()
    assert "scoring decision: rerun_required" in capsys.readouterr().out
    # The retry replaces it and the summary counts both runs.
    argv_fix = argv
    runner_outcomes = {"item_002": good}
    monkeypatch.setattr(
        runner, "orchestrate" if config == "orchestrator" else "review",
        lambda ctx, t, b, router, item_id=None, config=None: runner_outcomes[item_id],
    )  # fmt: skip
    assert runner.main([*argv_fix, "--reason", f"re-run after {kind}"]) == 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Items scored: 2; items needing a re-run: 0" in summary
    assert "Recorded runs: 3, of which `rerun_required` (not scored, re-run): 1" in summary


@pytest.mark.parametrize("config", ["single", "orchestrator"])
def test_quota_paused_stops_with_exit_3_and_records_the_cause(monkeypatch, tmp_path, config):
    paused = res("quota_paused", error_class="QuotaPaused", error="x")
    if config == "orchestrator":
        paused = orch("quota_paused", res("quota_paused", SEC), error_class="QuotaPaused")
    argv, out = run_with(monkeypatch, tmp_path, config, {"item_001": paused})
    assert runner.main(argv) == 3
    run = json.loads((out / "runs" / "item_001.json").read_bytes())
    assert run["scoring"]["decision"] == "rerun_required"
    assert "quota_paused" in run["scoring"]["cause"]
    assert not (out / "item_001.json").exists()
