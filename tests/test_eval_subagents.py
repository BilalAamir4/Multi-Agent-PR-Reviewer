"""``eval.run`` for stage 4A (decision 0012): ``--config orchestrator`` forces spawning off,
``--config subagents`` turns it on and applies the D6 reuse rule; the scoring policy for
sub-agent outcomes. Fake results only: no model, network, Docker or git call."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import run as runner
from reviewer.agents.base import ReviewResult, SpawnInfo
from reviewer.agents.orchestrator import OrchestratorResult
from reviewer.config import load_review_config
from reviewer.tools import ReviewContext
from tests.test_eval_run import finding
from tests.test_scoring_policy import BUG, LIMITS, SANDBOX, SEC, TST, err, orch, res
from tests.test_subagents_batching import make_ctx, py

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SUB1, SUB2 = f"{BUG}/sub1", f"{BUG}/sub2"


def spawner(agent, status, *subs, **kw) -> ReviewResult:
    """A specialist that spawned ``subs``."""
    return res(
        status, agent, children=list(subs),
        spawn=SpawnInfo(enabled=True, spawned=True, triggers=["files"], py_files=5), **kw,
    )  # fmt: skip


# --- The scoring policy with sub-agents (extends decision 0010) ----------------------------

CASES = [
    (  # a sub-agent that hit its call limit: scored, the specialist is partial
        orch(
            "partial",
            spawner(
                BUG,
                "partial",
                res("complete", SUB1),
                res("limit_reached", SUB2, limit="max_model_calls"),
            ),
        ),
        "scored",
        f"partial: {BUG} partial",
    ),
    (  # a sub-agent with a non-infrastructure error: scored
        orch(
            "partial",
            spawner(BUG, "partial", res("complete", SUB1), err(SUB2, "NoStructuredResponse")),
        ),
        "scored",
        f"partial: {BUG} partial",
    ),
    (  # a sub-agent partial with no recorded cause: re-run
        orch("partial", spawner(BUG, "partial", res("complete", SUB1), res("partial", SUB2))),
        "rerun_required",
        f"partial: {SUB2} partial with no recorded cause",
    ),
    (  # a specialist partial with nothing below it to explain it: re-run, as before
        orch("partial", res("partial", BUG)),
        "rerun_required",
        f"partial: {BUG} partial with no recorded cause",
    ),
    (  # the cross-file pass hit its limit: the specialist's own status, scored
        orch(
            "partial", spawner(BUG, "limit_reached", res("complete", SUB1), limit="max_model_calls")
        ),
        "scored",
        f"partial: {BUG} limit_reached",
    ),
    (  # a sub-agent without Docker: infrastructure, re-run
        orch("partial", spawner(TST, "partial", res("partial", f"{TST}/sub1", notes=[SANDBOX]))),
        "rerun_required",
        f"sandbox unavailable ({TST}/sub1) ({TST})",
    ),
    (  # a sub-agent whose retries ran out: the whole review is re-run
        orch(
            "error",
            spawner(BUG, "error", err(SUB1, "RetriesExhausted"), error_class="RetriesExhausted"),
            error_class="RetriesExhausted",
        ),
        "rerun_required",
        "RetriesExhausted",
    ),
    (
        orch("partial", spawner(BUG, "partial", res("complete", SUB1), err(SUB2, "AllExhausted"))),
        "rerun_required",
        f"AllExhausted ({SUB2}) ({BUG})",
    ),
    (  # quota paused in a sub-agent
        orch(
            "quota_paused",
            spawner(BUG, "quota_paused", res("quota_paused", SUB2)),
            error_class="QuotaPaused",
        ),
        "rerun_required",
        "quota_paused",
    ),
    (  # the security service reported an error in one of its sub-agents (children came back)
        orch(
            "partial",
            spawner(
                SEC,
                "partial",
                res("complete", f"{SEC}/sub1"),
                res("limit_reached", f"{SEC}/sub2", limit="max_tool_calls"),
            ),
        ),
        "scored",
        f"partial: {SEC} partial",
    ),
    (  # a spawning specialist that is complete below a complete orchestrator
        orch("complete", spawner(BUG, "complete", res("complete", SUB1), res("complete", SUB2))),
        "scored",
        "complete",
    ),
]


@pytest.mark.parametrize(("result", "decision", "cause"), CASES)
def test_decide_scoring_with_subagents(result, decision, cause):
    got_decision, got_cause = runner.decide_scoring(result)
    assert got_decision == decision
    assert cause in got_cause


def test_a_specialist_that_spawned_can_be_the_top_level_result():
    """``reviewer review --agent bug_reviewer`` is scored the same way (children are looked at)."""
    ok = spawner(
        BUG, "partial", res("complete", SUB1), res("limit_reached", SUB2, limit="max_model_calls")
    )
    assert runner.decide_scoring(ok) == ("scored", f"partial: {SUB2} limit_reached")
    bad = spawner(BUG, "partial", res("complete", SUB1), res("partial", SUB2))
    assert runner.decide_scoring(bad)[0] == "rerun_required"


# --- Row 2 forces spawning off, row 3 turns it on --------------------------------------------


def orchestrator_result(item_id, *children) -> OrchestratorResult:
    r = orch("complete", *children, summary="s", item_id=item_id, model_calls=9, tool_calls=7)
    return r.model_copy(update={"findings": [finding(agent=BUG)]})


@pytest.fixture
def env(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "dataset", dataset)
    results = tmp_path / "results"
    seen: list[tuple[str, bool]] = []
    outcomes: dict[str, OrchestratorResult] = {}
    contexts: dict[str, ReviewContext] = {}

    def fake_orchestrate(ctx, title, body, router, item_id=None, config=None):
        seen.append((item_id, config.spawn.enabled))
        return outcomes[item_id]

    monkeypatch.setattr(runner, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(runner, "item_inputs", lambda item_id: (contexts[item_id], "t", "b"))
    monkeypatch.setattr(runner, "QuotaRouter", lambda evaluation: None)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)

    def argv(config, *more):
        return ["--config", config, "--dataset", str(dataset), "--results", str(results), *more]

    return argv, results, seen, outcomes, contexts, tmp_path


def small_ctx(tmp_path, name):
    return make_ctx(tmp_path / name, py("a.py", "b.py"))


def large_ctx(tmp_path, name):
    return make_ctx(tmp_path / name, py("a.py", "b.py", "c.py", "d.py"))


def test_the_orchestrator_configuration_forces_spawning_off_and_records_it(env):
    argv, results, seen, outcomes, contexts, tmp = env
    for item in ("item_001", "item_002"):
        contexts[item] = large_ctx(tmp, item)  # it would spawn
        outcomes[item] = orchestrator_result(item, res("complete", BUG))
    assert load_review_config().spawn.enabled is True  # the file says on
    assert runner.main(argv("orchestrator")) == 0
    assert seen == [("item_001", False), ("item_002", False)]
    run = json.loads((results / "orchestrator" / "runs" / "item_001.json").read_bytes())
    assert run["spawn_enabled"] is False
    log = run["run_log"][-1]
    assert log["spawn_enabled"] is False and log["spawned"] == {
        BUG: {"spawned": False, "subagents": 0, "model_calls": 0}
    }
    assert "Spawning: off (forced" in (results / "orchestrator" / "summary.md").read_text("utf-8")


def seed_orchestrator(argv, outcomes, contexts, tmp, large: set[str]):
    for item in ("item_001", "item_002"):
        contexts[item] = (large_ctx if item in large else small_ctx)(tmp, item)
        outcomes[item] = orchestrator_result(item, res("complete", BUG), res("complete", TST))
    assert runner.main(argv("orchestrator")) == 0


def test_subagents_reuses_the_items_where_nothing_would_spawn(env, capsys):
    argv, results, seen, outcomes, contexts, tmp = env
    seed_orchestrator(argv, outcomes, contexts, tmp, large={"item_002"})
    seen.clear()
    outcomes["item_002"] = orchestrator_result(
        "item_002",
        spawner(BUG, "complete", res("complete", SUB1), res("complete", SUB2)),
        res("complete", TST),
    )
    assert runner.main(argv("subagents")) == 0
    assert seen == [("item_002", True)]  # only the large item was run, with spawning on
    out = results / "subagents"
    reused = json.loads((out / "runs" / "item_001.json").read_bytes())
    assert reused["reused_from"] == "orchestrator" and reused["spawn_enabled"] is False
    assert reused["run_log"][-1]["reused_from"] == "orchestrator"
    assert (
        reused["result"]
        == json.loads((results / "orchestrator" / "runs" / "item_001.json").read_bytes())["result"]
    )
    scored = json.loads((out / "item_001.json").read_bytes())
    original = json.loads((results / "orchestrator" / "item_001.json").read_bytes())
    assert scored == {**original, "config": "subagents"}  # the scorer needs the folder's name
    ran = json.loads((out / "runs" / "item_002.json").read_bytes())
    assert "reused_from" not in ran and ran["spawn_enabled"] is True
    assert ran["run_log"][-1]["spawned"][BUG] == {"spawned": True, "subagents": 2, "model_calls": 0}
    assert "item_001: reused from orchestrator" in capsys.readouterr().out
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "# Results: `subagents`" in summary and "## Reuse (decision 0012, D6)" in summary
    assert "- Reused from `orchestrator` (1): item_001" in summary
    assert "- Run with spawning on (1): item_002" in summary
    assert "(`src/reviewer/prompts/orchestrator.md`" in summary


def test_only_called_specialists_decide_the_reuse(env):
    """A large diff is reused when the saved run called no specialist at all."""
    argv, results, seen, outcomes, contexts, tmp = env
    for item in ("item_001", "item_002"):
        contexts[item] = large_ctx(tmp, item)
        outcomes[item] = orchestrator_result(item)  # nothing called
    assert runner.main(argv("orchestrator")) == 0
    seen.clear()
    assert runner.main(argv("subagents")) == 0
    assert seen == []  # both reused: no called specialist would spawn
    reuse = runner.load_reuse(results / "subagents")
    assert reuse == {"item_001": "orchestrator", "item_002": "orchestrator"}


def test_missing_orchestrator_results_stop_the_run_with_a_clear_error(env, capsys):
    argv, results, seen, outcomes, contexts, tmp = env
    assert runner.main(argv("subagents")) == 2
    err_text = capsys.readouterr().err
    assert "run --config orchestrator first" in err_text and "item_001" in err_text
    assert seen == [] and not (results / "subagents").exists()
    # Only the items it would run need the results: a complete subagents folder is not blocked.
    seed_orchestrator(argv, outcomes, contexts, tmp, large=set())
    assert runner.main(argv("subagents")) == 0
    shutil.rmtree(results / "orchestrator")
    assert runner.main(argv("subagents")) == 0  # everything already scored: nothing to read


def test_an_orchestrator_item_that_was_not_scored_counts_as_missing(env, capsys):
    argv, results, seen, outcomes, contexts, tmp = env
    seed_orchestrator(argv, outcomes, contexts, tmp, large=set())
    (results / "orchestrator" / "item_002.json").unlink()  # for example rerun_required
    assert runner.main(argv("subagents")) == 2
    assert "item_002" in capsys.readouterr().err


def test_force_runs_a_reused_item_again_from_the_saved_results(env):
    argv, results, seen, outcomes, contexts, tmp = env
    seed_orchestrator(argv, outcomes, contexts, tmp, large=set())
    assert runner.main(argv("subagents")) == 0
    assert (
        runner.main(argv("subagents", "--items", "item_001", "--force", "--reason", "again")) == 0
    )
    log = json.loads((results / "subagents" / "runs" / "item_001.json").read_bytes())["run_log"]
    assert [e.get("reused_from") for e in log][-2:] == ["orchestrator", "orchestrator"]
    assert log[-1]["reason"] == "again"


def test_the_subagents_summary_loads_the_orchestrator_shaped_runs(env):
    argv, results, seen, outcomes, contexts, tmp = env
    seed_orchestrator(argv, outcomes, contexts, tmp, large=set())
    assert runner.main(argv("subagents")) == 0
    runs = runner.load_runs(results / "subagents", "subagents")
    assert all(isinstance(r, OrchestratorResult) for r in runs.values())
    assert LIMITS.max_model_calls == 25  # (the fixture limits are not what the summary prints)
    assert (
        "`orchestrator`"
        not in (results / "subagents" / "summary.md").read_text("utf-8").split("Reuse")[0]
    )
