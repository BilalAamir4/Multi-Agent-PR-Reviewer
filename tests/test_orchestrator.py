"""The stage 2B orchestrator (decision 0009): plan check, merge, statuses, accounting, nested and
parallel waits, the specialist tools, the CLI and the ``--config orchestrator`` runner.

Fake models and fake specialists only: no network, model or Docker call, no real sleeping.
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.exceptions import ModelRateLimitError

from eval import run as runner_mod
from reviewer import cli
from reviewer.agents import orchestrator as orch
from reviewer.agents.base import ReviewResult
from reviewer.agents.orchestrator import (
    OrchestratorResult,
    PlanEntry,
    SpecialistAbort,
    SpecialistCollector,
    check_plan,
    finding_id,
    format_specialist_output,
    merge_findings,
    orchestrate,
    orchestrator_spec,
)
from reviewer.agents.output import Dropped
from reviewer.config import AgentLimits, load_models_config, load_review_config
from reviewer.models import (
    QuotaRouter,
    QuotaTracker,
    WaitLog,
    attach_waits,
    record_wait,
    review_waits,
)
from reviewer.schemas import Finding
from tests.dev_pr import BODY, TITLE, make_dev_workspace
from tests.test_single_agent import USAGE, FakeTime, ScriptedModel, ai, answer, tool_call

EVAL_MODEL = load_models_config().evaluation.model
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BUG, SEC, TST = "bug_reviewer", "security_reviewer", "test_reviewer"
LIMITS = AgentLimits(max_model_calls=20, max_tool_calls=30)


# --- Helpers ---------------------------------------------------------------------------------


def fnd(agent="bug_reviewer", file="shop/catalog.py", start=36, end=36, category="off_by_one",
        confidence=0.8, title="A defect", model=EVAL_MODEL) -> Finding:  # fmt: skip
    return Finding(
        file=file, line_start=start, line_end=end, category=category, severity="medium",
        confidence=confidence, title=title, explanation="why", evidence="x", agent=agent,
        model=model,
    )  # fmt: skip


def child(name, status="complete", findings=(), **kw) -> ReviewResult:
    base = dict(
        model_calls=4, tool_calls=3, tool_calls_by_tool={"read_diff": 3}, input_tokens=1000,
        output_tokens=100, duration_s=10.0, wait_s=1.0, net_duration_s=9.0, model_id=EVAL_MODEL,
    )  # fmt: skip
    base.update(kw)
    return ReviewResult(
        status=status, agent=name, findings=list(findings), prompt_version="abc123def456",
        limits=LIMITS, evaluation=True, **base,
    )  # fmt: skip


def entry(name, called=True, reason="because"):
    return {"specialist": name, "called": called, "reason": reason}


def orch_answer(*entries, summary="Reviewed it."):
    return ai(tool_call("OrchestratorOutput", plan=list(entries), summary=summary))


def full_plan(called=(BUG, SEC, TST)):
    return [entry(n, n in called) for n in (BUG, SEC, TST)]


@pytest.fixture
def ctx(tmp_path):
    return make_dev_workspace(tmp_path)


@pytest.fixture
def fake_time():
    return FakeTime()


@pytest.fixture
def router(tmp_path, fake_time):
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota_state.json", providers)
    return QuotaRouter(
        config, evaluation=True, tracker=tracker, limiters={}, clock=fake_time.clock,
        sleep=fake_time.sleep,
    )  # fmt: skip


def script(router, messages):
    source = iter(messages)
    router.build = lambda model_id: ScriptedModel(messages=source)


class Fakes:
    """A fake specialist runner: records every real run (name, focus, thread)."""

    def __init__(self, results=None, hook=None):
        self.results = results or {}
        self.hook = hook
        self.runs: list[tuple[str, str | None, int]] = []
        self.lock = threading.Lock()

    def __call__(self, name, focus, config):
        with self.lock:
            self.runs.append((name, focus, threading.get_ident()))
        if self.hook:
            self.hook(name)
        return self.results.get(name) or child(name)


def run_orch(ctx, router, fake_time, fakes):
    return orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=fakes)


def call(name, focus="look at the diff"):
    return tool_call(f"call_{name}", focus=focus)


# --- Spec, tools and limits ------------------------------------------------------------------


def test_spec_limits_and_tools():
    spec = orchestrator_spec()
    assert spec.name == "orchestrator" and spec.role == "orchestrator"
    assert spec.tools == ("list_changed_files", "read_diff", "search_code")
    assert spec.limits == load_review_config().limits.orchestrator
    assert (spec.limits.max_model_calls, spec.limits.max_tool_calls) == (8, 12)
    assert "orchestrator" in load_models_config().chains.model_dump()


def test_prompt_has_untrusted_fragment_and_no_evaluation_words():
    text = orchestrator_spec().system_prompt()
    assert "untrusted data" in text and "<!-- include" not in text
    for word in ("dataset", "planted", "evaluation", "scoring", "ground truth"):
        assert word not in text.lower()


def test_agent_is_built_with_the_six_tools(ctx, router, fake_time, monkeypatch):
    seen = {}
    real = orch.build_agent

    def spy(spec, chat_model, ctx_, notes, mw=None, **kw):
        agent = real(spec, chat_model, ctx_, notes, mw, **kw)
        seen["tools"] = [t.name for t in kw["extra_tools"]]
        seen["schema"] = kw["response_schema"]
        return agent

    monkeypatch.setattr(orch, "build_agent", spy)
    script(router, [orch_answer(*full_plan(()))])
    run_orch(ctx, router, fake_time, Fakes())
    assert seen["tools"] == ["call_bug_reviewer", "call_security_reviewer", "call_test_reviewer"]
    assert seen["schema"] is orch.OrchestratorOutput


def test_tool_output_is_compact():
    r = child(BUG, findings=[fnd(title="Slice\nis short"), fnd(start=40, end=41, category="other")],
              notes=["sandbox unavailable: Docker Desktop is not running"],
              dropped=[Dropped(draft={}, reason="evidence", detail="d")])  # fmt: skip
    text = format_specialist_output(r)
    lines = text.splitlines()
    assert lines[0] == "specialist: bug_reviewer" and lines[1] == "status: complete"
    assert "note: sandbox unavailable: Docker Desktop is not running" in lines
    assert (
        "bug_reviewer#1 | shop/catalog.py:36-36 | off_by_one | medium | 0.80 | Slice is short"
        in lines
    )
    assert "bug_reviewer#2 | shop/catalog.py:40-41 | other | medium | 0.80 | A defect" in lines
    assert lines[-1] == "dropped: evidence 1"
    assert "why" not in text  # no explanations
    assert finding_id("bug_reviewer", 1) == "bug_reviewer#1"


# --- Once per review, and the cache (S6) -----------------------------------------------------


def test_second_call_is_refused_and_does_not_rerun(ctx, router, fake_time, monkeypatch):
    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    script(router, [
        ai(call(BUG)),
        ai(call(BUG)),
        orch_answer(*full_plan((BUG,))),
    ])  # fmt: skip
    seen: list[str] = []
    real = SpecialistCollector.call

    def spy(self, name, focus, config):
        out = real(self, name, focus, config)
        seen.append(out)
        return out

    monkeypatch.setattr(SpecialistCollector, "call", spy)
    res = run_orch(ctx, router, fake_time, fakes)
    assert [r[0] for r in fakes.runs] == [BUG]
    assert seen[1] == (
        "ERROR: bug_reviewer was already called in this review; its findings are listed above"
    )
    assert res.status == "complete" and len(res.findings) == 1


def test_collector_once_and_cache_across_attempts():
    fakes = Fakes()
    col = SpecialistCollector(fakes)
    first = col.call(BUG, "f", {})
    assert first.startswith("specialist: bug_reviewer")
    assert col.call(BUG, "f", {}).startswith("ERROR: bug_reviewer was already called")
    col.begin_attempt()  # an S4 re-run on another model
    again = col.call(BUG, "other focus", {})
    assert again == first  # served from the cache
    assert len(fakes.runs) == 1 and col.runs == 1


def test_fallback_rerun_reuses_finished_specialists(tmp_path, fake_time, ctx):
    """S4 + S6: the orchestrator's first model hits its daily quota after the specialists
    finished; the router re-runs the task on the next model and the specialists are not run
    again."""
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "q.json", providers)
    router = QuotaRouter(config, evaluation=False, tracker=tracker, limiters={},
                         clock=fake_time.clock, sleep=fake_time.sleep)  # fmt: skip
    chain = router.chain("orchestrator")
    built: list[str] = []

    class Dies(ScriptedModel):
        def _generate(self, messages, *a, **kw):
            if any(getattr(m, "type", "") == "tool" for m in messages):
                raise ModelRateLimitError("quota exceeded: requests per day")
            return super()._generate(messages, *a, **kw)

    def build(model_id):
        built.append(model_id)
        if len(built) == 1:
            return Dies(messages=iter([ai(call(BUG), call(SEC))]))
        return ScriptedModel(messages=iter([
            ai(call(BUG), call(SEC)),
            orch_answer(*full_plan((BUG, SEC))),
        ]))  # fmt: skip

    router.build = build
    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=fakes)
    assert built == chain[:2]
    assert res.status == "complete"
    assert sorted(r[0] for r in fakes.runs) == [BUG, SEC]  # each specialist ran exactly once
    assert len(res.attempts) == 2 and res.attempts[0].outcome.startswith("ModelRateLimitError")
    assert [c.agent for c in res.children] == [BUG, SEC]
    assert res.plan and not [n for n in res.notes if n.startswith("plan_mismatch")]
    assert res.model_id == chain[1]


def test_parallel_specialist_calls_run_in_threads(ctx, router, fake_time):
    barrier = threading.Barrier(3, timeout=5)  # all three must be running at the same time
    fakes = Fakes(hook=lambda name: barrier.wait())
    script(router, [ai(call(BUG), call(SEC), call(TST)), orch_answer(*full_plan())])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "complete"
    assert {r[0] for r in fakes.runs} == {BUG, SEC, TST}
    assert len({r[2] for r in fakes.runs}) == 3  # three worker threads
    assert res.own_tool_calls == 3


# --- Plan check ------------------------------------------------------------------------------


def plan(*items):
    return [PlanEntry(**e) for e in items]


def test_plan_ok():
    assert check_plan(plan(*full_plan((BUG,))), {BUG}) == []


def test_plan_missing_specialist():
    notes = check_plan(plan(entry(BUG), entry(SEC)), {BUG, SEC})
    assert notes == ["plan_mismatch: the plan has no entry for test_reviewer"]


def test_plan_duplicate_entry():
    notes = check_plan(plan(*full_plan(()), entry(SEC, False)), set())
    assert notes == ["plan_mismatch: the plan has 2 entries for security_reviewer"]


def test_plan_called_values_must_match_real_calls():
    notes = check_plan(plan(*full_plan((BUG, SEC))), {BUG, TST})
    assert notes == [
        "plan_mismatch: the plan says security_reviewer is called but it was not called",
        "plan_mismatch: the plan says test_reviewer is not called but it was called",
    ]


def test_mismatch_adds_note_and_keeps_real_calls(ctx, router, fake_time):
    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    script(router, [ai(call(BUG)), orch_answer(entry(BUG, False), entry(SEC, False))])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "complete"  # a mismatch is a note, not a failure
    assert [c.agent for c in res.children] == [BUG] and len(res.findings) == 1
    assert "plan_mismatch: the plan has no entry for test_reviewer" in res.notes
    assert "plan_mismatch: the plan says bug_reviewer is not called but it was called" in res.notes


# --- Merge -----------------------------------------------------------------------------------


def test_merge_keeps_agent_and_model_and_order():
    a = fnd(BUG, model="m1")
    b = fnd(SEC, file="shop/db.py", start=3, end=3, category="sql_injection", model="m2")
    kept, dropped = merge_findings({TST: child(TST), SEC: child(SEC, findings=[b]),
                                    BUG: child(BUG, findings=[a])})  # fmt: skip
    assert kept == [a, b] and dropped == []
    assert [(f.agent, f.model) for f in kept] == [(BUG, "m1"), (SEC, "m2")]


def test_merge_duplicate_keeps_higher_confidence():
    low = fnd(BUG, confidence=0.5, start=10, end=12)
    high = fnd(SEC, confidence=0.9, start=12, end=14)
    kept, dropped = merge_findings(
        {BUG: child(BUG, findings=[low]), SEC: child(SEC, findings=[high])}
    )
    assert kept == [high]
    (d,) = dropped
    assert d.reason == "duplicate"
    assert (
        d.draft["finding_id"] == "bug_reviewer#1"
        and d.draft["duplicate_of"] == "security_reviewer#1"
    )
    assert "bug_reviewer#1 duplicates security_reviewer#1" in d.detail


def test_merge_tie_keeps_the_earlier_specialist():
    b = fnd(BUG, confidence=0.7, category="other")
    s = fnd(SEC, confidence=0.7, category="other")
    t = fnd(TST, confidence=0.7, category="other")
    kept, dropped = merge_findings({BUG: child(BUG, findings=[b]), SEC: child(SEC, findings=[s]),
                                    TST: child(TST, findings=[t])})  # fmt: skip
    assert kept == [b]
    assert [d.draft["finding_id"] for d in dropped] == ["security_reviewer#1", "test_reviewer#1"]
    assert {d.draft["duplicate_of"] for d in dropped} == {"bug_reviewer#1"}


def test_merge_different_categories_at_the_same_lines_are_both_kept():
    a = fnd(BUG, category="off_by_one")
    b = fnd(SEC, category="other")
    kept, dropped = merge_findings({BUG: child(BUG, findings=[a]), SEC: child(SEC, findings=[b])})
    assert kept == [a, b] and dropped == []


def test_merge_non_overlapping_or_other_file_kept():
    a = fnd(BUG, start=1, end=2)
    b = fnd(SEC, start=3, end=4)
    c = fnd(TST, file="tests/test_x.py", start=1, end=2)
    kept, _ = merge_findings({BUG: child(BUG, findings=[a]), SEC: child(SEC, findings=[b]),
                              TST: child(TST, findings=[c])})  # fmt: skip
    assert kept == [a, b, c]


def test_orchestrator_result_dropped_includes_children_drops(ctx, router, fake_time):
    drop = Dropped(draft={"x": 1}, reason="evidence", detail="no match")
    fakes = Fakes({
        BUG: child(BUG, findings=[fnd(BUG, confidence=0.5)], dropped=[drop]),
        SEC: child(SEC, findings=[fnd(SEC, confidence=0.9)]),
    })  # fmt: skip
    script(router, [ai(call(BUG), call(SEC)), orch_answer(*full_plan((BUG, SEC)))])
    res = run_orch(ctx, router, fake_time, fakes)
    assert [f.agent for f in res.findings] == [SEC]
    assert [(d.reason, d.detail.split(" ")[0]) for d in res.dropped] == [
        ("evidence", "[bug_reviewer]"), ("duplicate", "bug_reviewer#1"),
    ]  # fmt: skip


# --- Status ----------------------------------------------------------------------------------


def test_status_complete(ctx, router, fake_time):
    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    script(router, [ai(call(BUG)), orch_answer(*full_plan((BUG,)))])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "complete" and res.notes == []
    assert res.summary == "Reviewed it." and res.agent == "orchestrator"


def test_no_specialist_called_is_complete(ctx, router, fake_time):
    script(router, [orch_answer(*full_plan(()))])
    res = run_orch(ctx, router, fake_time, Fakes())
    assert res.status == "complete" and res.children == [] and res.findings == []
    assert [e.called for e in res.plan] == [False, False, False]


@pytest.mark.parametrize(
    "child_status,extra",
    [
        ("partial", {"notes": ["sandbox unavailable: Docker Desktop is not running"]}),
        ("limit_reached", {"limit": "max_model_calls"}),
        ("error", {"error_class": "NoStructuredResponse", "error": "none"}),
    ],
)
def test_status_partial_keeps_the_others_findings(ctx, router, fake_time, child_status, extra):
    fakes = Fakes({
        BUG: child(BUG, findings=[fnd()]),
        TST: child(TST, child_status, **extra),
    })  # fmt: skip
    script(router, [ai(call(BUG), call(TST)), orch_answer(*full_plan((BUG, TST)))])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "partial"
    assert f"test_reviewer ended {child_status}; its findings are kept" in res.notes
    assert len(res.findings) == 1  # the bug reviewer's finding is kept


def test_partial_because_docker_is_down_with_real_run_tests(ctx, router, fake_time, monkeypatch):
    """The test specialist's own ``partial`` (D2b) makes the review partial: a real specialist
    run, with run_tests failing because Docker is not running."""
    from reviewer.agents import base
    from reviewer.sandbox import SandboxUnavailable
    from reviewer.tools import SandboxSpec
    from reviewer.tools import testing as rt

    spec = SandboxSpec("img:local", ("python", "-m", "pytest", "-q"), (".",), 60)
    ctx = type(ctx)(ctx.workspace, ctx.diff_text, ctx.files, spec)
    monkeypatch.setattr(rt, "check_image", lambda image: None)

    def no_docker(*a, **kw):
        raise SandboxUnavailable()

    monkeypatch.setattr(rt, "run_in_sandbox", no_docker)
    assert base  # the specialist runs through the shared machinery
    script(router, [
        ai(call(TST)),                                  # orchestrator
        ai(tool_call("run_tests", test_ids=["tests/test_catalog.py"])),  # test reviewer
        answer([]),                                     # test reviewer's answer
        orch_answer(*full_plan((TST,))),                # orchestrator
    ])  # fmt: skip
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock)
    assert res.status == "partial"
    (c,) = res.children
    assert c.agent == TST and c.status == "partial"
    assert any("Docker Desktop is not running" in n for n in c.notes)
    assert "test_reviewer ended partial; its findings are kept" in res.notes


def test_quota_paused_in_a_specialist_ends_the_review(ctx, router, fake_time):
    reset = datetime(2026, 11, 22, 8, tzinfo=UTC)
    paused = child(BUG, "quota_paused", error_class="QuotaPaused", error="paused",
                   model_id=EVAL_MODEL, reset_at=reset)  # fmt: skip
    fakes = Fakes({BUG: paused, SEC: child(SEC, findings=[fnd(SEC, category="other")])})
    script(router, [ai(call(SEC), call(BUG)), orch_answer(*full_plan((BUG, SEC)))])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "quota_paused"
    assert res.reset_at == reset and res.model_id == EVAL_MODEL
    assert res.error_class == "QuotaPaused"
    assert len(res.attempts) == 1  # the abort is not retried on another model
    assert [c.agent for c in res.children] == [BUG, SEC]  # finished work is kept


def test_all_exhausted_or_retries_exhausted_in_a_specialist_is_an_error(ctx, router, fake_time):
    for cls in ("AllExhausted", "RetriesExhausted"):
        r = child(SEC, "error", error_class=cls, error="nope")
        script(router, [ai(call(SEC)), orch_answer(*full_plan((SEC,)))])
        res = run_orch(ctx, router, fake_time, Fakes({SEC: r}))
        assert (res.status, res.error_class) == ("error", cls)


def test_other_specialist_errors_do_not_abort(ctx, router, fake_time):
    r = child(SEC, "error", error_class="PromptTooLarge", error="too big")
    script(router, [ai(call(SEC)), orch_answer(*full_plan((SEC,)))])
    res = run_orch(ctx, router, fake_time, Fakes({SEC: r}))
    assert res.status == "partial"


def test_quota_paused_tool_error_is_not_swallowed_by_toolnode(ctx):
    """The abort escapes a real ToolNode: it is not turned into an error message."""
    from langchain.agents import create_agent

    col = SpecialistCollector(Fakes({BUG: child(BUG, "quota_paused", error_class="QuotaPaused")}))
    tool = orch.make_specialist_tools(col)[0]
    model = ScriptedModel(messages=iter([ai(call(BUG)), ai(call(BUG))]))
    agent = create_agent(model, tools=[tool])
    with pytest.raises(SpecialistAbort):
        agent.invoke({"messages": [("user", "go")]})


def test_orchestrator_limit_keeps_the_specialists_findings(ctx, router, fake_time):
    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    script(router, [
        ai(call(BUG)),
        ai(tool_call("list_changed_files")),
        ai(tool_call("list_changed_files")),
        orch_answer(*full_plan((BUG,))),
    ])  # fmt: skip
    cfg = load_review_config()
    small = cfg.model_copy(update={"limits": cfg.limits.model_copy(
        update={"orchestrator": AgentLimits(max_model_calls=3, max_tool_calls=12)})})  # fmt: skip
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=fakes, config=small)
    assert res.status == "limit_reached" and res.limit == "max_model_calls"
    assert len(res.findings) == 1 and [c.agent for c in res.children] == [BUG]
    assert any(n.startswith("orchestrator reached max_model_calls") for n in res.notes)
    assert res.plan == []


def test_tool_limit_counts_the_specialist_calls(ctx, router, fake_time):
    fakes = Fakes()
    script(router, [ai(call(BUG), call(SEC), call(TST)), orch_answer(*full_plan())])
    cfg = load_review_config()
    small = cfg.model_copy(update={"limits": cfg.limits.model_copy(
        update={"orchestrator": AgentLimits(max_model_calls=8, max_tool_calls=2)})})  # fmt: skip
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=fakes, config=small)
    assert res.status == "limit_reached" and res.limit == "max_tool_calls"
    assert fakes.runs == []  # the batch of three would pass the limit, so it did not run


def test_no_structured_answer_is_an_error_but_keeps_findings(ctx, router, fake_time):
    from langchain_core.messages import AIMessage

    fakes = Fakes({BUG: child(BUG, findings=[fnd()])})
    script(router, [ai(call(BUG)), AIMessage(content="done", usage_metadata=USAGE)])
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.status == "error" and res.error_class == "NoStructuredResponse"
    assert len(res.findings) == 1


def test_fatal_orchestrator_model_error(ctx, router, fake_time):
    class Broken(ScriptedModel):
        def _generate(self, *a, **kw):
            raise ValueError("bad request")

    router.build = lambda model_id: Broken(messages=iter([]))
    res = run_orch(ctx, router, fake_time, Fakes())
    assert (res.status, res.error_class) == ("error", "ValueError")


def test_orchestrator_own_quota_pause(ctx, router, fake_time):
    reset = router.tracker.mark_exhausted(EVAL_MODEL)
    script(router, [orch_answer(*full_plan(()))])
    res = run_orch(ctx, router, fake_time, Fakes())
    assert res.status == "quota_paused" and res.reset_at == reset


# --- Totals ----------------------------------------------------------------------------------


def test_totals_add_up(ctx, router, fake_time):
    fakes = Fakes({
        BUG: child(BUG, model_calls=5, tool_calls=4,
                   tool_calls_by_tool={"read_diff": 3, "run_ruff": 1},
                   input_tokens=2000, output_tokens=200),
        SEC: child(SEC, model_calls=3, tool_calls=2, tool_calls_by_tool={"read_diff": 2},
                   input_tokens=500, output_tokens=50),
    })  # fmt: skip
    script(router, [
        ai(tool_call("list_changed_files")),
        ai(call(BUG), call(SEC)),
        orch_answer(*full_plan((BUG, SEC))),
    ])  # fmt: skip
    res = run_orch(ctx, router, fake_time, fakes)
    assert res.own_model_calls == 3 and res.own_tool_calls == 3  # list + two specialist tools
    assert res.own_input_tokens == 3 * USAGE["input_tokens"]
    assert res.own_output_tokens == 3 * USAGE["output_tokens"]
    assert res.model_calls == 3 + 5 + 3
    assert res.tool_calls == 3 + 4 + 2
    assert res.input_tokens == 3000 + 2000 + 500
    assert res.output_tokens == 150 + 200 + 50
    assert res.tool_calls_by_tool == {
        "call_bug_reviewer": 1, "call_security_reviewer": 1, "list_changed_files": 1,
        "read_diff": 5, "run_ruff": 1,
    }  # fmt: skip
    assert res.own_tool_calls_by_tool == {
        "call_bug_reviewer": 1, "call_security_reviewer": 1, "list_changed_files": 1,
    }  # fmt: skip


# --- Waits (D7) ------------------------------------------------------------------------------


def test_nested_waits_are_counted_once():
    with review_waits() as outer:
        record_wait("m", "rate_limiter", 2.0, end=102.0)
        with review_waits() as inner:
            record_wait("m", "retry_minute", 5.0, end=110.0)
        assert inner.total() == 5.0 and inner.own_total() == 5.0
        assert outer.total() == 7.0 and outer.own_total() == 2.0
        assert outer.by_reason() == {"rate_limiter": 2.0, "retry_minute": 5.0}
        assert [w.nested for w in outer.waits] == [False, True]
        record_wait("m", "retry_server", 1.0, end=111.0)  # back in the outer review
    assert outer.total() == 8.0 and inner.total() == 5.0


def test_wall_is_the_union_of_the_wait_intervals():
    log = WaitLog()
    log.record("m", "retry_minute", 5.0, 105.0)  # 100 to 105
    log.record("m", "retry_minute", 5.0, 107.0)  # 102 to 107, overlaps
    log.record("m", "rate_limiter", 1.0, 120.0)  # 119 to 120, apart
    assert log.total() == 11.0
    assert log.wall() == pytest.approx(7.0 + 1.0)
    assert WaitLog().wall() == 0.0


def test_parent_waits_reach_the_log_from_a_thread_without_the_context():
    log = WaitLog()
    done = threading.Event()

    def worker():  # a plain Thread starts with an empty context
        with attach_waits(log), review_waits() as inner:
            record_wait("m", "retry_minute", 4.0, end=14.0)
            assert inner.total() == 4.0
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert done.is_set() and log.total() == 4.0 and log.own_total() == 0.0


def test_parallel_specialists_with_overlapping_waits(ctx, router, fake_time):
    """Two fake specialists wait 6 s each over overlapping intervals: the nested total is the
    sum, the wall time the union, and net time subtracts the wall time (decision 0009)."""
    barrier = threading.Barrier(2, timeout=5)

    ends = {BUG: 1006.0, SEC: 1008.0}  # intervals 1000-1006 and 1002-1008: union 8, sum 12

    def hook(name):
        barrier.wait()
        with review_waits():
            record_wait("m", "retry_minute", 6.0, end=ends[name])

    fakes = Fakes({BUG: child(BUG), SEC: child(SEC)}, hook=hook)
    script(router, [ai(call(BUG), call(SEC)), orch_answer(*full_plan((BUG, SEC)))])

    def slow_orchestrator_time():  # the orchestrator's clock: starts at 1000, ends at 1030
        slow_orchestrator_time.t += 15.0
        return slow_orchestrator_time.t

    slow_orchestrator_time.t = 985.0
    res = orchestrate(ctx, TITLE, BODY, router, clock=slow_orchestrator_time, runner=fakes)
    assert res.duration_s == 15.0
    assert res.wait_s_sum == 12.0 and res.wait_s == 12.0
    assert res.wait_s_wall == 8.0 and res.wait_s_wall < res.wait_s_sum
    assert res.net_duration_s == 15.0 - 8.0
    assert res.waits_by_reason == {"retry_minute": 12.0}


def test_real_specialist_runs_nest_their_waits(ctx, router, fake_time):
    """Full stack: real ``run_specialist`` through the tool, one model for all. The specialist's
    waits are in its own result and in the orchestrator's total, once."""
    from reviewer.agents.specialists import run_specialist

    def runner(name, focus, config):
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=router,
            evaluation=True, clock=fake_time.clock, parent_config=config,
        )  # fmt: skip

    # floor(15 * 0.9) = 13 starts per 60 s for this model; 17 model calls in total make the later
    # ones wait on the rate limiter (fake clock, no real sleeping).
    msgs = [ai(call(BUG))]
    msgs += [ai(tool_call("list_changed_files")) for _ in range(14)]
    msgs += [answer([]), orch_answer(*full_plan((BUG,)))]
    script(router, msgs)
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=runner)
    assert res.status == "complete"
    (c,) = res.children
    assert c.model_calls == 15 and c.wait_s > 0
    assert res.own_model_calls == 2
    assert res.wait_s_sum >= c.wait_s and res.wait_s_sum == pytest.approx(
        sum(w for w in res.waits_by_reason.values())
    )
    # Counted once: the orchestrator's total is its own waits plus the child's, no more.
    own = res.wait_s_sum - c.wait_s
    assert own >= 0 and res.wait_s_sum == pytest.approx(c.wait_s + own)
    assert res.model_calls == 17  # counted once, not twice (callbacks are inherited)


# --- Real specialists under the orchestrator -------------------------------------------------


def test_child_events_are_not_counted_as_the_orchestrators_own(ctx, router, fake_time):
    from reviewer.agents.specialists import run_specialist

    seen_configs = []

    def runner(name, focus, config):
        seen_configs.append(config)
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=router,
            evaluation=True, clock=fake_time.clock, parent_config=config,
        )  # fmt: skip

    good = {
        "file": "shop/catalog.py", "line_start": 36, "line_end": 36, "category": "off_by_one",
        "severity": "medium", "confidence": 0.9, "title": "Each page is one product short",
        "explanation": "The slice end is start + size - 1.",
        "evidence": "return products[start : start + size - 1]",
    }  # fmt: skip
    script(router, [
        ai(tool_call("list_changed_files")),                       # orchestrator
        ai(call(BUG, "the catalog slicing")),                       # orchestrator
        ai(tool_call("read_diff", path="shop/catalog.py")),         # bug reviewer
        answer([good]),                                             # bug reviewer
        orch_answer(*full_plan((BUG,))),                            # orchestrator
    ])  # fmt: skip
    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=runner)
    assert res.status == "complete"
    assert (res.own_model_calls, res.own_tool_calls) == (3, 2)  # not 5 and 3
    (c,) = res.children
    assert (c.model_calls, c.tool_calls) == (2, 1)
    assert (res.model_calls, res.tool_calls) == (5, 3)
    assert res.own_input_tokens == 3 * USAGE["input_tokens"]
    assert c.input_tokens == 2 * USAGE["input_tokens"]
    assert res.input_tokens == 5 * USAGE["input_tokens"]
    assert [(f.agent, f.model) for f in res.findings] == [(BUG, EVAL_MODEL)]
    # The tool's config reached the specialist (it nests under the orchestrator's tool call).
    assert seen_configs and "callbacks" in seen_configs[0]


def test_quota_paused_inside_a_real_specialist_ends_the_review(ctx, router, fake_time):
    from reviewer.agents.specialists import run_specialist

    builds = []

    class Daily(ScriptedModel):
        def _generate(self, *a, **kw):
            raise ModelRateLimitError("quota exceeded: requests per day")

    def build(model_id):
        builds.append(model_id)
        if len(builds) == 1:
            return ScriptedModel(messages=iter([ai(call(BUG))]))
        return Daily(messages=iter([]))

    router.build = build

    def runner(name, focus, config):
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=router,
            evaluation=True, clock=fake_time.clock, parent_config=config,
        )  # fmt: skip

    res = orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, runner=runner)
    assert res.status == "quota_paused" and res.error_class == "QuotaPaused"
    assert res.reset_at is not None and res.model_id == EVAL_MODEL
    assert [c.status for c in res.children] == ["quota_paused"]


def test_security_goes_through_run_security_review(ctx, router, fake_time, monkeypatch):
    calls = []

    def fake_security(request, *, router=None, clock=None, parent_config=None):
        calls.append(request)
        return orch_response()

    def orch_response():
        from reviewer.agents.specialists import result_to_response

        return result_to_response(child(SEC, findings=[fnd(SEC, category="other")]))

    monkeypatch.setattr(orch, "run_security_review", fake_security)
    runner = orch.default_runner(ctx, "T", "B", router, item_id="x", clock=fake_time.clock,
                                 config=load_review_config())  # fmt: skip
    result = runner(SEC, "the sql", {})
    assert set(calls[0]) == {"workspace", "diff_text", "pr_title", "pr_body", "focus", "evaluation"}
    assert calls[0]["focus"] == "the sql" and calls[0]["evaluation"] is True
    assert result.agent == SEC and result.status == "complete" and len(result.findings) == 1
    assert (
        result.model_calls == 4 and result.limits == load_review_config().limits.security_reviewer
    )


# --- CLI -------------------------------------------------------------------------------------


def canned(status="complete", **kw) -> OrchestratorResult:
    base = dict(
        status=status, prompt_version="abc123def456", limits=LIMITS, evaluation=True,
        model_id=EVAL_MODEL, summary="A summary.", plan=plan(*full_plan((BUG,))),
        findings=[fnd()], children=[child(BUG, findings=[fnd()])], model_calls=9, tool_calls=7,
        own_model_calls=5, own_tool_calls=3, input_tokens=9000, output_tokens=400,
        own_input_tokens=5000, own_output_tokens=200, duration_s=30.0, wait_s=12.0, wait_s_sum=12.0,
        wait_s_wall=8.0, net_duration_s=22.0,
    )  # fmt: skip
    base.update(kw)
    return OrchestratorResult(**base)


@pytest.fixture
def cli_env(monkeypatch):
    out = {"result": canned()}
    monkeypatch.setattr(cli, "orchestrate", lambda *a, **kw: out["result"])
    monkeypatch.setattr(cli, "make_router", lambda evaluation: object())
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **kw: None)
    return out


def cli_args(tmp_path, *extra):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    diff = FIXTURES / "dev_pr" / "pr.diff"
    return [
        "review",
        "--agent",
        "orchestrator",
        "--workspace",
        str(ws),
        "--diff",
        str(diff),
        *extra,
    ]


def test_cli_prints_plan_findings_specialists_and_totals(cli_env, tmp_path, capsys):
    assert cli.main(cli_args(tmp_path, "--eval-mode")) == 0
    out = capsys.readouterr().out
    assert "bug_reviewer: called: because" in out
    assert "security_reviewer: skipped: because" in out
    assert "bug_reviewer  shop/catalog.py:36" in out  # the merged finding with its agent
    assert "status:  complete" in out
    assert "bug_reviewer: complete, 1 finding(s), 4 model / 3 tool calls" in out
    assert "calls:   9 model, 7 tool in total (orchestrator alone: 5 model, 3 tool)" in out
    assert "8.0s wall-clock" in out and "12.0s summed" in out
    assert "22.0s net" in out


@pytest.mark.parametrize(
    "status,code",
    [("partial", 5), ("limit_reached", 4), ("error", 1), ("complete", 0)],
)
def test_cli_exit_codes(cli_env, tmp_path, capsys, status, code):
    kw = {"limit": "max_model_calls"} if status == "limit_reached" else {}
    if status == "error":
        kw = {"error_class": "X", "error": "boom"}
    cli_env["result"] = canned(status, **kw)
    assert cli.main(cli_args(tmp_path)) == code


def test_cli_quota_paused_exit_code(cli_env, tmp_path, capsys):
    cli_env["result"] = canned("quota_paused", reset_at=datetime(2026, 11, 22, 8, tzinfo=UTC))
    assert cli.main(cli_args(tmp_path)) == 3
    assert "quota paused" in capsys.readouterr().out


def test_cli_writes_out_file(cli_env, tmp_path):
    out = tmp_path / "o" / "r.json"
    assert cli.main(cli_args(tmp_path, "--out", str(out))) == 0
    data = json.loads(out.read_bytes())
    assert data["agent"] == "orchestrator" and len(data["children"]) == 1
    assert OrchestratorResult.model_validate(data).wait_s_wall == 8.0


def test_cli_refuses_item_and_focus_for_the_orchestrator(cli_env, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["review", "--agent", "orchestrator", "--item", "item_001"])
    assert e.value.code == 2
    assert "not run on dataset items" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["review", "--agent", "orchestrator", "--workspace", "w", "--diff", "d",
                  "--focus", "x"])  # fmt: skip


# --- eval.run --config orchestrator ----------------------------------------------------------


def test_runner_writes_files_and_scorer_reads_them(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "dataset", dataset)
    results = tmp_path / "results"
    items = sorted(p.name for p in dataset.iterdir() if (p / "ground_truth.json").is_file())
    seen = []

    def fake_orchestrate(ctx, title, body, router, item_id=None):
        seen.append(item_id)
        return canned(findings=[fnd(file="pkg/util.py", start=2, end=2)], item_id=item_id)

    monkeypatch.setattr(runner_mod, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(runner_mod, "item_inputs", lambda item_id: (object(), "t", "b"))
    monkeypatch.setattr(runner_mod, "QuotaRouter", lambda evaluation: None)
    monkeypatch.setattr(runner_mod, "load_dotenv", lambda *a, **kw: None)
    code = runner_mod.main(["--config", "orchestrator", "--dataset", str(dataset),
                            "--results", str(results)])  # fmt: skip
    assert code == 0 and seen == items
    out = results / "orchestrator"
    scored = json.loads((out / f"{items[0]}.json").read_bytes())
    assert scored["config"] == "orchestrator" and scored["findings"][0]["agent"] == "bug_reviewer"
    run = json.loads((out / "runs" / f"{items[0]}.json").read_bytes())
    assert run["result"]["agent"] == "orchestrator" and len(run["result"]["children"]) == 1
    assert [e["specialist"] for e in run["result"]["plan"]] == [BUG, SEC, TST]
    log = run["run_log"][0]
    assert log["children"] == {BUG: "complete"} and log["wait_s_wall"] == 8.0
    assert (out / "score.json").is_file() and (out / "summary.md").is_file()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "# Results: `orchestrator`" in summary
    assert "`max_model_calls` 8" in summary  # limits.orchestrator from config, not limits.single
    loaded = runner_mod.load_runs(out, "orchestrator")
    assert all(isinstance(r, OrchestratorResult) for r in loaded.values())
