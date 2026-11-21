"""Per-call retries (stage 2C, decision 0010): ``MINUTE`` and ``SERVER`` errors are retried on the
same model call inside the agent loop; only a fallback re-runs the whole task (S4).

Fake chat models raise the real provider exception classes (built in ``tests/test_models.py``);
the clock and sleep are fakes. No network, no model call, no real sleeping.
"""

from __future__ import annotations

from typing import Any

import pytest

from reviewer.agents.orchestrator import orchestrate
from reviewer.agents.single import review
from reviewer.agents.specialists import run_specialist
from reviewer.config import AgentLimits, load_models_config
from reviewer.models import (
    EVALUATION_POLICY,
    NORMAL_POLICY,
    CallRetrier,
    QuotaRouter,
    QuotaTracker,
    RateLimiter,
    RetryPolicy,
    WaitLog,
    retries_made,
)
from tests.dev_pr import BODY, TITLE, make_dev_workspace
from tests.test_models import (
    gemini_daily,
    gemini_minute,
    gemini_overloaded,
    gemini_too_big,
    groq_server,
    schema_error,
)
from tests.test_orchestrator import BUG, SEC, call, full_plan, orch_answer
from tests.test_single_agent import FakeTime, ScriptedModel, ai, answer, tool_call

CONFIG = load_models_config()
EVAL_MODEL = CONFIG.evaluation.model
LIMITS = AgentLimits(max_model_calls=25, max_tool_calls=40)

# A run of three model calls: two tool turns, then the answer.
THREE = [
    ai(tool_call("list_changed_files")),
    ai(tool_call("read_diff", path="shop/catalog.py")),
    answer([]),
]


class State:
    """Shared by every fake model of one test: counts ``_generate`` calls (retries included) and
    raises ``errors[n]()`` on the n-th (1-based)."""

    def __init__(self, errors: dict[int, Any] | None = None) -> None:
        self.errors = dict(errors or {})
        self.n = 0


class Flaky(ScriptedModel):
    state: Any = None

    def _generate(self, messages, *a, **kw):
        self.state.n += 1
        make = self.state.errors.get(self.state.n)
        if make is not None:
            raise make()
        return super()._generate(messages, *a, **kw)


@pytest.fixture
def ctx(tmp_path):
    return make_dev_workspace(tmp_path)


@pytest.fixture
def ft():
    return FakeTime()


def make_router(tmp_path, ft, *, evaluation: bool, **kw) -> QuotaRouter:
    providers = {m: s.provider for m, s in CONFIG.models.items()}
    return QuotaRouter(
        CONFIG,
        evaluation=evaluation,
        tracker=QuotaTracker(tmp_path / "quota_state.json", providers),
        limiters={},
        clock=ft.clock,
        sleep=ft.sleep,
        **kw,
    )


@pytest.fixture
def ev(tmp_path, ft):
    return make_router(tmp_path, ft, evaluation=True)


@pytest.fixture
def normal(tmp_path, ft):
    return make_router(tmp_path, ft, evaluation=False)


def script(router, messages, errors=None) -> State:
    """Every model the router builds draws from one message stream."""
    state = State(errors)
    source = iter(messages)
    router.build = lambda model_id: Flaky(messages=source, state=state)
    return state


def run(ctx, router, ft):
    return review(ctx, TITLE, BODY, router, limits=LIMITS, clock=ft.clock)


# --- One 503 in the middle of a run ---------------------------------------------------------


def test_a_503_in_the_middle_is_retried_on_the_same_call(ctx, ev, ft):
    state = script(ev, THREE, {2: gemini_overloaded})
    res = run(ctx, ev, ft)
    assert res.status == "complete"
    # Three real calls plus one failed attempt: the first call was NOT repeated.
    assert state.n == 4
    assert res.model_calls == 3 and len(res.attempts) == 1
    assert ft.now == pytest.approx(1000.0 + 15.0)  # one 15 s wait (evaluation backoff)
    assert res.waits_by_reason == {"retry_server": 15.0}
    assert res.retries_by_reason == {"retry_server": 1}


def test_the_retry_resends_the_same_message_history(ctx, ev, ft):
    seen: list[int] = []

    class Spy(Flaky):
        def _generate(self, messages, *a, **kw):
            seen.append(len(messages))
            return super()._generate(messages, *a, **kw)

    state = State({2: gemini_overloaded})
    source = iter(THREE)
    ev.build = lambda model_id: Spy(messages=source, state=state)
    assert run(ctx, ev, ft).status == "complete"
    assert seen == [
        2,
        4,
        4,
        6,
    ]  # the failed call and its retry saw the same 4 messages (system incl.)


def test_a_minute_limit_in_the_middle_is_retried_and_logged_as_retry_minute(ctx, ev, ft):
    script(ev, THREE, {3: lambda: gemini_minute("41s")})
    res = run(ctx, ev, ft)
    assert res.status == "complete"
    assert res.waits_by_reason == {"retry_minute": 41.0}  # retry_after, under the 60 s cap
    assert res.retries_by_reason == {"retry_minute": 1}


# --- Limits and backoff -----------------------------------------------------------------------


def test_policies():
    assert NORMAL_POLICY == RetryPolicy(2, 10.0, 30.0)
    assert EVALUATION_POLICY == RetryPolicy(6, 15.0, 60.0)


def test_evaluation_backoff_and_six_retries(ctx, ev, ft):
    errors = {n: gemini_overloaded for n in range(1, 7)}  # the first call fails 6 times
    state = script(ev, THREE, errors)
    res = run(ctx, ev, ft)
    assert res.status == "complete" and state.n == 9
    assert res.waits_by_reason == {"retry_server": 15 + 30 + 60 + 60 + 60 + 60}
    assert res.retries_by_reason == {"retry_server": 6}


def test_evaluation_seventh_failure_raises_retries_exhausted_without_fallback(ctx, ev, ft):
    state = script(ev, THREE, {n: gemini_overloaded for n in range(1, 8)})
    res = run(ctx, ev, ft)
    assert (res.status, res.error_class) == ("error", "RetriesExhausted")
    assert state.n == 7 and len(res.attempts) == 1  # no re-run, no other model
    assert "after 6 retries" in (res.error or "")
    assert res.retries_by_reason == {"retry_server": 6}


def test_evaluation_retry_after_is_capped_at_60(ctx, ev, ft):
    script(ev, THREE, {1: lambda: gemini_minute("90s")})
    assert run(ctx, ev, ft).waits_by_reason == {"retry_minute": 60.0}


def test_normal_backoff_and_two_retries(ctx, normal, ft):
    state = script(normal, THREE, {1: gemini_overloaded, 2: gemini_overloaded})
    res = run(ctx, normal, ft)
    assert res.status == "complete" and state.n == 5
    assert res.waits_by_reason == {"retry_server": 10 + 20}
    assert len(res.attempts) == 1


def test_normal_retry_after_is_capped_at_30(ctx, normal, ft):
    script(normal, THREE, {1: lambda: gemini_minute("41s")})
    assert run(ctx, normal, ft).waits_by_reason == {"retry_minute": 30.0}


def test_normal_mode_falls_over_with_a_whole_task_rerun_after_two_retries(ctx, normal, ft):
    chain = normal.chain("single")
    streams = {chain[0]: iter(THREE), chain[1]: iter(THREE)}
    state = State({n: gemini_overloaded for n in range(2, 5)})  # 2nd call of model 1 fails x3
    builds: list[str] = []

    def build(model_id):
        builds.append(model_id)
        return Flaky(messages=streams[model_id], state=state)

    normal.build = build
    res = run(ctx, normal, ft)
    assert builds == chain[:2]
    assert res.status == "complete" and res.model_id == chain[1]
    assert [a.model for a in res.attempts] == chain[:2]
    # Model 1: call 1 ok, call 2 fails 3 times (2 retries); model 2 re-runs all 3 calls.
    assert state.n == 1 + 3 + 3
    assert res.waits_by_reason == {"retry_server": 10 + 20}
    assert res.attempts[0].outcome.startswith("ServerError") or "503" in res.attempts[0].outcome


# --- Errors that are not retried per call -----------------------------------------------------


def test_daily_is_not_retried_per_call(ctx, ev, ft):
    state = script(ev, THREE, {2: gemini_daily})
    res = run(ctx, ev, ft)
    assert res.status == "quota_paused" and state.n == 2
    assert ft.now == 1000.0 and res.retries_by_reason == {}


def test_too_big_is_not_retried_per_call(ctx, ev, ft):
    state = script(ev, THREE, {2: gemini_too_big})
    res = run(ctx, ev, ft)
    assert (res.status, res.error_class) == ("error", "PromptTooLarge") and state.n == 2
    assert ft.now == 1000.0 and res.retries_by_reason == {}


def test_fatal_is_not_retried_per_call(ctx, ev, ft):
    state = script(ev, THREE, {2: schema_error})
    res = run(ctx, ev, ft)
    assert res.status == "error" and state.n == 2
    assert ft.now == 1000.0 and res.retries_by_reason == {}


def test_daily_in_normal_mode_falls_over_at_once(ctx, normal, ft):
    chain = normal.chain("single")
    streams = {chain[0]: iter(THREE), chain[1]: iter(THREE)}
    state = State({2: gemini_daily})
    normal.build = lambda model_id: Flaky(messages=streams[model_id], state=state)
    res = run(ctx, normal, ft)
    assert res.status == "complete" and res.model_id == chain[1] and state.n == 5
    assert ft.now == 1000.0  # no wait


# --- The retrier on its own -------------------------------------------------------------------


def test_retrier_raises_the_original_error_with_the_retry_count():
    clock = FakeTime()
    log = WaitLog()
    r = CallRetrier(NORMAL_POLICY, "m", log, clock=clock.clock, sleep=clock.sleep)
    err = groq_server(503)

    def boom():
        raise err

    with pytest.raises(type(err)) as e:
        r.call(boom)
    assert e.value is err and retries_made(err) == 2
    assert r.counts == {"retry_server": 2}
    assert [(w.reason, w.seconds) for w in log.waits] == [
        ("retry_server", 10.0),
        ("retry_server", 20.0),
    ]
    assert log.waits[-1].end == clock.now


def test_the_rate_limiter_runs_before_each_retry(ctx, tmp_path, ft):
    """One start per 60 s for this model: after the 15 s retry wait the retry still has to wait
    45 s on the limiter, and both are logged."""
    router = make_router(tmp_path, ft, evaluation=True)
    router._limiters[EVAL_MODEL] = RateLimiter(2, clock=ft.clock, sleep=ft.sleep)  # 1 / 60 s
    script(router, THREE, {1: gemini_overloaded})
    res = run(ctx, router, ft)
    # The router's own start used the single slot; the retry waits for it: 15 s backoff + 45 s.
    assert res.status == "complete"
    assert res.waits_by_reason["retry_server"] == 15.0
    assert res.waits_by_reason["rate_limiter"] >= 45.0


# --- The orchestrator ---------------------------------------------------------------------------


def test_a_503_inside_one_specialist_does_not_rerun_the_others(ctx, ev, ft):
    def runner(name, focus, config):
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=ev,
            evaluation=True, clock=ft.clock, parent_config=config,
        )  # fmt: skip

    msgs = [
        ai(call(BUG)),  # 1 orchestrator
        answer([]),  # 2 bug reviewer
        ai(call(SEC)),  # 3 orchestrator
        answer([]),  # 4 security reviewer: its call fails once (503), then succeeds
        orch_answer(*full_plan((BUG, SEC))),  # 5 orchestrator
    ]
    state = script(ev, msgs, {4: gemini_overloaded})
    res = orchestrate(ctx, TITLE, BODY, ev, clock=ft.clock, runner=runner)
    assert res.status == "complete", res.error
    assert state.n == 6  # five real calls plus the one failed attempt
    by = {c.agent: c for c in res.children}
    assert by[BUG].model_calls == 1 and by[SEC].model_calls == 1  # bug reviewer ran once
    assert len(res.attempts) == 1  # the orchestrator task was not re-run
    assert by[BUG].retries_by_reason == {} and by[SEC].retries_by_reason == {"retry_server": 1}
    assert res.retries_by_reason == {"retry_server": 1}
    assert res.wait_s_sum == 15.0 and res.waits_by_reason == {"retry_server": 15.0}


def test_a_503_in_the_orchestrator_itself_does_not_rerun_the_specialists(ctx, ev, ft):
    runs: list[str] = []

    def runner(name, focus, config):
        runs.append(name)
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=ev,
            evaluation=True, clock=ft.clock, parent_config=config,
        )  # fmt: skip

    msgs = [ai(call(BUG)), answer([]), orch_answer(*full_plan((BUG,)))]
    state = script(ev, msgs, {3: gemini_overloaded})  # the orchestrator's last call
    res = orchestrate(ctx, TITLE, BODY, ev, clock=ft.clock, runner=runner)
    assert res.status == "complete" and runs == [BUG] and state.n == 4
    assert res.retries_by_reason == {"retry_server": 1}
