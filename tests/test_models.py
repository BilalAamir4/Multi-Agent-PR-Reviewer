"""src/reviewer/models.py: classify, retry_after_seconds, RateLimiter, WaitLog, QuotaTracker and
QuotaRouter. No network and no model calls: errors are the real provider exception classes
built in the test (wrapped by the real LangChain handlers), and clocks and sleeps are fakes."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import google.genai.errors as genai_errors
import groq
import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import chat_models as lc_google
from langchain_groq import chat_models as lc_groq
from pydantic import BaseModel, ValidationError

from reviewer import models
from reviewer.config import load_models_config
from reviewer.models import (
    AllExhausted,
    ErrorKind,
    PromptTooLarge,
    QuotaPaused,
    QuotaRouter,
    QuotaTracker,
    RateLimiter,
    RetriesExhausted,
    classify,
    estimate_tokens,
    next_pacific_midnight,
    record_wait,
    retry_after_seconds,
    review_waits,
    total_wait_seconds,
)

CONFIG = load_models_config()
PROVIDERS = {m: s.provider for m, s in CONFIG.models.items()}

# --- Building real provider errors -----------------------------------------------------------


def gemini_body(code, status, message, details=None):
    error = {"code": code, "message": message, "status": status}
    if details is not None:
        error["details"] = details
    return {"error": error}


def quota_failure(quota_id, metric="generate_content_free_tier_requests"):
    return {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
            {
                "quotaMetric": f"generativelanguage.googleapis.com/{metric}",
                "quotaId": quota_id,
                "quotaDimensions": {"location": "global", "model": "gemini-3.1-flash-lite"},
                "quotaValue": "15",
            }
        ],
    }


HELP = {
    "@type": "type.googleapis.com/google.rpc.Help",
    "links": [{"description": "Learn more", "url": "https://ai.google.dev/gemini-api/docs"}],
}
QUOTA_MSG = (
    "You exceeded your current quota, please check your plan and billing details. For more "
    "information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits."
)


def retry_info(delay):
    return {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}


def gemini_raw(code, body):
    """The google.genai error the SDK raises for an HTTP error (as APIError.raise_error does)."""
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body, None)


def gemini_wrapped(code, body):
    """The exception langchain-google-genai actually raises for that SDK error."""
    raw = gemini_raw(code, body)
    try:
        try:
            raise raw
        except genai_errors.ClientError as e:
            lc_google._handle_client_error(e, {"model": "models/gemini-3.1-flash-lite"})
        except genai_errors.ServerError as e:
            lc_google._handle_server_error(e)
    except Exception as wrapped:
        return wrapped
    raise AssertionError("handler did not raise")


def gemini_minute(delay="41s"):
    details = [
        quota_failure("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
        HELP,
    ]
    if delay is not None:
        details.append(retry_info(delay))
    return gemini_wrapped(429, gemini_body(429, "RESOURCE_EXHAUSTED", QUOTA_MSG, details))


def gemini_input_tokens_minute():
    details = [
        quota_failure(
            "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "generate_content_free_tier_input_token_count",
        ),
        retry_info("12s"),
    ]
    return gemini_wrapped(429, gemini_body(429, "RESOURCE_EXHAUSTED", QUOTA_MSG, details))


def gemini_daily():
    details = [
        quota_failure("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
        HELP,
        retry_info("17s"),  # Gemini sends a short delay even for the daily quota
    ]
    return gemini_wrapped(429, gemini_body(429, "RESOURCE_EXHAUSTED", QUOTA_MSG, details))


def gemini_overloaded():
    body = gemini_body(503, "UNAVAILABLE", "The model is overloaded. Please try again later.")
    return gemini_wrapped(503, body)


def gemini_too_big():
    msg = "The input token count (1200345) exceeds the maximum number of tokens allowed (1048576)."
    return gemini_wrapped(400, gemini_body(400, "INVALID_ARGUMENT", msg))


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def groq_error(status, body, headers=None):
    """The groq SDK's own status-code mapping (``_make_status_error_from_response``)."""
    request = httpx.Request("POST", GROQ_URL)
    response = httpx.Response(status, json=body, headers=headers or {}, request=request)
    client = groq.Groq(api_key="gsk_test_not_a_key", max_retries=0)
    return client._make_status_error_from_response(response)


def groq_rate(window, label, extra="Please try again in 7.5s."):
    message = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01test` service "
        f"tier `on_demand` on {window} ({label}): Limit 8000, Used 6100, Requested 2900. {extra} "
        "Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
    )
    return {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}


def groq_minute():
    return groq_error(429, groq_rate("tokens per minute", "TPM"), {"retry-after": "8"})


def groq_daily_tokens():
    body = groq_rate("tokens per day", "TPD", "Please try again in 12m57.6s.")
    return groq_error(429, body, {"retry-after": "778"})


def groq_daily_requests():
    body = groq_rate("requests per day", "RPD", "Please try again in 1m26.4s.")
    return groq_error(429, body, {"retry-after": "87"})


def groq_too_big_413():
    message = (
        "Request too large for model `openai/gpt-oss-120b` in organization `org_01test` service "
        "tier `on_demand` on tokens per minute (TPM): Limit 8000, Requested 12034, please reduce "
        "your message size and try again."
    )
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    return groq_error(413, body)


def groq_context_overflow():
    body = {
        "error": {
            "message": "Please reduce the length of the messages or completion.",
            "type": "invalid_request_error",
            "param": "messages",
        }
    }
    raw = groq_error(400, body)
    try:
        lc_groq._handle_groq_invalid_request(raw)
    except Exception as wrapped:
        return wrapped
    raise AssertionError("handler did not raise")


def groq_server(status):
    body = {"error": {"message": "Service Unavailable", "type": "internal_server_error"}}
    return groq_error(status, body)


def schema_error():
    class M(BaseModel):
        x: int

    try:
        M.model_validate({"x": "not a number"})
    except ValidationError as e:
        return e
    raise AssertionError


# --- Task 1: classify ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "kind"),
    [
        # Gemini, through langchain-google-genai
        (gemini_minute, ErrorKind.MINUTE),
        (gemini_input_tokens_minute, ErrorKind.MINUTE),
        (gemini_daily, ErrorKind.DAILY),
        (gemini_overloaded, ErrorKind.SERVER),
        (lambda: gemini_wrapped(500, gemini_body(500, "INTERNAL", "Internal error")), "SERVER"),
        (lambda: gemini_wrapped(504, gemini_body(504, "DEADLINE_EXCEEDED", "Deadline")), "SERVER"),
        (gemini_too_big, ErrorKind.TOO_BIG),
        (lambda: gemini_raw(413, gemini_body(413, "", "Request Entity Too Large")), "TOO_BIG"),
        (
            lambda: gemini_wrapped(
                400, gemini_body(400, "INVALID_ARGUMENT", 'Invalid JSON payload. Unknown name "x"')
            ),
            ErrorKind.FATAL,
        ),
        (
            lambda: gemini_wrapped(
                400, gemini_body(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid.")
            ),
            ErrorKind.FATAL,
        ),
        (lambda: gemini_wrapped(403, gemini_body(403, "PERMISSION_DENIED", "Denied")), "FATAL"),
        (
            lambda: gemini_wrapped(
                404, gemini_body(404, "NOT_FOUND", "models/x is not found for API version v1beta")
            ),
            ErrorKind.FATAL,
        ),
        (lambda: httpx.ConnectError("[Errno 11001] getaddrinfo failed"), ErrorKind.SERVER),
        (lambda: httpx.ReadTimeout("The read operation timed out"), ErrorKind.SERVER),
        # Groq, through the groq SDK (langchain-groq passes these through) and langchain-groq
        (groq_minute, ErrorKind.MINUTE),
        (groq_daily_tokens, ErrorKind.DAILY),
        (groq_daily_requests, ErrorKind.DAILY),
        (groq_too_big_413, ErrorKind.TOO_BIG),
        (groq_context_overflow, ErrorKind.TOO_BIG),
        (lambda: groq_server(500), ErrorKind.SERVER),
        (lambda: groq_server(502), ErrorKind.SERVER),
        (lambda: groq_server(503), ErrorKind.SERVER),
        (lambda: groq_server(504), ErrorKind.SERVER),
        (
            lambda: groq.APIConnectionError(request=httpx.Request("POST", GROQ_URL)),
            ErrorKind.SERVER,
        ),
        (lambda: groq.APITimeoutError(request=httpx.Request("POST", GROQ_URL)), ErrorKind.SERVER),
        (
            lambda: groq_error(
                401,
                {"error": {"message": "Invalid API Key", "code": "invalid_api_key"}},
            ),
            ErrorKind.FATAL,
        ),
        (
            lambda: groq_error(
                400,
                {"error": {"message": "Failed to call a function.", "code": "tool_use_failed"}},
            ),
            ErrorKind.FATAL,
        ),
        (
            lambda: groq_error(
                404, {"error": {"message": "does not exist", "code": "model_not_found"}}
            ),
            ErrorKind.FATAL,
        ),
        # Schema and parsing errors
        (schema_error, ErrorKind.FATAL),
        (lambda: OutputParserException("Invalid json output"), ErrorKind.FATAL),
        (lambda: ValueError("anything else"), ErrorKind.FATAL),
    ],
)
def test_classify(make, kind):
    kind = ErrorKind[kind] if isinstance(kind, str) else kind
    assert classify(make()) is kind


def test_classify_sees_through_the_langchain_wrapper():
    err = gemini_daily()
    # langchain-google-genai keeps only a message on the outer exception
    assert type(err).__name__ == "GoogleRateLimitError"
    assert isinstance(err.__cause__, genai_errors.ClientError)
    assert classify(err) is ErrorKind.DAILY
    assert classify(err.__cause__) is ErrorKind.DAILY


def test_classify_sees_an_error_raised_inside_an_agent_task():
    # An agent framework may raise its own error from the provider error.
    try:
        try:
            raise groq_daily_tokens()
        except groq.RateLimitError as e:
            raise RuntimeError("agent step failed") from e
    except RuntimeError as outer:
        assert classify(outer) is ErrorKind.DAILY


def test_retry_after_seconds():
    assert retry_after_seconds(gemini_minute("41s")) == 41.0
    assert retry_after_seconds(gemini_minute("2.5s")) == 2.5
    assert retry_after_seconds(gemini_minute(None)) is None
    assert retry_after_seconds(gemini_overloaded()) is None
    assert retry_after_seconds(groq_minute()) == 8.0
    assert retry_after_seconds(groq_daily_tokens()) == 778.0
    ms = groq_error(429, groq_rate("tokens per minute", "TPM"), {"retry-after-ms": "1500"})
    assert retry_after_seconds(ms) == 1.5
    date = groq_error(
        429, groq_rate("tokens per minute", "TPM"), {"retry-after": "Fri, 20 Nov 2026 12:00:30 GMT"}
    )
    now = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC).timestamp()
    assert retry_after_seconds(date, now=now) == 30.0
    assert retry_after_seconds(ValueError("x")) is None


# --- Task 2: rate limiter, waits, quota tracker ----------------------------------------------


class FakeTime:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.mark.parametrize(("rpm", "limit"), [(15, 13), (5, 4), (30, 27), (1, 1)])
def test_rate_limiter_allows_90_percent_of_rpm_per_rolling_minute(rpm, limit):
    ft = FakeTime()
    rl = RateLimiter(rpm, clock=ft.clock, sleep=ft.sleep)
    assert rl.limit == limit
    for _ in range(limit):
        assert rl.acquire() == 0.0
        ft.t += 1.0  # one request a second
    # The next one waits until the first start leaves the rolling 60 seconds.
    assert rl.acquire() == pytest.approx(60.0 - limit)
    assert ft.sleeps == [pytest.approx(60.0 - limit)]


def test_rate_limiter_is_rolling_not_fixed_window():
    ft = FakeTime()
    rl = RateLimiter(5, clock=ft.clock, sleep=ft.sleep)  # 4 per minute
    for at in (0, 10, 20, 50):
        ft.t = 1000.0 + at
        assert rl.acquire() == 0.0
    ft.t = 1000.0 + 59
    assert rl.acquire() == pytest.approx(1.0)  # the start at 0 frees at 60
    ft.t = 1000.0 + 65
    assert rl.acquire() == pytest.approx(5.0)  # the start at 10 frees at 70


def test_rate_limiter_null_rpm_is_one_per_ten_seconds():
    ft = FakeTime()
    rl = RateLimiter(None, clock=ft.clock, sleep=ft.sleep)
    assert rl.acquire() == 0.0
    assert rl.acquire() == pytest.approx(10.0)
    assert rl.acquire() == pytest.approx(10.0)


def test_rate_limiter_is_thread_safe():
    ft = FakeTime()
    lock = threading.Lock()

    def sleep(seconds):
        with lock:
            ft.sleep(seconds)

    rl = RateLimiter(15, clock=ft.clock, sleep=sleep)  # 13 per minute
    results: list[float] = []

    def worker():
        results.append(rl.acquire())

    threads = [threading.Thread(target=worker) for _ in range(13)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [0.0] * 13  # all fit, none double-counted
    assert len(rl._starts) == 13


def test_wait_log_totals_and_reasons():
    with review_waits() as wl:
        record_wait("m1", "rate_limiter", 2.0)
        record_wait("m1", "retry_minute", 10.0)
        record_wait("m2", "retry_server", 20.0)
        record_wait("m2", "rate_limiter", 0.0)  # not a wait
        assert total_wait_seconds() == 32.0
        assert wl.by_reason() == {"rate_limiter": 2.0, "retry_minute": 10.0, "retry_server": 20.0}
        assert len(wl.waits) == 3
    assert total_wait_seconds() == 0.0  # outside the review


def test_wait_logs_are_separate_per_thread():
    barrier = threading.Barrier(2)
    totals: dict[str, float] = {}

    def review(name, seconds):
        with review_waits() as wl:
            barrier.wait()
            record_wait(name, "retry_minute", seconds)
            barrier.wait()
            totals[name] = wl.total()

    threads = [
        threading.Thread(target=review, args=("a", 1.0)),
        threading.Thread(target=review, args=("b", 5.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert totals == {"a": 1.0, "b": 5.0}


class FakeNow:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def __call__(self) -> datetime:
        return self.at


@pytest.mark.parametrize(
    ("detected", "reset"),
    [
        # Summer (PDT, UTC-7): 13:00 local on Jul 15 -> midnight Jul 16 = 07:00 UTC
        ("2026-07-15T20:00:00+00:00", "2026-07-16T07:00:00+00:00"),
        # Still Jul 15 in Los Angeles (23:59 PDT)
        ("2026-07-16T06:59:00+00:00", "2026-07-16T07:00:00+00:00"),
        # Winter (PST, UTC-8): 12:00 local on Dec 1 -> midnight Dec 2 = 08:00 UTC
        ("2026-12-01T20:00:00+00:00", "2026-12-02T08:00:00+00:00"),
        # Daylight saving ends Sun Nov 1, 2026: 23:00 PDT Oct 31 -> 00:00 PDT Nov 1
        ("2026-11-01T06:00:00+00:00", "2026-11-01T07:00:00+00:00"),
        # Later on Nov 1 (PST) -> 00:00 PST Nov 2
        ("2026-11-01T20:00:00+00:00", "2026-11-02T08:00:00+00:00"),
        # Exactly at midnight: the next one, a day later
        ("2026-12-02T08:00:00+00:00", "2026-12-03T08:00:00+00:00"),
    ],
)
def test_next_pacific_midnight(detected, reset):
    assert next_pacific_midnight(datetime.fromisoformat(detected)) == datetime.fromisoformat(reset)


def test_tracker_gemini_reset_persists_and_expires(tmp_path):
    path = tmp_path / "quota_state.json"
    now = FakeNow(datetime(2026, 7, 15, 20, 0, tzinfo=UTC))  # during daylight saving time
    t = QuotaTracker(path, PROVIDERS, now=now)
    reset = t.mark_exhausted("gemini-3.8-flash", retry_after=17.0)  # Gemini ignores retry_after
    assert reset == datetime(2026, 7, 16, 7, 0, tzinfo=UTC)
    assert json.loads(path.read_bytes()) == {"gemini-3.8-flash": "2026-07-16T07:00:00+00:00"}
    assert not list(tmp_path.glob(".quota_state.json.*"))  # temp file renamed away

    # A restart reloads it.
    t2 = QuotaTracker(path, PROVIDERS, now=now)
    assert t2.is_exhausted("gemini-3.8-flash")
    assert t2.reset_at("gemini-3.8-flash") == reset

    # Expired at the reset time; dropped from memory and from the file.
    now.at = reset
    assert not t2.is_exhausted("gemini-3.8-flash")
    assert json.loads(path.read_bytes()) == {}


def test_tracker_drops_expired_entries_on_load(tmp_path):
    path = tmp_path / "quota_state.json"
    path.write_text(
        json.dumps(
            {
                "gemini-3.8-flash": "2026-12-02T08:00:00+00:00",
                "openai/gpt-oss-120b": "2026-12-01T00:00:00+00:00",
            }
        )
    )
    t = QuotaTracker(path, PROVIDERS, now=FakeNow(datetime(2026, 12, 1, 20, 0, tzinfo=UTC)))
    assert t.exhausted() == {"gemini-3.8-flash": datetime(2026, 12, 2, 8, 0, tzinfo=UTC)}
    assert list(json.loads(path.read_bytes())) == ["gemini-3.8-flash"]


def test_tracker_groq_reset(tmp_path):
    at = datetime(2026, 12, 1, 20, 0, tzinfo=UTC)
    t = QuotaTracker(tmp_path / "q.json", PROVIDERS, now=FakeNow(at))
    assert t.mark_exhausted("openai/gpt-oss-120b") == at + timedelta(hours=24)
    assert t.mark_exhausted("qwen/qwen3.8-27b", retry_after=778.0) == at + timedelta(seconds=778)


# --- Task 3: the router ----------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    ft = FakeTime()
    now = FakeNow(datetime(2026, 12, 1, 20, 0, tzinfo=UTC))
    tracker = QuotaTracker(tmp_path / "quota_state.json", PROVIDERS, now=now)

    def make(evaluation=False):
        r = QuotaRouter(
            CONFIG,
            evaluation=evaluation,
            tracker=tracker,
            limiters={},
            clock=ft.clock,
            sleep=ft.sleep,
        )
        r.build = lambda model_id: f"chat:{model_id}"
        return r

    return make, tracker, ft, now


class Script:
    """A fake agent task: per model, a list of exceptions to raise before succeeding."""

    def __init__(self, plan: dict[str, list]) -> None:
        self.plan = {m: list(v) for m, v in plan.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, chat_model, model_id):
        self.calls.append((chat_model, model_id))
        steps = self.plan.get(model_id, [])
        if steps:
            step = steps.pop(0)
            raise step() if callable(step) else step
        return f"review by {model_id}"

    @property
    def models(self):
        return [m for _, m in self.calls]


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 2
    assert QuotaRouter.estimate_tokens("x" * 30000) == 10000


def test_build_turns_client_retries_off(env, monkeypatch):
    make, *_ = env
    calls = []
    monkeypatch.setattr(models, "init_chat_model", lambda *a, **kw: calls.append((a, kw)))
    r = QuotaRouter(CONFIG, tracker=make().tracker, limiters={})
    r.build("openai/gpt-oss-120b")
    r.build("gemini-3.1-flash-lite")
    assert calls == [
        (("openai/gpt-oss-120b",), {"model_provider": "groq", "temperature": 0, "max_retries": 0}),
        (
            ("gemini-3.1-flash-lite",),
            {"model_provider": "google_genai", "temperature": 0, "max_retries": 0},
        ),
    ]


def test_pick_follows_chain_order(env):
    make, tracker, *_ = env
    r = make()
    assert r.pick("orchestrator", 1000) == "gemini-3.8-flash"
    assert r.pick("verifier", 1000) == "openai/gpt-oss-120b"
    tracker.mark_exhausted("gemini-3.8-flash")
    tracker.mark_exhausted("gemini-3.6-flash")
    assert r.pick("orchestrator", 1000) == "gemini-3-flash-preview"


def test_pick_skips_models_whose_tpm_is_too_small(env):
    make, tracker, *_ = env
    r = make()
    tracker.mark_exhausted("gemini-3.1-flash-lite")
    # worker: 3.1 Flash Lite (exhausted), gpt-oss-120b 8K, qwen 8K, Gemma 26B 16K
    assert r.pick("worker", 7200) == "openai/gpt-oss-120b"  # 7200 / 0.9 = 8000
    assert r.pick("worker", 7201) == "gemma-4-26b-a4b-it"
    with pytest.raises(AllExhausted) as e:
        r.pick("worker", 14401)
    assert "needs TPM >= 16002, model has 16000" in str(e.value)


def test_success_is_tagged_with_its_model(env):
    make, *_ = env
    task = Script({})
    res = make().run("specialist", task, 1000)
    assert (res.value, res.model) == ("review by gemini-3.5-flash-lite", "gemini-3.5-flash-lite")
    assert task.calls == [("chat:gemini-3.5-flash-lite", "gemini-3.5-flash-lite")]


def test_minute_retries_same_model_twice_then_falls_over(env):
    make, tracker, ft, _ = env
    task = Script({"gemini-3.5-flash-lite": [lambda: gemini_minute(None)] * 3})
    with review_waits() as wl:
        res = make().run("specialist", task, 1000)
    assert task.models == ["gemini-3.5-flash-lite"] * 3 + ["gemini-3.1-flash-lite"]
    assert res.model == "gemini-3.1-flash-lite"
    assert ft.sleeps == [10.0, 20.0]  # 10 x 2^attempt
    assert [(w.model, w.reason, w.seconds) for w in wl.waits] == [
        ("gemini-3.5-flash-lite", "retry_minute", 10.0),
        ("gemini-3.5-flash-lite", "retry_minute", 20.0),
    ]
    assert tracker.exhausted() == {}  # a minute limit marks nothing


def test_minute_uses_retry_after_capped_at_30(env):
    make, _, ft, _ = env
    task = Script(
        {"gemini-3.5-flash-lite": [lambda: gemini_minute("41s"), gemini_input_tokens_minute]}
    )
    assert make().run("specialist", task, 1000).model == "gemini-3.5-flash-lite"
    assert ft.sleeps == [30.0, 12.0]


def test_server_errors_retry_twice_then_fall_over(env):
    make, tracker, ft, _ = env
    task = Script({"openai/gpt-oss-120b": [lambda: groq_server(503)] * 3})
    with review_waits() as wl:
        res = make().run("verifier", task, 1000)
    assert task.models == ["openai/gpt-oss-120b"] * 3 + ["qwen/qwen3.8-27b"]
    assert res.model == "qwen/qwen3.8-27b"
    assert ft.sleeps == [10.0, 20.0]
    assert {w.reason for w in wl.waits} == {"retry_server"}
    assert tracker.exhausted() == {}


def test_overloaded_then_success_stays_on_model(env):
    make, *_ = env
    task = Script({"gemini-3.8-flash": [gemini_overloaded]})
    assert make().run("orchestrator", task, 1000).model == "gemini-3.8-flash"


def test_daily_falls_over_at_once_and_persists(env, tmp_path):
    make, tracker, ft, now = env
    task = Script({"gemini-3.8-flash": [gemini_daily]})
    res = make().run("orchestrator", task, 1000)
    assert task.models == ["gemini-3.8-flash", "gemini-3.6-flash"]  # whole task re-run (S4)
    assert task.calls[1][0] == "chat:gemini-3.6-flash"
    assert res.model == "gemini-3.6-flash"
    assert ft.sleeps == []
    assert tracker.reset_at("gemini-3.8-flash") == datetime(2026, 12, 2, 8, 0, tzinfo=UTC)
    # A restarted process skips it.
    restarted = QuotaTracker(tmp_path / "quota_state.json", PROVIDERS, now=now)
    assert restarted.is_exhausted("gemini-3.8-flash")
    r = make()
    r.tracker = restarted
    assert r.pick("orchestrator", 1000) == "gemini-3.6-flash"


def test_groq_daily_uses_retry_after_for_reset(env):
    make, tracker, _, now = env
    task = Script({"openai/gpt-oss-120b": [groq_daily_tokens]})
    assert make().run("verifier", task, 1000).model == "qwen/qwen3.8-27b"
    assert tracker.reset_at("openai/gpt-oss-120b") == now.at + timedelta(seconds=778)


def test_fatal_raises_without_retry_or_fallback(env):
    make, tracker, ft, _ = env
    err = groq_error(401, {"error": {"message": "Invalid API Key", "code": "invalid_api_key"}})
    task = Script({"openai/gpt-oss-120b": [err]})
    with pytest.raises(groq.AuthenticationError):
        make().run("verifier", task, 1000)
    assert task.models == ["openai/gpt-oss-120b"]
    assert ft.sleeps == [] and tracker.exhausted() == {}


def test_too_big_skips_models_with_no_larger_tpm_without_marking(env):
    make, tracker, *_ = env
    tracker.mark_exhausted("gemini-3.5-flash-lite")
    tracker.mark_exhausted("gemini-3.1-flash-lite")
    # specialist: ..., Gemma 4 31B (16K), gpt-oss-120b (8K). Gemma rejects a 5,000-token estimate.
    task = Script({"gemma-4-31b-it": [gemini_too_big]})
    with pytest.raises(AllExhausted) as e:
        make().run("specialist", task, 5000)
    assert task.models == ["gemma-4-31b-it"]  # gpt-oss-120b was never tried
    assert e.value.reasons["gemma-4-31b-it"] == "TOO_BIG: the provider rejected the prompt"
    assert "8000 <= 16000" in e.value.reasons["openai/gpt-oss-120b"]
    assert not tracker.is_exhausted("gemma-4-31b-it")


def test_too_big_falls_over_to_a_larger_model(env):
    make, *_ = env
    task = Script({"openai/gpt-oss-120b": [groq_too_big_413], "qwen/qwen3.8-27b": []})
    # verifier: gpt-oss-120b (8K), qwen (8K, skipped), 3.6 Flash (250K)
    res = make().run("verifier", task, 1000)
    assert task.models == ["openai/gpt-oss-120b", "gemini-3.6-flash"]
    assert res.model == "gemini-3.6-flash"


def test_all_exhausted_lists_every_skip_reason(env):
    make, tracker, *_ = env
    tracker.mark_exhausted("gemini-3.6-flash")
    task = Script(
        {
            "openai/gpt-oss-120b": [groq_daily_requests],
            "qwen/qwen3.8-27b": [lambda: groq_server(502)] * 3,
        }
    )
    with pytest.raises(AllExhausted) as e:
        make().run("verifier", task, 1000)
    assert e.value.role == "verifier"
    assert list(e.value.reasons) == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemini-3.6-flash"]
    msg = str(e.value)
    assert msg.startswith("no usable model for role 'verifier': openai/gpt-oss-120b: DAILY quota")
    assert "qwen/qwen3.8-27b: SERVER: still failing after 2 retries" in msg
    assert "gemini-3.6-flash: daily quota exhausted until 2026-12-02T08:00:00+00:00" in msg


def test_rate_limiter_wait_is_logged(env):
    make, _, ft, _ = env
    r = make()
    task = Script({})
    with review_waits() as wl:
        for _ in range(5):  # 3.8 Flash: 5 RPM -> 4 per rolling minute
            r.run("orchestrator", task, 100)
        assert [(w.reason, w.seconds) for w in wl.waits] == [("rate_limiter", 60.0)]
        assert total_wait_seconds() == 60.0


def test_waits_sum_all_three_reasons(env):
    make, _, ft, _ = env
    r = make()
    task = Script({"gemini-3.8-flash": [lambda: gemini_minute(None), gemini_overloaded]})
    with review_waits() as wl:
        for _ in range(4):
            r.run("orchestrator", Script({}), 100)
        r.run("orchestrator", task, 100)
    reasons = wl.by_reason()
    assert set(reasons) == {"rate_limiter", "retry_minute", "retry_server"}
    assert reasons["retry_minute"] == 10.0 and reasons["retry_server"] == 20.0
    assert wl.total() == pytest.approx(sum(reasons.values()))


def test_router_waits_in_two_threads_are_separate(env):
    make, *_ = env
    r = make()
    barrier = threading.Barrier(2)
    out: dict[str, dict[str, float]] = {}

    def review(name, err):
        with review_waits() as wl:
            barrier.wait()
            r.run("verifier", Script({"openai/gpt-oss-120b": [err]}), 100)
            out[name] = wl.by_reason()

    threads = [
        threading.Thread(target=review, args=("minute", lambda: groq_minute())),
        threading.Thread(target=review, args=("server", lambda: groq_server(500))),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == {"minute": {"retry_minute": 8.0}, "server": {"retry_server": 10.0}}


# --- Evaluation mode (S9, D4) ----------------------------------------------------------------


def test_evaluation_ignores_chains(env):
    make, *_ = env
    r = make(evaluation=True)
    for role in ("orchestrator", "specialist", "worker", "verifier"):
        assert r.pick(role, 1000) == "gemini-3.1-flash-lite"
        assert r.run(role, Script({}), 1000).model == "gemini-3.1-flash-lite"


def test_evaluation_retries_five_times_and_never_falls_back(env):
    make, tracker, ft, _ = env
    task = Script({"gemini-3.1-flash-lite": [lambda: gemini_minute(None)] * 6})
    with pytest.raises(RetriesExhausted) as e:
        make(evaluation=True).run("specialist", task, 1000)
    assert task.models == ["gemini-3.1-flash-lite"] * 6
    assert ft.sleeps == [10.0, 20.0, 30.0, 30.0, 30.0]
    assert e.value.kind is ErrorKind.MINUTE
    assert tracker.exhausted() == {}


def test_evaluation_server_errors_recover_within_five_retries(env):
    make, _, ft, _ = env
    task = Script({"gemini-3.1-flash-lite": [gemini_overloaded] * 5})
    with review_waits() as wl:
        res = make(evaluation=True).run("worker", task, 1000)
    assert res.model == "gemini-3.1-flash-lite" and len(task.calls) == 6
    assert {w.reason for w in wl.waits} == {"retry_server"}


def test_evaluation_daily_pauses(env):
    make, tracker, ft, _ = env
    task = Script({"gemini-3.1-flash-lite": [gemini_daily]})
    r = make(evaluation=True)
    with pytest.raises(QuotaPaused) as e:
        r.run("specialist", task, 1000)
    assert task.models == ["gemini-3.1-flash-lite"]
    assert e.value.model == "gemini-3.1-flash-lite"
    assert e.value.reset_at == datetime(2026, 12, 2, 8, 0, tzinfo=UTC)
    assert tracker.is_exhausted("gemini-3.1-flash-lite") and ft.sleeps == []
    # Until the reset, every run pauses before calling the model.
    with pytest.raises(QuotaPaused):
        r.run("orchestrator", Script({}), 1000)


def test_evaluation_too_big_raises(env):
    make, tracker, *_ = env
    task = Script({"gemini-3.1-flash-lite": [gemini_too_big]})
    with pytest.raises(PromptTooLarge, match="no fallback model"):
        make(evaluation=True).run("specialist", task, 1000)
    with pytest.raises(PromptTooLarge, match="TPM 250000"):
        make(evaluation=True).run("specialist", Script({}), 250000)
    assert tracker.exhausted() == {}


def test_evaluation_fatal_raises(env):
    make, *_ = env
    task = Script({"gemini-3.1-flash-lite": [schema_error]})
    with pytest.raises(ValidationError):
        make(evaluation=True).run("specialist", task, 1000)
