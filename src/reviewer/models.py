"""Quota-aware model router (docs/model-routing.md, S1 to S9; decision 0006).

- ``classify`` / ``retry_after_seconds``: what a provider error means for the router.
- ``RateLimiter``: one per model, under 90% of its RPM.
- ``WaitLog`` / ``review_waits``: every wait in one review (D7).
- ``QuotaTracker``: models whose daily quota is gone, persisted to ``quota_state.json`` (S2, S5).
- ``QuotaRouter``: picks a model per role, runs one whole agent task, retries or falls over.

The router owns all retries: chat models are built with ``max_retries=0``. Same-model retries
(``MINUTE`` and ``SERVER``) happen per model call, inside the agent loop (``CallRetrier``, decision
0010); the router only re-runs a whole task to fall over to another model (S4).
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import google.genai.errors as genai_errors
import groq
import httpx
import httpx2
from langchain.chat_models import init_chat_model
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel

from reviewer.config import ModelsConfig, load_models_config, repo_root

log = logging.getLogger(__name__)

PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini daily quotas reset at midnight here (S2)
RPM_HEADROOM = 0.9  # rate limiter and TPM check stay under 90% of the published limit
NULL_RPM_INTERVAL_S = 10.0  # models with rpm: null get one request per 10 seconds
GROQ_DAILY_RESET = timedelta(hours=24)
SERVER_STATUS = frozenset({500, 502, 503, 504})


# --- Task 1: error classification -----------------------------------------------------------


class ErrorKind(enum.Enum):
    TOO_BIG = "too_big"  # S3: this prompt never fits this model; skip it, mark nothing
    DAILY = "daily"  # S2: daily requests or tokens are gone; mark exhausted until reset
    MINUTE = "minute"  # S1: per-minute limit; wait and retry the same model
    SERVER = "server"  # 5xx, overloaded, connection or timeout; wait and retry
    FATAL = "fatal"  # anything else (400, 401, 403, 404, schema errors); never retried


_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    # google-genai talks to the Gemini API through httpx (or the httpx2 fork) and lets their
    # transport errors through unwrapped.
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx2.TimeoutException,
    httpx2.NetworkError,
    httpx2.RemoteProtocolError,
    groq.APIConnectionError,  # includes groq.APITimeoutError
    ModelConnectionError,
    ModelTimeoutError,
    ConnectionError,
    TimeoutError,
)


def _causes(err: BaseException) -> Iterator[BaseException]:
    """``err`` and the exceptions it was raised from, outermost first.

    langchain-google-genai re-raises 4xx errors as ``GoogleRateLimitError(msg) from e`` and
    similar, which keep only a message; the structured ``google.genai`` error is ``__cause__``.
    """
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen and len(seen) < 10:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)


def _genai_error_body(err: genai_errors.APIError) -> dict[str, Any]:
    """The ``error`` object of a Gemini error body: ``{"code", "message", "status", "details"}``."""
    body = err.details
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    if not isinstance(body, dict):
        return {}
    inner = body.get("error")
    return inner if isinstance(inner, dict) else body


def _genai_detail(err: genai_errors.APIError, type_suffix: str) -> list[dict[str, Any]]:
    """Entries of ``error.details`` whose ``@type`` ends with ``type_suffix``
    (``google.rpc.QuotaFailure``, ``google.rpc.RetryInfo``)."""
    details = _genai_error_body(err).get("details")
    if not isinstance(details, list):
        return []
    return [
        d for d in details if isinstance(d, dict) and str(d.get("@type", "")).endswith(type_suffix)
    ]


def _genai_quota_ids(err: genai_errors.APIError) -> list[str]:
    """``quotaId`` of every ``QuotaFailure`` violation, e.g.
    ``GenerateRequestsPerDayPerProjectPerModel-FreeTier``."""
    ids = []
    for failure in _genai_detail(err, "google.rpc.QuotaFailure"):
        for v in failure.get("violations") or []:
            if isinstance(v, dict) and v.get("quotaId"):
                ids.append(str(v["quotaId"]))
    return ids


def _genai_message(err: genai_errors.APIError) -> str:
    return str(err.message or _genai_error_body(err).get("message") or err)


def _classify_genai(err: genai_errors.APIError) -> ErrorKind:
    code = err.code
    if code == 413:
        return ErrorKind.TOO_BIG
    if code == 429 or err.status == "RESOURCE_EXHAUSTED":
        quota_ids = _genai_quota_ids(err)
        if quota_ids:
            return ErrorKind.DAILY if any("PerDay" in q for q in quota_ids) else ErrorKind.MINUTE
        # Fallback: a 429 without a QuotaFailure detail. Only the message can name the window.
        msg = _genai_message(err).lower()
        return ErrorKind.DAILY if "per day" in msg or "perday" in msg else ErrorKind.MINUTE
    if code in SERVER_STATUS:
        return ErrorKind.SERVER  # includes 503 UNAVAILABLE "The model is overloaded"
    if code == 400:
        # Fallback: Gemini's input-token-limit error is a plain 400 INVALID_ARGUMENT; only the
        # message identifies it. Same test langchain-google-genai uses for
        # GoogleContextOverflowError, for a raw google.genai ClientError.
        msg = _genai_message(err)
        if "exceeds the maximum number of tokens allowed" in msg or "token limit" in msg.lower():
            return ErrorKind.TOO_BIG
    return ErrorKind.FATAL


def _groq_error_body(err: groq.APIStatusError) -> dict[str, Any]:
    """The ``error`` object of a Groq error body: ``{"message", "type", "code"}``."""
    body = err.body
    if isinstance(body, dict):
        inner = body.get("error")
        return inner if isinstance(inner, dict) else body
    return {}


_GROQ_DAILY = re.compile(r"per day|\((?:RPD|TPD)\)", re.IGNORECASE)


def _classify_groq(err: groq.APIStatusError) -> ErrorKind:
    status = err.status_code
    body = _groq_error_body(err)
    msg = str(body.get("message") or err.message or err)
    if status == 413:
        return ErrorKind.TOO_BIG
    if status == 400 and body.get("code") == "context_length_exceeded":
        return ErrorKind.TOO_BIG
    if status == 429:
        # Fallback: Groq's body has "type" (tokens / requests) and "code" (rate_limit_exceeded)
        # but no field for the window, so the message is read:
        # "... on tokens per day (TPD): Limit 200000, Used ..." or "... (RPM): ...".
        if "request too large" in msg.lower():
            return ErrorKind.TOO_BIG
        return ErrorKind.DAILY if _GROQ_DAILY.search(msg) else ErrorKind.MINUTE
    if status in SERVER_STATUS:
        return ErrorKind.SERVER
    return ErrorKind.FATAL


def classify(err: BaseException) -> ErrorKind:
    """What ``err``, raised by a chat model call or by an agent task, means for the router."""
    chain = list(_causes(err))
    # langchain-google-genai and langchain-groq both raise ContextOverflowError subclasses for
    # an input over the model's limit (GoogleContextOverflowError, GroqContextOverflowError).
    if any(isinstance(e, ContextOverflowError) for e in chain):
        return ErrorKind.TOO_BIG
    for e in chain:
        if isinstance(e, genai_errors.APIError):
            return _classify_genai(e)
        if isinstance(e, groq.APIStatusError):
            return _classify_groq(e)
        if isinstance(e, _TRANSPORT_ERRORS):
            return ErrorKind.SERVER
    # No provider error in the chain: a LangChain error type raised by some other integration.
    for e in chain:
        if isinstance(e, ModelRateLimitError):
            # Fallback: no structured quota field, only the message.
            msg = str(e).lower()
            return ErrorKind.DAILY if "per day" in msg or "perday" in msg else ErrorKind.MINUTE
        if isinstance(e, ModelAPIError):
            return ErrorKind.SERVER
    return ErrorKind.FATAL


_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


def _header_retry_after(headers: Any, now: float) -> float | None:
    if headers is None:
        return None
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).timestamp() - now
    except (TypeError, ValueError):
        return None


def retry_after_seconds(err: BaseException, *, now: float | None = None) -> float | None:
    """Seconds the provider asked us to wait: Gemini's ``RetryInfo.retryDelay`` (``"41s"``) or a
    ``retry-after-ms`` / ``retry-after`` header (Groq sends these on 429). None if absent."""
    now = time.time() if now is None else now
    for e in _causes(err):
        if isinstance(e, genai_errors.APIError):
            for info in _genai_detail(e, "google.rpc.RetryInfo"):
                m = _DURATION.match(str(info.get("retryDelay", "")))
                if m:
                    return float(m.group(1))
            headers = getattr(e.response, "headers", None)
            seconds = _header_retry_after(headers, now)
            if seconds is not None:
                return max(0.0, seconds)
        elif isinstance(e, groq.APIStatusError):
            seconds = _header_retry_after(e.response.headers, now)
            if seconds is not None:
                return max(0.0, seconds)
    return None


# --- Task 2: waits, the rate limiter and the quota tracker -----------------------------------

type WaitReason = Literal["rate_limiter", "retry_minute", "retry_server"]


@dataclass(frozen=True)
class Wait:
    """One wait. ``end`` is when it finished on the recording clock (``time.monotonic`` unless the
    caller passes its own), so ``[end - seconds, end]`` is the interval it covered. ``nested`` is
    true for a wait forwarded from a child review's log (decision 0009)."""

    model: str
    reason: WaitReason
    seconds: float
    end: float = 0.0
    nested: bool = False


class WaitLog:
    """Every wait in one review (D7). Reports subtract the waiting from the review time.

    Logs nest (decision 0009): a log made while another is current (``review_waits()``) or given a
    ``parent`` forwards every wait it records to that parent, which stores it marked ``nested``.
    A wait is therefore stored once per log on its path to the root and counted once in each log's
    own totals, however many threads the children ran in. ``total()`` / ``by_reason()`` add the
    waits up; ``wall()`` is the time during which at least one wait was in progress (the union of
    the wait intervals), which is what D7 subtracts when children wait in parallel."""

    def __init__(self, parent: WaitLog | None = None) -> None:
        self._lock = threading.Lock()
        self._waits: list[Wait] = []
        self.parent = parent

    def record(
        self,
        model: str,
        reason: WaitReason,
        seconds: float,
        end: float | None = None,
        *,
        nested: bool = False,
    ) -> None:
        end = time.monotonic() if end is None else end
        with self._lock:
            self._waits.append(Wait(model, reason, seconds, end, nested))
        if self.parent is not None:
            self.parent.record(model, reason, seconds, end, nested=True)

    @property
    def waits(self) -> list[Wait]:
        with self._lock:
            return list(self._waits)

    def total(self) -> float:
        """All waits added up, children's included."""
        with self._lock:
            return sum(w.seconds for w in self._waits)

    def own_total(self) -> float:
        """Waits recorded directly in this review, not forwarded from a child."""
        with self._lock:
            return sum(w.seconds for w in self._waits if not w.nested)

    def by_reason(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for w in self.waits:
            out[w.reason] = out.get(w.reason, 0.0) + w.seconds
        return out

    def counts_by_reason(self) -> dict[str, int]:
        """How many waits of each reason were recorded (children's included)."""
        out: dict[str, int] = {}
        for w in self.waits:
            out[w.reason] = out.get(w.reason, 0) + 1
        return out

    def wall(self) -> float:
        """Seconds during which at least one wait was in progress (union of the intervals)."""
        spans = sorted((w.end - w.seconds, w.end) for w in self.waits if w.seconds > 0)
        total = 0.0
        cur_start = cur_end = None
        for start, end in spans:
            if cur_end is None or start > cur_end:
                if cur_end is not None:
                    total += cur_end - cur_start
                cur_start, cur_end = start, end
            else:
                cur_end = max(cur_end, end)
        if cur_end is not None:
            total += cur_end - cur_start
        return total


_current_waits: ContextVar[WaitLog | None] = ContextVar("reviewer_wait_log", default=None)


@contextmanager
def review_waits(parent: WaitLog | None = None) -> Iterator[WaitLog]:
    """Scope a new ``WaitLog`` to the current context (one review). Parallel reviews in their own
    threads or contexts get separate logs. The new log nests under ``parent``, or under the log
    that is current when this is entered (so a specialist's review counts toward the
    orchestrator's, also from a worker thread that copied the context)."""
    wait_log = WaitLog(parent if parent is not None else _current_waits.get())
    token = _current_waits.set(wait_log)
    try:
        yield wait_log
    finally:
        _current_waits.reset(token)


@contextmanager
def attach_waits(wait_log: WaitLog | None) -> Iterator[None]:
    """Make ``wait_log`` the current log in this thread (for code that runs in a thread that did
    not copy the context). ``review_waits()`` inside then nests under it."""
    token = _current_waits.set(wait_log)
    try:
        yield
    finally:
        _current_waits.reset(token)


def current_wait_log() -> WaitLog | None:
    return _current_waits.get()


def record_wait(
    model: str, reason: WaitReason, seconds: float, *, end: float | None = None
) -> None:
    """Add a wait to the current review's log. Zero waits and waits outside a review are not
    recorded (they are still logged at debug level). ``end`` is the clock reading when the wait
    finished (default ``time.monotonic()``)."""
    log.debug("wait %.2fs on %s (%s)", seconds, model, reason)
    wait_log = _current_waits.get()
    if wait_log is not None and seconds > 0:
        wait_log.record(model, reason, seconds, end)


def total_wait_seconds() -> float:
    """Total waiting in the current review: rate limiter plus both retry kinds (D7)."""
    wait_log = _current_waits.get()
    return wait_log.total() if wait_log is not None else 0.0


class RateLimiter:
    """At most ``floor(rpm * 0.9)`` requests (minimum 1) in any rolling 60 seconds; with
    ``rpm=None``, 1 request per 10 seconds. Thread-safe."""

    def __init__(
        self,
        rpm: int | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm is None:
            self.limit, self.window = 1, NULL_RPM_INTERVAL_S
        else:
            self.limit, self.window = max(1, math.floor(rpm * RPM_HEADROOM)), 60.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()

    def now(self) -> float:
        """The limiter's clock, so waits can be stamped on the same clock."""
        return self._clock()

    def acquire(self) -> float:
        """Block until a request may start; return the seconds waited."""
        began = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                while self._starts and self._starts[0] <= now - self.window:
                    self._starts.popleft()
                if len(self._starts) < self.limit:
                    self._starts.append(now)
                    return now - began
                delay = self._starts[0] + self.window - now
            self._sleep(delay)


def next_pacific_midnight(at: datetime) -> datetime:
    """The first midnight in America/Los_Angeles strictly after ``at``, in UTC."""
    local = at.astimezone(PACIFIC)
    tomorrow = local.date() + timedelta(days=1)
    midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=PACIFIC)
    return midnight.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class QuotaTracker:
    """Models whose daily quota is gone, with the time each one resets (S2, S5).

    One instance per process (``shared_quota_tracker``). Saved atomically to ``path`` on every
    change and reloaded on start-up, dropping expired entries. Thread-safe.

    Reset rules: ``google_genai`` models reset at the next midnight Pacific time. ``groq``
    models reset after the error's retry time when it gives one, else 24 hours after detection.
    """

    def __init__(
        self,
        path: Path,
        providers: Mapping[str, str],
        *,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = path
        self._providers = dict(providers)
        self._now = now
        self._lock = threading.RLock()
        self._reset_at: dict[str, datetime] = self._load()
        if self._drop_expired():
            self._save()

    def _load(self) -> dict[str, datetime]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_bytes())
            return {str(m): datetime.fromisoformat(t).astimezone(UTC) for m, t in data.items()}
        except (ValueError, AttributeError, TypeError) as e:
            log.warning("ignoring unreadable %s: %s", self.path, e)
            return {}

    def _save(self) -> None:
        data = {m: t.isoformat() for m, t in sorted(self._reset_at.items())}
        payload = (json.dumps(data, indent=2) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _drop_expired(self) -> bool:
        now = self._now()
        expired = [m for m, t in self._reset_at.items() if t <= now]
        for m in expired:
            del self._reset_at[m]
        return bool(expired)

    def reset_time(self, model: str, retry_after: float | None = None) -> datetime:
        now = self._now()
        if self._providers.get(model) == "google_genai":
            return next_pacific_midnight(now)
        if retry_after is not None and retry_after > 0:
            return now + timedelta(seconds=retry_after)
        return now + GROQ_DAILY_RESET

    def mark_exhausted(self, model: str, retry_after: float | None = None) -> datetime:
        with self._lock:
            reset_at = self.reset_time(model, retry_after)
            self._reset_at[model] = reset_at
            self._drop_expired()
            self._save()
            log.warning("%s daily quota exhausted until %s", model, reset_at.isoformat())
            return reset_at

    def reset_at(self, model: str) -> datetime | None:
        """When ``model`` resets, or None if it is not exhausted."""
        with self._lock:
            if self._drop_expired():
                self._save()
            return self._reset_at.get(model)

    def is_exhausted(self, model: str) -> bool:
        return self.reset_at(model) is not None

    def exhausted(self) -> dict[str, datetime]:
        with self._lock:
            if self._drop_expired():
                self._save()
            return dict(self._reset_at)


_shared_lock = threading.Lock()
_shared_tracker: QuotaTracker | None = None
_shared_limiters: dict[str, RateLimiter] = {}


def default_state_path() -> Path:
    return repo_root() / "quota_state.json"


def shared_quota_tracker(config: ModelsConfig | None = None) -> QuotaTracker:
    """The process-wide tracker (S5), created on first use at ``<repo root>/quota_state.json``."""
    global _shared_tracker
    with _shared_lock:
        if _shared_tracker is None:
            config = config or load_models_config()
            providers = {m: s.provider for m, s in config.models.items()}
            _shared_tracker = QuotaTracker(default_state_path(), providers)
        return _shared_tracker


def shared_rate_limiter(model: str, rpm: int | None) -> RateLimiter:
    """The process-wide rate limiter for ``model``."""
    with _shared_lock:
        if model not in _shared_limiters:
            _shared_limiters[model] = RateLimiter(rpm)
        return _shared_limiters[model]


# --- Per-call retries (decision 0010) ---------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Retries of one model call: at most ``max_retries``, each after
    ``min(retry_after or base_s * 2**n, cap_s)`` seconds."""

    max_retries: int
    base_s: float
    cap_s: float

    def delay(self, err: BaseException, retry_number: int) -> float:
        return min(retry_after_seconds(err) or self.base_s * 2**retry_number, self.cap_s)


NORMAL_POLICY = RetryPolicy(max_retries=2, base_s=10.0, cap_s=30.0)  # S1: then fall over
# Evaluation has no fallback (S9) and D7 removes waits from the reported time, so it waits longer.
EVALUATION_POLICY = RetryPolicy(max_retries=6, base_s=15.0, cap_s=60.0)

RETRY_ATTR = "reviewer_call_retries"


def retries_made(err: BaseException) -> int:
    """How many same-call retries ``CallRetrier`` made before it let ``err`` through."""
    return int(getattr(err, RETRY_ATTR, 0) or 0)


class CallRetrier:
    """Re-sends one failed model call to the same model (S1), inside the agent loop.

    ``call(fn)`` runs ``fn`` (one model call, including its rate-limiter wait). On a ``MINUTE`` or
    ``SERVER`` error it waits, logs the wait as ``retry_minute`` / ``retry_server`` in ``wait_log``
    and runs ``fn`` again; the agent's message history is untouched, so earlier calls are not
    repeated. When the retries run out the original error is raised (with the retry count on it,
    see ``retries_made``) for the router to handle at task level. Every other kind is raised at
    once, unchanged. ``counts`` is the number of retries made, by reason."""

    def __init__(
        self,
        policy: RetryPolicy,
        model_id: str,
        wait_log: WaitLog | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy
        self.model_id = model_id
        self.wait_log = wait_log
        self._clock = clock
        self._sleep = sleep
        self.counts: dict[str, int] = {}

    def call[T](self, fn: Callable[[], T]) -> T:
        retries = 0
        while True:
            try:
                return fn()
            except Exception as err:
                kind = classify(err)
                if kind not in (ErrorKind.MINUTE, ErrorKind.SERVER):
                    raise
                if retries >= self.policy.max_retries:
                    try:
                        setattr(err, RETRY_ATTR, retries)
                    except Exception:  # an exception that refuses attributes
                        pass
                    raise
                delay = self.policy.delay(err, retries)
                why: WaitReason = "retry_minute" if kind is ErrorKind.MINUTE else "retry_server"
                log.info(
                    "%s call failed with %s (%s); retry %d in %.1fs",
                    self.model_id,
                    kind.name,
                    str(err)[:120],
                    retries + 1,
                    delay,
                )
                self._sleep(delay)
                if self.wait_log is not None:
                    self.wait_log.record(self.model_id, why, delay, self._clock())
                self.counts[why] = self.counts.get(why, 0) + 1
                retries += 1


# --- Task 3: the router ----------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimated prompt tokens: ``ceil(len(text) / 3)``.

    A deliberate overestimate (code and English average nearer 4 characters per token), so a
    prompt the router lets through never passes a model's TPM limit by accident. It costs no API
    call and does not depend on the provider's tokenizer.
    """
    return math.ceil(len(text) / 3)


class RouterError(Exception):
    """Base class for errors the router raises instead of a provider error."""


class AllExhausted(RouterError):
    """S8: no model in the role's chain can take this task."""

    def __init__(self, role: str, reasons: Mapping[str, str]) -> None:
        self.role = role
        self.reasons = dict(reasons)
        why = "; ".join(f"{m}: {r}" for m, r in self.reasons.items()) or "empty chain"
        super().__init__(f"no usable model for role {role!r}: {why}")


class QuotaPaused(RouterError):
    """S9: the evaluation model's daily quota is gone. Save progress and resume after
    ``reset_at``."""

    def __init__(self, model: str, reset_at: datetime) -> None:
        self.model = model
        self.reset_at = reset_at
        super().__init__(
            f"evaluation model {model!r} daily quota exhausted until {reset_at.isoformat()}; "
            "save progress and resume after the reset"
        )


class PromptTooLarge(RouterError):
    """Evaluation mode: the prompt does not fit the evaluation model, and there is no fallback."""

    def __init__(self, model: str, prompt_tokens: int, detail: str) -> None:
        self.model = model
        self.prompt_tokens = prompt_tokens
        super().__init__(
            f"prompt of about {prompt_tokens} tokens is too large for evaluation model "
            f"{model!r} ({detail}); evaluation has no fallback model, so trim the context"
        )


class RetriesExhausted(RouterError):
    """Evaluation mode: per-minute or server errors outlasted every retry."""

    def __init__(self, model: str, kind: ErrorKind, retries: int) -> None:
        self.model = model
        self.kind = kind
        self.retries = retries
        super().__init__(
            f"evaluation model {model!r} still failing with {kind.name} after {retries} retries"
        )


@dataclass(frozen=True)
class RouterResult[T]:
    """A task's result and the model that produced it (S7: findings are tagged with it)."""

    value: T
    model: str


type Task[T] = Callable[[BaseChatModel, str], T]


class _FallOver(Exception):
    """Internal: give up on this model for this task with ``reason``."""

    def __init__(self, reason: str, *, too_big: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.too_big = too_big


class QuotaRouter:
    """Chooses a model for each agent task and owns retries (S1 to S9).

    Normal mode tries the role's chain from ``config/models.yaml`` left to right. Evaluation mode
    (D4, S9) always uses ``evaluation.model``, never falls back, and pauses on a daily limit.
    """

    estimate_tokens = staticmethod(estimate_tokens)

    def __init__(
        self,
        config: ModelsConfig | None = None,
        *,
        evaluation: bool = False,
        tracker: QuotaTracker | None = None,
        limiters: Mapping[str, RateLimiter] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """``tracker`` defaults to the process-wide one. ``limiters=None`` uses the process-wide
        rate limiters; a mapping gives this router its own, and missing ones are created with
        ``clock`` and ``sleep`` (tests inject fakes)."""
        self.config = config or load_models_config()
        self.evaluation = evaluation
        self.tracker = tracker or shared_quota_tracker(self.config)
        self._limiters = dict(limiters) if limiters is not None else None
        self._clock = clock
        self._sleep = sleep
        self.retry_policy = retry_policy or (EVALUATION_POLICY if evaluation else NORMAL_POLICY)

    def call_retrier(self, model_id: str, wait_log: WaitLog | None) -> CallRetrier:
        """The per-call retrier for one agent task on ``model_id`` (decision 0010)."""
        return CallRetrier(
            self.retry_policy, model_id, wait_log, clock=self._clock, sleep=self._sleep
        )

    # -- choosing a model --

    def chain(self, role: str) -> list[str]:
        if self.evaluation:
            return [self.config.evaluation.model]
        chains = self.config.chains.model_dump()
        if role not in chains:
            raise KeyError(f"unknown role {role!r}; known: {sorted(chains)}")
        return list(chains[role])

    def _fits(self, model: str, prompt_tokens: int) -> bool:
        tpm = self.config.models[model].tpm
        return tpm is None or tpm >= prompt_tokens / RPM_HEADROOM

    def _skip_reason(self, model: str, prompt_tokens: int) -> str | None:
        reset_at = self.tracker.reset_at(model)
        if reset_at is not None:
            return f"daily quota exhausted until {reset_at.isoformat()}"
        if not self._fits(model, prompt_tokens):
            tpm = self.config.models[model].tpm
            return f"prompt of about {prompt_tokens} tokens needs TPM >= " + (
                f"{math.ceil(prompt_tokens / RPM_HEADROOM)}, model has {tpm}"
            )
        return None

    def _pick(self, role: str, prompt_tokens: int, skipped: dict[str, str]) -> str:
        for model in self.chain(role):
            if model in skipped:
                continue
            reason = self._skip_reason(model, prompt_tokens)
            if reason is None:
                return model
            skipped[model] = reason
        raise AllExhausted(role, skipped)

    def pick(self, role: str, prompt_tokens: int) -> str:
        """The first model in ``role``'s chain that is not exhausted and whose TPM is null or at
        least ``prompt_tokens / 0.9``. Raises ``AllExhausted`` (S8) if none qualifies; in
        evaluation mode raises ``QuotaPaused`` or ``PromptTooLarge`` instead."""
        if self.evaluation:
            return self._pick_evaluation(prompt_tokens)
        return self._pick(role, prompt_tokens, {})

    def _pick_evaluation(self, prompt_tokens: int) -> str:
        model = self.config.evaluation.model
        reset_at = self.tracker.reset_at(model)
        if reset_at is not None:
            raise QuotaPaused(model, reset_at)
        if not self._fits(model, prompt_tokens):
            raise PromptTooLarge(model, prompt_tokens, f"TPM {self.config.models[model].tpm}")
        return model

    # -- building and running --

    def build(self, model_id: str) -> BaseChatModel:
        """The chat model with temperature 0 and the client's own retries off."""
        provider = self.config.models[model_id].provider
        return init_chat_model(model_id, model_provider=provider, temperature=0, max_retries=0)

    def limiter(self, model: str) -> RateLimiter:
        rpm = self.config.models[model].rpm
        if self._limiters is None:
            return shared_rate_limiter(model, rpm)
        if model not in self._limiters:
            self._limiters[model] = RateLimiter(rpm, clock=self._clock, sleep=self._sleep)
        return self._limiters[model]

    def run[T](self, role: str, task: Task[T], prompt_tokens: int) -> RouterResult[T]:
        """Run ``task(chat_model, model_id)``, one whole agent task, on the role's model.

        The task retries ``MINUTE`` and ``SERVER`` errors per model call itself (``call_retrier``).
        An error that reaches the router has outlasted those retries: a fallback re-runs the whole
        task from the start on the next model (S4), never a continuation. See decision 0010.
        """
        if self.evaluation:
            model = self._pick_evaluation(prompt_tokens)
            return RouterResult(self._attempt(model, task, prompt_tokens), model)
        skipped: dict[str, str] = {}
        while True:
            model = self._pick(role, prompt_tokens, skipped)
            try:
                value = self._attempt(model, task, prompt_tokens)
            except _FallOver as fo:
                skipped[model] = fo.reason
                log.warning("role %s: falling over from %s: %s", role, model, fo.reason)
                if fo.too_big:
                    self._skip_smaller(role, model, skipped)
                continue
            return RouterResult(value, model)

    def _skip_smaller(self, role: str, failed: str, skipped: dict[str, str]) -> None:
        """After TOO_BIG on ``failed``, also skip later models whose TPM is no larger."""
        tpm = self.config.models[failed].tpm
        if tpm is None:
            return
        for model in self.chain(role):
            other = self.config.models[model].tpm
            if model not in skipped and other is not None and other <= tpm:
                skipped[model] = f"TPM {other} <= {tpm} of {failed}, which rejected the prompt"

    def _attempt[T](self, model: str, task: Task[T], prompt_tokens: int) -> T:
        chat_model = self.build(model)
        waited = self.limiter(model).acquire()
        record_wait(model, "rate_limiter", waited, end=self._clock())
        try:
            return task(chat_model, model)
        except Exception as err:
            kind = classify(err)
            log.info("%s failed with %s: %s", model, kind.name, err)
            if kind is ErrorKind.FATAL:
                raise
            if kind is ErrorKind.DAILY:
                reset_at = self.tracker.mark_exhausted(model, retry_after_seconds(err))
                if self.evaluation:
                    raise QuotaPaused(model, reset_at) from err
                reason = f"DAILY quota hit, exhausted until {reset_at.isoformat()}"
                raise _FallOver(reason) from err
            if kind is ErrorKind.TOO_BIG:
                if self.evaluation:
                    raise PromptTooLarge(model, prompt_tokens, "rejected as too large") from err
                reason = "TOO_BIG: the provider rejected the prompt"
                raise _FallOver(reason, too_big=True) from err
            # MINUTE or SERVER: the per-call retries are used up.
            retries = retries_made(err)
            if self.evaluation:
                raise RetriesExhausted(model, kind, retries) from err
            reason = f"{kind.name}: still failing after {retries} retries"
            raise _FallOver(reason) from err
