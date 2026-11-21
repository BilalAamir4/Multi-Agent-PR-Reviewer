"""Shared agent machinery (stage 2A, decision 0008): one ``create_agent`` reviewer described by an
``AgentSpec`` and run by ``run_agent``.

``run_agent`` does what the stage 1 ``review()`` did, for any agent: it runs the whole agent
task through ``QuotaRouter.run`` (so a fallback re-runs it from the start, S4) inside
``review_waits()`` (D7), with ``ToolStrategy(ReviewOutput)``, the call-limit middleware (counting
only review tools), the per-call rate-limit middleware, ``to_findings`` post-processing (D11,
decision 0007) and token, call and time accounting with LangSmith tags and metadata.

Prompts are composed: a line ``<!-- include: NAME -->`` is replaced by
``prompts/common/NAME.md``, and ``<!-- focus_categories -->`` by the spec's focus categories.
A prompt without those lines (``single.md``) is used byte for byte.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.tools import BaseTool
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, ConfigDict, Field

from reviewer.agents.output import Dropped, ReviewOutput, to_findings
from reviewer.config import AgentLimits
from reviewer.models import (
    CallRetrier,
    QuotaPaused,
    QuotaRouter,
    RateLimiter,
    WaitLog,
    review_waits,
)
from reviewer.schemas import Category, Finding
from reviewer.tools import ALL_TOOLS, REVIEW_TOOLS, ReviewContext, ToolNotes, make_tools

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
COMMON_DIR = PROMPTS_DIR / "common"
RUN_TESTS = "run_tests"

# complete: finished with every tool it needed. partial (D2b): finished, but a tool it needed
# could not run (see ``notes``). limit_reached: a call limit ended it before an answer.
type Status = Literal["complete", "partial", "limit_reached", "quota_paused", "error"]


# --- The spec --------------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """One reviewer agent.

    ``name`` is the agent's name and ``Finding.agent``; ``role`` is its router chain in
    ``config/models.yaml``; ``tool_names`` is a subset of the six stage 1A tools and
    ``extra_tools`` adds others (``run_tests``); ``focus_categories`` is used only in its
    prompt."""

    name: str
    role: str
    prompt_path: Path
    tool_names: tuple[str, ...]
    limits: AgentLimits
    extra_tools: tuple[str, ...] = ()
    focus_categories: tuple[Category, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        bad = sorted(set(self.tool_names) - set(REVIEW_TOOLS))
        if bad:
            raise ValueError(f"{self.name}: tool_names must be stage 1A tools, not {bad}")
        bad = sorted(set(self.extra_tools) - (set(ALL_TOOLS) - set(REVIEW_TOOLS)))
        if bad:
            raise ValueError(f"{self.name}: unknown extra_tools {bad}")
        if RUN_TESTS in self.extra_tools and self.limits.max_run_tests_calls is None:
            raise ValueError(f"{self.name}: run_tests needs limits.max_run_tests_calls")

    @property
    def tools(self) -> tuple[str, ...]:
        """Every tool name the agent gets, in ``ALL_TOOLS`` order."""
        wanted = set(self.tool_names) | set(self.extra_tools)
        return tuple(t for t in ALL_TOOLS if t in wanted)

    @property
    def run_name(self) -> str:
        return f"{self.name}-review"

    def system_prompt(self) -> str:
        return compose_prompt(self.prompt_path, self.focus_categories)

    def prompt_version(self) -> str:
        """First 12 hex characters of the SHA-256 of the composed system prompt (UTF-8). For a
        prompt with no include lines that is the hash of the file's bytes."""
        return hashlib.sha256(self.system_prompt().encode("utf-8")).hexdigest()[:12]


# --- Prompts ---------------------------------------------------------------------------------

_INCLUDE = re.compile(r"^<!-- include: (?P<name>[a-z_]+) -->(?=\r?$)", re.MULTILINE)
FOCUS_MARK = "<!-- focus_categories -->"


@cache
def fragment(name: str) -> str:
    """``prompts/common/<name>.md`` without its final newline."""
    path = COMMON_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt fragment {name!r} ({path})")
    return path.read_bytes().decode("utf-8").removesuffix("\n")


def focus_line(categories: tuple[str, ...]) -> str:
    return "Your focus categories: " + ", ".join(f"`{c}`" for c in categories) + "."


@cache
def compose_prompt(path: Path, focus_categories: tuple[str, ...] = ()) -> str:
    text = path.read_bytes().decode("utf-8")
    text = _INCLUDE.sub(lambda m: fragment(m.group("name")), text)
    if FOCUS_MARK in text:
        if not focus_categories:
            raise ValueError(f"{path.name} has {FOCUS_MARK} but the spec has no focus categories")
        text = text.replace(FOCUS_MARK, focus_line(focus_categories))
    if "<!-- include:" in text:
        raise ValueError(f"{path.name}: an include line must be exactly <!-- include: name -->")
    return text


# --- Result ----------------------------------------------------------------------------------


class Attempt(BaseModel):
    """One run of the agent task on one model (the router may make several, S4)."""

    model_config = ConfigDict(extra="forbid")

    model: str
    langsmith_run_id: str | None = None
    model_calls: int = 0
    tool_calls: dict[str, int] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    outcome: str = "running"  # complete, limit_reached, no_structured_response, or an error


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Status
    agent: str = "single"  # AgentSpec.name; results from before stage 2A are all "single"
    notes: list[str] = Field(default_factory=list)  # for example "sandbox unavailable: ..."
    error_class: str | None = None
    error: str | None = None
    limit: Literal["max_model_calls", "max_tool_calls"] | None = None
    reset_at: datetime | None = None  # quota_paused: when the model's daily quota resets
    findings: list[Finding] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    summary: str | None = None
    model_id: str | None = None
    prompt_version: str
    limits: AgentLimits
    evaluation: bool
    item_id: str | None = None
    model_calls: int = 0
    tool_calls: int = 0
    tool_calls_by_tool: dict[str, int] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    wait_s: float = 0.0
    net_duration_s: float = 0.0
    waits_by_reason: dict[str, float] = Field(default_factory=dict)
    # Per-call retries (decision 0010), by wait reason: {"retry_server": 2}. Children's included.
    retries_by_reason: dict[str, int] = Field(default_factory=dict)
    langsmith_run_id: str | None = None
    attempts: list[Attempt] = Field(default_factory=list)


# --- Accounting ------------------------------------------------------------------------------


class _UsageCounter(BaseCallbackHandler):
    """Counts completed model calls, tool runs and ``usage_metadata`` tokens for one attempt.
    Tools may run in parallel threads, hence the lock.

    Callbacks in a run config are inherited by every child run, so an agent that calls another
    agent inside a tool would also see the child's model and tool events. ``foreign_tools`` names
    the tools whose descendants belong to another agent (the orchestrator's specialist tools):
    the tool call itself is counted, everything below it is not (decision 0009). The handler
    follows ``parent_run_id`` links to tell."""

    def __init__(self, attempt: Attempt, foreign_tools: frozenset[str] = frozenset()) -> None:
        self.attempt = attempt
        self.foreign_tools = foreign_tools
        self._lock = threading.Lock()
        self._tools: Counter[str] = Counter()
        self._parents: dict[Any, Any] = {}
        self._foreign_runs: set[Any] = set()

    def _link(self, run_id: Any, parent_run_id: Any) -> None:
        if self.foreign_tools:
            with self._lock:
                self._parents[run_id] = parent_run_id

    def _below_foreign(self, run_id: Any) -> bool:
        """True when a strict ancestor of ``run_id`` is a foreign tool run."""
        if not self._foreign_runs:
            return False
        with self._lock:
            cur = self._parents.get(run_id)
            for _ in range(200):
                if cur is None:
                    return False
                if cur in self._foreign_runs:
                    return True
                cur = self._parents.get(cur)
        return False

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs: Any):
        self._link(run_id, parent_run_id)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs: Any
    ) -> None:
        self._link(run_id, parent_run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs: Any):
        self._link(run_id, parent_run_id)

    def on_llm_end(self, response: LLMResult, *, run_id=None, **kwargs: Any) -> None:
        if self._below_foreign(run_id):
            return
        tin = tout = 0
        for gens in response.generations:
            for gen in gens:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                tin += int(usage.get("input_tokens", 0) or 0)
                tout += int(usage.get("output_tokens", 0) or 0)
        with self._lock:
            self.attempt.model_calls += 1
            self.attempt.input_tokens += tin
            self.attempt.output_tokens += tout

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id=None,
        parent_run_id=None,
        **kwargs: Any,
    ) -> None:
        self._link(run_id, parent_run_id)
        if self._below_foreign(run_id):
            return
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        with self._lock:
            if name in self.foreign_tools:
                self._foreign_runs.add(run_id)
            self._tools[name] += 1
            self.attempt.tool_calls = dict(sorted(self._tools.items()))


# --- Middleware ------------------------------------------------------------------------------


class _ModelCallLimit(ModelCallLimitMiddleware):
    """``ModelCallLimitMiddleware`` with ``exit_behavior="end"`` that remembers it ended the
    run."""

    def __init__(self, run_limit: int) -> None:
        super().__init__(run_limit=run_limit, exit_behavior="end")
        self.hit = False

    def before_model(self, state, runtime):  # type: ignore[override]
        update = super().before_model(state, runtime)
        if update and update.get("jump_to") == "end":
            self.hit = True
        return update


class _ToolCallLimit(ToolCallLimitMiddleware):
    """``ToolCallLimitMiddleware`` with ``exit_behavior="end"`` over the review tools only.

    The installed middleware counts every tool call on the AI message when ``tool_name`` is
    None, including ToolStrategy's structured-output call; that final answer must never be
    blocked or counted as a tool call, so only the named review tools are tracked."""

    def __init__(self, run_limit: int, tool_names: frozenset[str]) -> None:
        super().__init__(run_limit=run_limit, exit_behavior="end")
        self.tool_names = tool_names
        self.hit = False

    def _matches_tool_filter(self, tool_call) -> bool:  # type: ignore[override]
        return tool_call["name"] in self.tool_names

    def after_model(self, state, runtime):  # type: ignore[override]
        update = super().after_model(state, runtime)
        if update and update.get("jump_to") == "end":
            self.hit = True
        return update


class _RunTestsLimit(ToolCallLimitMiddleware):
    """At most ``run_limit`` ``run_tests`` calls per agent run, counted apart from the overall
    tool-call limit (which counts them too). Past the limit the call is refused with an error
    tool message and the review goes on (``exit_behavior="continue"``)."""

    def __init__(self, run_limit: int) -> None:
        super().__init__(tool_name=RUN_TESTS, run_limit=run_limit, exit_behavior="continue")


class _PerCallRateLimit(AgentMiddleware):
    """Per-model-call pacing and retries (decision 0010).

    Waits on the model's rate limiter before every model call after the first (the router
    already waited before starting the task), so a 25-call loop stays under the model's RPM.
    Waits go to the review's ``WaitLog`` as ``rate_limiter`` (D7).

    With a ``retrier``, a ``MINUTE`` or ``SERVER`` error is retried here, on the same model with
    the same request, so one 503 does not re-run the whole task. The rate limiter runs before every
    attempt, retries included. The retry lives in this class, not in a second middleware, so the
    frozen single agent builds the same middleware list as before (``test_single_identical``)."""

    def __init__(
        self,
        limiter: RateLimiter,
        model_id: str,
        wait_log: WaitLog | None,
        retrier: CallRetrier | None = None,
    ) -> None:
        super().__init__()
        self.limiter = limiter
        self.model_id = model_id
        self.wait_log = wait_log
        self.retrier = retrier
        self._first = True

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        def attempt():
            if self._first:
                self._first = False
            else:
                waited = self.limiter.acquire()
                if self.wait_log is not None and waited > 0:
                    self.wait_log.record(self.model_id, "rate_limiter", waited, self.limiter.now())
            return handler(request)

        return attempt() if self.retrier is None else self.retrier.call(attempt)


# --- The agent -------------------------------------------------------------------------------


@dataclass
class _Run:
    structured: ReviewOutput | None
    limit: Literal["max_model_calls", "max_tool_calls"] | None


def build_agent(
    spec: AgentSpec,
    chat_model: BaseChatModel,
    ctx: ReviewContext,
    notes: ToolNotes,
    extra_middleware: list[AgentMiddleware] | None = None,
    *,
    extra_tools: Sequence[BaseTool] = (),
    response_schema: type[BaseModel] = ReviewOutput,
) -> tuple[Any, _ModelCallLimit, _ToolCallLimit]:
    """``extra_tools`` are ready-made tools (the orchestrator's specialist tools) that count
    toward the tool-call limit; ``response_schema`` is the structured final answer."""
    tools = [*make_tools(ctx, spec.tools, notes), *extra_tools]
    limits = spec.limits
    model_limit = _ModelCallLimit(limits.max_model_calls)
    tool_limit = _ToolCallLimit(limits.max_tool_calls, frozenset(t.name for t in tools))
    middleware: list[AgentMiddleware] = [model_limit, tool_limit]
    if RUN_TESTS in spec.tools:
        assert limits.max_run_tests_calls is not None  # AgentSpec checks this
        middleware.append(_RunTestsLimit(limits.max_run_tests_calls))
    agent = create_agent(
        chat_model,
        tools=tools,
        system_prompt=spec.system_prompt(),
        response_format=ToolStrategy(response_schema),
        middleware=[*middleware, *(extra_middleware or [])],
        name=spec.name,
    )
    return agent, model_limit, tool_limit


def recursion_limit(limits: AgentLimits) -> int:
    """Graph steps for ``max_model_calls`` loops (each loop is about 5 nodes), with headroom so
    the call-limit middleware, not LangGraph's recursion limit, always ends a long run."""
    return 10 * limits.max_model_calls + 20


def retry_counts(wait_log: WaitLog) -> dict[str, int]:
    """Per-call retries recorded in ``wait_log`` by reason (``retry_minute``, ``retry_server``)."""
    counts = wait_log.counts_by_reason()
    return {k: counts[k] for k in sorted(counts) if k.startswith("retry_")}


def run_agent(
    spec: AgentSpec,
    ctx: ReviewContext,
    user_message: str,
    router: QuotaRouter,
    *,
    item_id: str | None = None,
    evaluation: bool = False,
    clock: Callable[[], float] = time.monotonic,
    parent_config: RunnableConfig | None = None,
) -> ReviewResult:
    """Run one agent on the pull request in ``ctx``. ``parent_config`` is the ``RunnableConfig``
    of the tool call that runs this agent (the orchestrator's specialist tools), so the run is a
    child of that call in LangSmith and in the callback tree. Never raises for a model, quota or
    agent failure: those become the result's ``status``. ``evaluation`` must match
    ``router.evaluation`` (a guard against labelling a normal run as an evaluation run or the
    other way round). ``clock`` measures ``duration_s`` (tests pass the fake clock their rate
    limiters use)."""
    if evaluation != router.evaluation:
        raise ValueError(
            f"evaluation={evaluation} but the router has evaluation={router.evaluation}"
        )
    limits = spec.limits
    system_prompt = spec.system_prompt()
    version = spec.prompt_version()
    prompt_tokens = router.estimate_tokens(system_prompt + user_message + ctx.diff_text)
    attempts: list[Attempt] = []
    tags = [spec.name, item_id or "adhoc"]
    notes = ToolNotes()

    def task(chat_model: BaseChatModel, model_id: str) -> _Run:
        attempt = Attempt(model=model_id)
        attempts.append(attempt)
        retrier = router.call_retrier(model_id, wait_log)
        rate = _PerCallRateLimit(router.limiter(model_id), model_id, wait_log, retrier)
        agent, model_limit, tool_limit = build_agent(spec, chat_model, ctx, notes, [rate])
        run_id = uuid.uuid4()
        if tracing_is_enabled():
            attempt.langsmith_run_id = str(run_id)
        config = {
            "callbacks": [_UsageCounter(attempt)],
            "run_name": spec.run_name,
            "run_id": run_id,
            "tags": tags,
            "metadata": {
                "prompt_version": version,
                "model_id": model_id,
                "evaluation": router.evaluation,
            },
            "recursion_limit": recursion_limit(limits),
        }
        if parent_config is not None:
            config = merge_configs(parent_config, config)
        try:
            state = agent.invoke({"messages": [HumanMessage(user_message)]}, config=config)
        except Exception as e:
            attempt.outcome = f"{type(e).__name__}: {e}"
            raise
        structured = state.get("structured_response")
        if isinstance(structured, ReviewOutput):
            attempt.outcome = "complete"
            return _Run(structured, None)
        if model_limit.hit or tool_limit.hit:
            attempt.outcome = "limit_reached"
            limit = "max_model_calls" if model_limit.hit else "max_tool_calls"
            return _Run(None, limit)
        attempt.outcome = "no_structured_response"
        return _Run(None, None)

    started = clock()
    with review_waits() as wait_log:
        status: Status
        out: dict[str, Any] = {}
        try:
            routed = router.run(spec.role, task, prompt_tokens)
        except QuotaPaused as e:
            status = "quota_paused"
            out.update(model_id=e.model, reset_at=e.reset_at)
            out.update(error_class=type(e).__name__, error=str(e))
        except Exception as e:
            log.debug("review failed", exc_info=True)
            status = "error"
            out.update(error_class=type(e).__name__, error=str(e))
        else:
            out["model_id"] = routed.model
            run = routed.value
            if run.structured is not None:
                status = "partial" if notes.unavailable else "complete"
                findings, dropped = to_findings(
                    run.structured.findings, spec.name, routed.model, ctx
                )
                out.update(findings=findings, dropped=dropped, summary=run.structured.summary)
            elif run.limit is not None:
                status = "limit_reached"
                out["limit"] = run.limit
            else:
                status = "error"
                out.update(
                    error_class="NoStructuredResponse",
                    error="the agent stopped without a structured response",
                )
        duration = clock() - started
        wait = wait_log.total()
        waits_by_reason = wait_log.by_reason()
        retries = retry_counts(wait_log)

    if "model_id" not in out and attempts:
        out["model_id"] = attempts[-1].model
    by_tool: Counter[str] = Counter()
    for a in attempts:
        by_tool.update(a.tool_calls)
    return ReviewResult(
        status=status,
        agent=spec.name,
        notes=list(notes.notes),
        prompt_version=version,
        limits=limits,
        evaluation=router.evaluation,
        item_id=item_id,
        model_calls=sum(a.model_calls for a in attempts),
        tool_calls=sum(by_tool.values()),
        tool_calls_by_tool=dict(sorted(by_tool.items())),
        input_tokens=sum(a.input_tokens for a in attempts),
        output_tokens=sum(a.output_tokens for a in attempts),
        duration_s=round(duration, 3),
        wait_s=round(wait, 3),
        net_duration_s=round(round(duration, 3) - round(wait, 3), 3),
        waits_by_reason={k: round(v, 3) for k, v in sorted(waits_by_reason.items())},
        retries_by_reason=retries,
        langsmith_run_id=attempts[-1].langsmith_run_id if attempts else None,
        attempts=attempts,
        **out,
    )
