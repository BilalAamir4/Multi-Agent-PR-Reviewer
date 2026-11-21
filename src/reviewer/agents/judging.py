"""What the verifier and the orchestrator-as-judge share (stage 4B, decision 0013).

Both judge the merged findings of a review and only **keep or drop** them: neither edits a
finding, adds one or changes its category. Both give each finding one of three verdicts
(``confirmed``, ``rejected``, ``uncertain``), and the policy is the same: drop only ``rejected``.

This module holds the verdict schemas, the stable finding id, ``run_structured`` (one
``create_agent`` run whose final answer is any schema, with the same accounting, per-call retries
and router use as ``run_agent``), the keep-or-drop policy and the ``JudgeStepResult`` both judges
return.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel, ConfigDict, Field

from reviewer.agents.base import (
    AgentSpec,
    Attempt,
    ReviewResult,
    Status,
    _PerCallRateLimit,
    _UsageCounter,
    build_agent,
    recursion_limit,
    retry_counts,
)
from reviewer.agents.output import Dropped
from reviewer.models import QuotaPaused, QuotaRouter, review_waits
from reviewer.schemas import Finding
from reviewer.tools import ReviewContext, ToolNotes

log = logging.getLogger(__name__)

type VerdictName = Literal["confirmed", "rejected", "uncertain"]
# Errors that end the whole judge step (and so the review) instead of counting as ``uncertain``.
ABORT_CLASSES = frozenset({"AllExhausted", "RetriesExhausted"})
# Tokens added to the prompt estimate for the tool outputs the checker will read (one
# ``read_file`` or ``read_diff`` output is at most ``MAX_OUTPUT_CHARS`` = 12K characters).
TOOL_HEADROOM_TOKENS = 4000


# --- Schemas ---------------------------------------------------------------------------------


class Verdict(BaseModel):
    """The verifier's answer for the one finding it checked."""

    model_config = ConfigDict(extra="forbid")

    verdict: VerdictName = Field(
        description="confirmed, rejected or uncertain, as defined in the instructions"
    )
    reason: str = Field(
        description="One or two sentences that cite what in the code supports the verdict"
    )


class JudgeVerdict(BaseModel):
    """The verdict on one finding of the list."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(description="The finding's ID, copied exactly from the list")
    verdict: VerdictName = Field(
        description="confirmed, rejected or uncertain, as defined in the instructions"
    )
    reason: str = Field(
        description="One or two sentences that cite what in the code supports the verdict"
    )


class JudgeOutput(BaseModel):
    """The lead reviewer's answer: one verdict for every finding in the list."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[JudgeVerdict] = Field(
        description="Exactly one verdict for each finding ID in the list"
    )


class VerificationRecord(BaseModel):
    """One finding's verdict. ``verdict`` is what the policy used: a check that failed or hit a
    limit is recorded as ``uncertain`` with ``source`` saying why. ``calls`` and ``tokens`` are
    this check's own for the verifier and ``None`` for the judge (one run covers every finding)."""

    model_config = ConfigDict(extra="forbid")

    finding_ref: str
    verdict: VerdictName
    reason: str
    source: Literal["agent", "limit_reached", "error", "missing"] = "agent"
    model_id: str | None = None
    calls: int | None = None
    tokens: int | None = None


class JudgeStepResult(ReviewResult):
    """The judge step of a review (the verifier's fan-out or the orchestrator-as-judge run).

    ``findings`` are the findings that were kept, ``dropped`` the rejected ones (reason
    ``verifier_rejected`` or ``judge_rejected``, ``detail`` the verdict's reason). ``children``
    are the agent runs below the step: one per finding for the verifier, the single run for the
    judge. The step makes no model call of its own, so the inherited counters are the children's
    sum, ``own_*`` stay 0, ``wait_s_sum`` / ``wait_s_wall`` are the step's nested waits and
    ``net_duration_s`` is ``duration_s - wait_s_wall``."""

    agent: str = "verifier"
    judge: Literal["verifier", "orchestrator"] = "verifier"
    input_findings: int = 0
    verifications: list[VerificationRecord] = Field(default_factory=list)


# --- Finding ids -----------------------------------------------------------------------------


def finding_ref(f: Finding) -> str:
    """A stable id: the same finding always gets the same one, in any run."""
    key = "|".join(
        [f.file, str(f.line_start), str(f.line_end), f.category, f.agent, f.title.strip()]
    )
    return "F" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def assign_refs(findings: Sequence[Finding]) -> list[str]:
    """``finding_ref`` of each finding; a repeated id gets ``-2``, ``-3`` ... so ids are unique."""
    seen: Counter[str] = Counter()
    refs: list[str] = []
    for f in findings:
        base = finding_ref(f)
        seen[base] += 1
        refs.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return refs


def span(f: Finding) -> str:
    return str(f.line_start) if f.line_start == f.line_end else f"{f.line_start}-{f.line_end}"


# --- One agent run with any structured answer -------------------------------------------------


def run_structured(
    spec: AgentSpec,
    ctx: ReviewContext,
    user_message: str,
    router: QuotaRouter,
    schema: type[BaseModel],
    *,
    item_id: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    parent_config: RunnableConfig | None = None,
    extra_tags: Sequence[str] = (),
) -> tuple[ReviewResult, BaseModel | None]:
    """Run ``spec`` once and return its ``ReviewResult`` (no findings) and the structured answer
    (``None`` unless the status is ``complete``). It follows ``run_agent`` step by step (the
    whole task through ``QuotaRouter.run``, waits nested in ``review_waits``, the call-limit and
    per-call retry middleware, the usage counter) with a different final schema. Never raises for
    a model, quota or agent failure: those become the result's ``status``."""
    evaluation = router.evaluation
    system_prompt = spec.system_prompt()
    version = spec.prompt_version()
    prompt_tokens = router.estimate_tokens(system_prompt + user_message) + TOOL_HEADROOM_TOKENS
    attempts: list[Attempt] = []
    tags = [spec.name, item_id or "adhoc", *extra_tags]
    notes = ToolNotes()

    def task(chat_model: BaseChatModel, model_id: str) -> tuple[BaseModel | None, str | None]:
        attempt = Attempt(model=model_id)
        attempts.append(attempt)
        retrier = router.call_retrier(model_id, wait_log)
        rate = _PerCallRateLimit(router.limiter(model_id), model_id, wait_log, retrier)
        agent, model_limit, tool_limit = build_agent(
            spec, chat_model, ctx, notes, [rate], response_schema=schema
        )
        run_id = uuid.uuid4()
        if tracing_is_enabled():
            attempt.langsmith_run_id = str(run_id)
        config: dict[str, Any] = {
            "callbacks": [_UsageCounter(attempt)],
            "run_name": spec.run_name,
            "run_id": run_id,
            "tags": tags,
            "metadata": {
                "prompt_version": version,
                "model_id": model_id,
                "evaluation": evaluation,
            },
            "recursion_limit": recursion_limit(spec.limits),
        }
        if parent_config is not None:
            config = merge_configs(parent_config, config)
        try:
            state = agent.invoke({"messages": [HumanMessage(user_message)]}, config=config)
        except Exception as e:
            attempt.outcome = f"{type(e).__name__}: {e}"
            raise
        structured = state.get("structured_response")
        if isinstance(structured, schema):
            attempt.outcome = "complete"
            return structured, None
        if model_limit.hit or tool_limit.hit:
            attempt.outcome = "limit_reached"
            return None, "max_model_calls" if model_limit.hit else "max_tool_calls"
        attempt.outcome = "no_structured_response"
        return None, None

    started = clock()
    answer: BaseModel | None = None
    out: dict[str, Any] = {}
    status: Status
    with review_waits() as wait_log:
        try:
            routed = router.run(spec.role, task, prompt_tokens)
        except QuotaPaused as e:
            status = "quota_paused"
            out.update(model_id=e.model, reset_at=e.reset_at)
            out.update(error_class=type(e).__name__, error=str(e))
        except Exception as e:
            log.debug("%s failed", spec.name, exc_info=True)
            status = "error"
            out.update(error_class=type(e).__name__, error=str(e))
        else:
            out["model_id"] = routed.model
            answer, limit = routed.value
            if answer is not None:
                status = "complete"
            elif limit is not None:
                status = "limit_reached"
                out["limit"] = limit
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
    result = ReviewResult(
        status=status,
        agent=spec.name,
        notes=list(notes.notes),
        prompt_version=version,
        limits=spec.limits,
        evaluation=evaluation,
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
    return result, answer


def is_abort(result: ReviewResult) -> bool:
    """A result whose end state aborts the judge step and the review (decision 0013): quota
    paused, every model exhausted, or retries exhausted."""
    return result.status == "quota_paused" or (
        result.status == "error" and result.error_class in ABORT_CLASSES
    )


# --- The keep-or-drop policy and the step result ----------------------------------------------


def uncertain_record(ref: str, child: ReviewResult | None, why: str) -> VerificationRecord:
    """The record of a check that gave no verdict (a limit, a non-infrastructure error or a
    missing id): ``uncertain``, so the finding is kept."""
    source: Literal["limit_reached", "error", "missing"] = "missing"
    if child is not None:
        source = "limit_reached" if child.status == "limit_reached" else "error"
    return VerificationRecord(
        finding_ref=ref,
        verdict="uncertain",
        reason=why,
        source=source,
        model_id=child.model_id if child else None,
        calls=child.model_calls if child else None,
        tokens=(child.input_tokens + child.output_tokens) if child else None,
    )


def apply_policy(
    findings: Sequence[Finding],
    refs: Sequence[str],
    records: dict[str, VerificationRecord],
    drop_reason: Literal["verifier_rejected", "judge_rejected"],
) -> tuple[list[Finding], list[Dropped]]:
    """Keep ``confirmed`` and ``uncertain``; drop ``rejected`` with the verdict's reason. Findings
    are returned unchanged (same ``agent`` and ``model``), in their original order."""
    kept: list[Finding] = []
    dropped: list[Dropped] = []
    for f, ref in zip(findings, refs, strict=True):
        rec = records[ref]
        if rec.verdict == "rejected":
            draft = {**f.model_dump(mode="json"), "finding_id": ref, "verdict": rec.verdict}
            dropped.append(Dropped(draft=draft, reason=drop_reason, detail=rec.reason))
        else:
            kept.append(f)
    return kept, dropped


def build_step_result(
    *,
    judge: Literal["verifier", "orchestrator"],
    agent: str,
    drop_reason: Literal["verifier_rejected", "judge_rejected"],
    findings: Sequence[Finding],
    refs: Sequence[str],
    records: dict[str, VerificationRecord],
    children: list[ReviewResult],
    notes: list[str],
    status: Status,
    abort: ReviewResult | None,
    prompt_version: str,
    limits: Any,
    evaluation: bool,
    item_id: str | None,
    duration_s: float,
    wait_sum: float,
    wait_wall: float,
    waits_by_reason: dict[str, float],
    retries: dict[str, int],
) -> JudgeStepResult:
    """Assemble a ``JudgeStepResult``: totals are the children's sum (nothing twice), and on an
    abort (``abort`` is the child that ended the review) nothing is dropped."""
    if abort is not None:
        kept, dropped = list(findings), []
        verifications: list[VerificationRecord] = []
    else:
        kept, dropped = apply_policy(findings, refs, records, drop_reason)
        verifications = [records[r] for r in refs]
    by_tool: Counter[str] = Counter()
    for c in children:
        by_tool.update(c.tool_calls_by_tool)
    extra: dict[str, Any] = {}
    if abort is not None:
        extra = {
            "model_id": abort.model_id,
            "reset_at": abort.reset_at,
            "error_class": abort.error_class,
            "error": abort.error,
        }
    else:
        model_ids = [c.model_id for c in children if c.model_id]
        extra["model_id"] = model_ids[0] if model_ids else None
    duration_s = round(duration_s, 3)
    wait_sum = round(wait_sum, 3)
    wait_wall = round(wait_wall, 3)
    return JudgeStepResult(
        status=status,
        agent=agent,
        judge=judge,
        notes=notes,
        prompt_version=prompt_version,
        limits=limits,
        evaluation=evaluation,
        item_id=item_id,
        input_findings=len(findings),
        findings=kept,
        dropped=dropped,
        verifications=verifications,
        children=children,
        model_calls=sum(c.model_calls for c in children),
        tool_calls=sum(c.tool_calls for c in children),
        tool_calls_by_tool=dict(sorted(by_tool.items())),
        input_tokens=sum(c.input_tokens for c in children),
        output_tokens=sum(c.output_tokens for c in children),
        duration_s=duration_s,
        wait_s=wait_sum,
        wait_s_sum=wait_sum,
        wait_s_wall=wait_wall,
        net_duration_s=round(duration_s - wait_wall, 3),
        waits_by_reason={k: round(v, 3) for k, v in sorted(waits_by_reason.items())},
        retries_by_reason=retries,
        **extra,
    )


def claim_text(f: Finding, ref: str | None = None) -> str:
    """A finding as text for a judge's user message (an untrusted claim)."""
    lines = [f"id: {ref}"] if ref else []
    lines += [
        f"file: {f.file}",
        f"lines: {span(f)}",
        f"category: {f.category}",
        f"severity: {f.severity}",
        f"confidence: {f.confidence:.2f}",
        f"title: {f.title.strip()}",
        f"explanation: {f.explanation.strip()}",
        "evidence:",
        f.evidence.rstrip(),
    ]
    return "\n".join(lines)
