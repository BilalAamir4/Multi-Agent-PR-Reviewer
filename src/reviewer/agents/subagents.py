"""A specialist's sub-agents (stage 4A, decision 0012; D12 as changed there).

When a specialist's own share of the change is large, code splits its files into batches and the
specialist spawns one sub-agent per batch. This module holds

- the spawn decision (``should_spawn_for``) and the batching (``plan_batches``): deterministic
  code, no model involved;
- the sub-agent and coordination specs: the specialist's prompt file unchanged plus one fragment
  at the end (``prompts/common/subagent_assignment.md``, ``prompts/common/coordination.md``);
- the fan-out: a small LangGraph ``StateGraph`` that sends one ``Send("subagent", ...)`` per batch,
  collects the results, and is invoked with ``max_concurrency`` (LangGraph runs the ``Send`` tasks
  of one step in a thread pool of that size, each in a copy of the caller's ``contextvars``);
- ``run_with_subagents``: fan-out, then the specialist's own cross-file pass, then the merge.

``QuotaPaused``, ``AllExhausted`` and ``RetriesExhausted`` in any sub-agent end the specialist's
review (and so the whole review); any other failure stays that child's status.
"""

from __future__ import annotations

import operator
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import merge_configs
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from reviewer.agents.base import (
    AgentSpec,
    BatchInfo,
    CrossFileInfo,
    FanOutInfo,
    ReviewResult,
    SpawnInfo,
    Status,
    SubagentTiming,
    fragment,
    retry_counts,
    run_agent,
)
from reviewer.agents.output import Dropped
from reviewer.agents.reports import finding_id, format_specialist_output, merge_candidates
from reviewer.config import AgentLimits, ReviewConfig, SpawnConfig, repo_root
from reviewer.models import QuotaRouter, attach_waits, estimate_tokens, review_waits
from reviewer.schemas import Finding
from reviewer.tools import ReviewContext, ToolError, numbered_diff
from reviewer.tools.files import _read_text, resolve_repo_path

SUBAGENT_ROLE = "worker"  # the router chain in config/models.yaml (D4 pins one model in evaluation)
SUBAGENT_LIMITS = "subagent"  # limits.subagent in config/review.yaml
# Errors in a sub-agent that end the whole review instead of making its specialist ``partial``.
ABORT_CLASSES = frozenset({"AllExhausted", "RetriesExhausted"})


# --- The spawn decision ----------------------------------------------------------------------


def _is_test_path(path: str) -> bool:
    """The test-path rule of ``eval/find_candidates.py`` (reused, not copied). ``eval`` is a
    top-level package beside ``src``, so the repository root is put on ``sys.path`` when needed
    (as ``reviewer.cli`` does)."""
    try:
        from eval.find_candidates import is_test_path
    except ModuleNotFoundError:
        root = str(repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        from eval.find_candidates import is_test_path
    return is_test_path(path)


def specialist_area(name: str, ctx: ReviewContext) -> list[str]:
    """The files specialist ``name`` reviews: the changed, non-deleted ``.py`` files, in path
    order. The same for all three specialists today; it is per specialist so it can differ
    later. Non-Python files never get a sub-agent (D12)."""
    return sorted(f.path for f in ctx.files if f.is_python and f.status != "deleted")


def _content(ctx: ReviewContext, path: str) -> str:
    try:
        clean, target = resolve_repo_path(ctx, path)
        return _read_text(target, clean)
    except (ToolError, OSError):
        return ""


def _diff_and_content(ctx: ReviewContext, path: str) -> str:
    try:
        diff = numbered_diff(ctx, path)
    except ToolError:
        diff = ""
    return diff + "\n" + _content(ctx, path)


@dataclass(frozen=True)
class SpawnDecision:
    """D12 applied to one specialist's area (decision 0012)."""

    spawn: bool
    triggers: tuple[str, ...]  # "files" and/or "tokens"
    area: tuple[str, ...]
    py_files: int
    estimated_tokens: int
    min_py_files: int
    max_specialist_prompt_tokens: int


def should_spawn_for(name: str, ctx: ReviewContext, cfg: SpawnConfig) -> SpawnDecision:
    """Spawn when the area has ``min_py_files`` or more files, or when the estimated tokens of
    the area's numbered diffs plus their full contents exceed ``max_specialist_prompt_tokens``.
    Does not look at ``cfg.enabled`` (the D6 reuse rule asks what *would* spawn)."""
    area = specialist_area(name, ctx)
    tokens = estimate_tokens("\n".join(_diff_and_content(ctx, p) for p in area)) if area else 0
    triggers = []
    if len(area) >= cfg.min_py_files:
        triggers.append("files")
    if tokens > cfg.max_specialist_prompt_tokens:
        triggers.append("tokens")
    return SpawnDecision(
        spawn=bool(triggers),
        triggers=tuple(triggers),
        area=tuple(area),
        py_files=len(area),
        estimated_tokens=tokens,
        min_py_files=cfg.min_py_files,
        max_specialist_prompt_tokens=cfg.max_specialist_prompt_tokens,
    )


# --- Batching --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    index: int  # 1-based, in packing order
    files: tuple[str, ...]
    estimated_tokens: int
    over_budget: bool = False


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[Batch, ...]
    file_limit_used: int
    budget_exceeded: bool  # more than max_subagents batches at the configured file limit
    token_budget_ignored: bool


def _groups(area: list[str]) -> list[list[str]]:
    """Related groups: a test file with the one source file of the area whose stem matches
    (``test_foo.py`` or ``foo_test.py`` with ``foo.py``, only when exactly one such file
    exists); every other file with the files of its parent directory. Sorted by ``(directory,
    first path)``; files inside a group in path order."""
    tests = [p for p in area if _is_test_path(p)]
    sources = [p for p in area if not _is_test_path(p)]
    by_stem: dict[str, list[str]] = {}
    for s in sources:
        by_stem.setdefault(PurePosixPath(s).stem, []).append(s)
    paired: dict[str, list[str]] = {}  # source path -> its tests
    unpaired_tests: list[str] = []
    for t in tests:
        stem = PurePosixPath(t).stem
        if stem.startswith("test_"):
            base = stem.removeprefix("test_")
        elif stem.endswith("_test"):
            base = stem.removesuffix("_test")
        else:
            base = ""
        match = by_stem.get(base, []) if base else []
        if len(match) == 1:
            paired.setdefault(match[0], []).append(t)
        else:
            unpaired_tests.append(t)
    groups: list[tuple[str, list[str]]] = []  # (directory, files)
    in_pair: set[str] = set()
    for src, its_tests in paired.items():
        files = sorted([src, *its_tests])
        in_pair.update(files)
        groups.append((str(PurePosixPath(src).parent), files))
    by_dir: dict[str, list[str]] = {}
    for p in [*sources, *unpaired_tests]:
        if p not in in_pair:
            by_dir.setdefault(str(PurePosixPath(p).parent), []).append(p)
    groups += [(d, sorted(files)) for d, files in by_dir.items()]
    groups.sort(key=lambda g: (g[0], g[1][0]))
    return [files for _, files in groups]


def _chunks(files: list[str], limit: int, budget: float, tokens: dict[str, int]) -> list[list[str]]:
    """``files`` (path order) cut into runs of at most ``limit`` files and ``budget`` tokens; a
    file over the budget on its own is a run of its own."""
    runs: list[list[str]] = []
    cur: list[str] = []
    used = 0
    for f in files:
        if cur and (len(cur) + 1 > limit or used + tokens[f] > budget):
            runs.append(cur)
            cur, used = [], 0
        cur.append(f)
        used += tokens[f]
    if cur:
        runs.append(cur)
    return runs


def _pack(
    groups: list[list[str]], limit: int, budget: float, tokens: dict[str, int]
) -> list[list[str]]:
    """Greedy packing in group order. A group that fits in the batch being filled joins it;
    otherwise that batch is closed. A group larger than the file limit or the token budget is cut
    into runs in path order first (``_chunks``); each run is packed the same way."""
    units: list[list[str]] = []
    for g in groups:
        fits = len(g) <= limit and sum(tokens[f] for f in g) <= budget
        units += [g] if fits else _chunks(g, limit, budget, tokens)
    batches: list[list[str]] = []
    cur: list[str] = []
    used = 0
    for u in units:
        t = sum(tokens[f] for f in u)
        if cur and len(cur) + len(u) <= limit and used + t <= budget:
            cur += u
            used += t
            continue
        if cur:
            batches.append(cur)
        cur, used = list(u), t
    if cur:
        batches.append(cur)
    return batches


def plan_batches(area: list[str], ctx: ReviewContext, cfg: SpawnConfig) -> BatchPlan:
    """Batches of the area's files for the sub-agents (decision 0012).

    1. Related groups (``_groups``). 2. Packing (``_pack``): groups in ``(directory, first path)``
    order are filled into batches of at most ``max_files_per_subagent`` files and
    ``batch_token_budget`` estimated tokens (numbered diff + file contents). 3. Cap: with more than
    ``max_subagents`` batches the file limit is raised by 1 and the files are packed again until
    the count fits (``budget_exceeded`` records that this happened). Should the file limit reach
    the number of files with the count still over the cap (only the token budget can cause that),
    the token budget is ignored (``token_budget_ignored``). 4. Batches are numbered from 1 in
    packing order; the result depends only on the input."""
    if not area:
        return BatchPlan((), cfg.max_files_per_subagent, False, False)
    tokens = {p: estimate_tokens(_diff_and_content(ctx, p)) for p in area}
    groups = _groups(sorted(area))
    limit = cfg.max_files_per_subagent
    budget: float = cfg.batch_token_budget
    packed = _pack(groups, limit, budget, tokens)
    exceeded = len(packed) > cfg.max_subagents
    ignored = False
    while len(packed) > cfg.max_subagents:
        if limit >= len(area):
            budget, ignored = float("inf"), True
            packed = _pack(groups, limit, budget, tokens)
            break
        limit += 1
        packed = _pack(groups, limit, budget, tokens)
    batches = tuple(
        Batch(
            index=i,
            files=tuple(files),
            estimated_tokens=sum(tokens[f] for f in files),
            over_budget=sum(tokens[f] for f in files) > cfg.batch_token_budget,
        )
        for i, files in enumerate(packed, 1)
    )
    return BatchPlan(batches, limit, exceeded, ignored)


def make_batches(area: list[str], ctx: ReviewContext, cfg: SpawnConfig) -> list[Batch]:
    """``plan_batches(...).batches`` as a list."""
    return list(plan_batches(area, ctx, cfg).batches)


# --- Specs -----------------------------------------------------------------------------------


def subagent_name(parent: str, index: int) -> str:
    return f"{parent}/sub{index}"


def subagent_appendix(parent: str, files: tuple[str, ...] | list[str]) -> str:
    text = fragment("subagent_assignment")
    text = text.replace("{specialist}", f"`{parent}`")
    text = text.replace("{assigned_files}", "\n".join(f"- `{f}`" for f in files))
    return "\n\n" + text


def subagent_spec(parent: AgentSpec, batch: Batch, limits: AgentLimits) -> AgentSpec:
    """Sub-agent ``batch.index`` of ``parent``: the parent's prompt file unchanged plus the
    assignment fragment, the parent's tools, router role ``worker`` and ``limits.subagent``. Its
    name is ``<parent>/sub<k>`` and becomes ``Finding.agent``."""
    return replace(
        parent,
        name=subagent_name(parent.name, batch.index),
        role=SUBAGENT_ROLE,
        limits=limits,
        appendix=subagent_appendix(parent.name, batch.files),
    )


def coordination_spec(parent: AgentSpec) -> AgentSpec:
    """The specialist's own pass after its sub-agents: its spec plus the coordination fragment."""
    return replace(parent, appendix="\n\n" + fragment("coordination"), run_suffix="cross-file")


def cross_file_message(base_message: str, children: list[ReviewResult]) -> str:
    """The specialist's user message plus its sub-agents' findings (the orchestrator's compact
    format) and their summaries."""
    parts = [base_message, "", "<subagent_reports>"]
    for c in children:
        parts += [format_specialist_output(c), f"summary: {c.summary or '(none)'}", ""]
    parts.append("</subagent_reports>")
    return "\n".join(parts)


# --- The fan-out -----------------------------------------------------------------------------


class SubagentAbort(Exception):
    """A sub-agent hit a router condition that ends the review (quota paused, every model
    exhausted, retries exhausted). Raised in the fan-out node so the graph stops."""

    def __init__(self, result: ReviewResult) -> None:
        self.result = result
        super().__init__(
            f"{result.agent} ended the review: {result.status} "
            f"({result.error_class}: {result.error})"
        )


def is_abort(result: ReviewResult) -> bool:
    return result.status == "quota_paused" or (
        result.status == "error" and result.error_class in ABORT_CLASSES
    )


class _FanState(TypedDict):
    batches: list[dict[str, Any]]
    results: Annotated[list[ReviewResult], operator.add]


class _Work(TypedDict):
    batch: dict[str, Any]


def drop_unassigned(result: ReviewResult, files: tuple[str, ...]) -> ReviewResult:
    """Move the findings outside ``files`` to ``dropped`` with the reason ``not_assigned``."""
    keep: list[Finding] = []
    dropped: list[Dropped] = []
    for f in result.findings:
        if f.file in files:
            keep.append(f)
            continue
        detail = f"{f.file!r} is not one of the assigned files: {', '.join(files)}"
        dropped.append(
            Dropped(draft=f.model_dump(mode="json"), reason="not_assigned", detail=detail)
        )
    if not dropped:
        return result
    return result.model_copy(update={"findings": keep, "dropped": [*result.dropped, *dropped]})


def run_fanout(
    parent: AgentSpec,
    ctx: ReviewContext,
    user_message: str,
    router: QuotaRouter,
    batches: list[Batch],
    cfg: ReviewConfig,
    limits: AgentLimits,
    *,
    item_id: str | None,
    evaluation: bool,
    clock: Callable[[], float],
    config: RunnableConfig,
) -> tuple[list[ReviewResult], FanOutInfo, SubagentAbort | None]:
    """Run one sub-agent per batch through a ``StateGraph`` (START sends one ``Send`` per batch to
    the ``subagent`` node, then ``collect``, then END), at most ``spawn.max_concurrency`` at a
    time. Returns the results in batch order, the fan-out's timing and, when a sub-agent ended the
    review, that abort."""
    lock = threading.Lock()
    done: dict[int, tuple[ReviewResult, float, float]] = {}  # also kept when the graph aborts
    by_index = {b.index: b for b in batches}
    abort: SubagentAbort | None = None
    t0 = clock()

    with review_waits() as fan_log:

        def subagent(work: _Work, config: RunnableConfig) -> dict[str, Any]:
            batch = by_index[work["batch"]["index"]]
            spec = subagent_spec(parent, batch, limits)
            child_config = {k: v for k, v in config.items() if k != "max_concurrency"}
            started = clock() - t0
            # The graph's thread pool copies the caller's context, which already holds the
            # fan-out's log; attach_waits makes the nesting certain for any thread.
            with attach_waits(fan_log):
                result = run_agent(
                    spec,
                    ctx,
                    user_message,
                    router,
                    item_id=item_id,
                    evaluation=evaluation,
                    clock=clock,
                    parent_config=child_config,
                )
            result = drop_unassigned(result, batch.files)
            with lock:
                done[batch.index] = (result, started, clock() - t0)
            if is_abort(result):
                raise SubagentAbort(result)
            return {"results": [result]}

        def fan_out(state: _FanState) -> list[Send]:
            return [Send("subagent", {"batch": b}) for b in state["batches"]]

        def collect(state: _FanState) -> dict[str, Any]:
            return {}

        builder = StateGraph(_FanState)
        builder.add_node("subagent", subagent, input_schema=_Work)
        builder.add_node("collect", collect)
        builder.add_conditional_edges(START, fan_out, ["subagent"])
        builder.add_edge("subagent", "collect")
        builder.add_edge("collect", END)
        graph = builder.compile()
        run_config = merge_configs(
            config,
            {
                "max_concurrency": cfg.spawn.max_concurrency,
                "run_name": f"{parent.name}-subagents",
                "recursion_limit": 10 + 4 * len(batches),
            },
        )
        try:
            graph.invoke(
                {"batches": [{"index": b.index} for b in batches], "results": []},
                config=run_config,
            )
        except SubagentAbort as e:
            abort = e
        wall = clock() - t0
        wait_wall = fan_log.wall()
        wait_sum = fan_log.total()
    order = sorted(done)
    info = FanOutInfo(
        wall_s=round(wall, 3),
        wait_s_wall=round(wait_wall, 3),
        wait_s_sum=round(wait_sum, 3),
        net_s=round(round(wall, 3) - round(wait_wall, 3), 3),
        timings=[
            SubagentTiming(
                agent=done[i][0].agent,
                start_s=round(done[i][1], 3),
                end_s=round(done[i][2], 3),
                status=done[i][0].status,
            )
            for i in order
        ],
    )
    return [done[i][0] for i in order], info, abort


# --- The specialist's flow when it spawns -----------------------------------------------------


def _status_of(
    cross: ReviewResult, children: list[ReviewResult]
) -> tuple[Status, list[str], dict[str, Any]]:
    """The spawning specialist's status, its notes and the fields to carry over from its own
    pass (decision 0012, after the 2B rules of decision 0009): its own pass failing decides the
    status, children that did not end ``complete`` make a ``complete`` review ``partial``."""
    notes = list(cross.notes)
    extra: dict[str, Any] = {}
    if cross.status in ("limit_reached", "error"):
        extra = {k: getattr(cross, k) for k in ("limit", "error_class", "error")}
        notes.append(f"the cross-file pass ended {cross.status}; kept the sub-agents' findings")
        return cross.status, notes, extra
    bad = [c for c in children if c.status != "complete"]
    for c in bad:
        notes.append(f"{c.agent} ended {c.status}; its findings are kept")
    if cross.status == "partial" or bad:
        return "partial", notes, extra
    return "complete", notes, extra


def run_with_subagents(
    parent: AgentSpec,
    ctx: ReviewContext,
    base_message: str,
    router: QuotaRouter,
    cfg: ReviewConfig,
    decision: SpawnDecision,
    plan: BatchPlan,
    *,
    item_id: str | None = None,
    evaluation: bool = False,
    clock: Callable[[], float] = time.monotonic,
    parent_config: RunnableConfig | None = None,
) -> ReviewResult:
    """Run specialist ``parent`` with sub-agents: the fan-out, then its own cross-file pass, then
    the merge (decision 0012). Never raises for a model, quota or agent failure."""
    limits = cfg.limits.for_agent(SUBAGENT_LIMITS)
    batches = list(plan.batches)
    coord = coordination_spec(parent)
    version = coord.prompt_version()
    info = SpawnInfo(
        enabled=True,
        spawned=True,
        triggers=list(decision.triggers),
        py_files=decision.py_files,
        estimated_tokens=decision.estimated_tokens,
        min_py_files=decision.min_py_files,
        max_specialist_prompt_tokens=decision.max_specialist_prompt_tokens,
        area=list(decision.area),
        batches=[
            BatchInfo(
                index=b.index,
                agent=subagent_name(parent.name, b.index),
                files=list(b.files),
                estimated_tokens=b.estimated_tokens,
                over_budget=b.over_budget,
            )
            for b in batches
        ],
        file_limit_used=plan.file_limit_used,
        budget_exceeded=plan.budget_exceeded,
        token_budget_ignored=plan.token_budget_ignored,
        max_concurrency=cfg.spawn.max_concurrency,
    )
    holder: dict[str, ReviewResult] = {}

    def body(_: Any, config: RunnableConfig) -> None:
        started = clock()
        with review_waits() as spawn_log:
            children, fan, abort = run_fanout(
                parent, ctx, base_message, router, batches, cfg, limits,
                item_id=item_id, evaluation=evaluation, clock=clock, config=config,
            )  # fmt: skip
            info.fan_out = fan
            cross: ReviewResult | None = None
            if abort is None:
                cross = run_agent(
                    coord,
                    ctx,
                    cross_file_message(base_message, children),
                    router,
                    item_id=item_id,
                    evaluation=evaluation,
                    clock=clock,
                    parent_config=config,
                )
                info.cross_file = CrossFileInfo(
                    status=cross.status,
                    duration_s=cross.duration_s,
                    wait_s=cross.wait_s,
                    net_duration_s=cross.net_duration_s,
                    model_calls=cross.model_calls,
                )
            duration = clock() - started
            wait_sum = spawn_log.total()
            wait_wall = spawn_log.wall()
            by_reason = spawn_log.by_reason()
            retries = retry_counts(spawn_log)
        holder["result"] = _assemble(
            parent, coord, version, info, children, cross, abort, router, item_id,
            duration, wait_sum, wait_wall, by_reason, retries,
        )  # fmt: skip

    # A named run around the fan-out and the cross-file pass: the sub-agents' runs and the
    # specialist's own pass are its children in LangSmith, and its tags (the specialist's name)
    # are inherited by every run below it, so the orchestrator's counter treats them as foreign
    # (decision 0011's tag rule, extended to sub-agents).
    run_config = merge_configs(
        parent_config or {},
        {
            "run_name": f"{parent.name}-review",
            "tags": [parent.name, item_id or "adhoc"],
            "metadata": {
                "prompt_version": version,
                "evaluation": evaluation,
                "spawned": True,
                "subagents": len(batches),
            },
        },
    )
    RunnableLambda(body, name=f"{parent.name}-review").invoke(None, config=run_config)
    return holder["result"]


def _assemble(
    parent: AgentSpec,
    coord: AgentSpec,
    version: str,
    info: SpawnInfo,
    children: list[ReviewResult],
    cross: ReviewResult | None,
    abort: SubagentAbort | None,
    router: QuotaRouter,
    item_id: str | None,
    duration: float,
    wait_sum: float,
    wait_wall: float,
    by_reason: dict[str, float],
    retries: dict[str, int],
) -> ReviewResult:
    """The specialist's ``ReviewResult``: its sub-agents' findings plus its own, merged; totals
    that count each agent once; waits from the spawn's own log."""
    out: dict[str, Any] = {}
    status: Status
    if abort is not None:
        r = abort.result
        status = r.status
        out.update(
            model_id=r.model_id, reset_at=r.reset_at, error_class=r.error_class, error=r.error
        )
        notes = [f"{r.agent} ended the review: {abort}"]
    else:
        assert cross is not None
        status, notes, extra = _status_of(cross, children)
        out.update(extra)
        out["model_id"] = cross.model_id or next((c.model_id for c in children if c.model_id), None)
        out["summary"] = cross.summary
    # Merge (code): the sub-agents in order, then the specialist's own findings; ties go to the
    # earlier candidate.
    parts = [*children, *([cross] if cross else [])]
    cands: list[tuple[str, Finding]] = []
    for c in parts:
        cands += [(finding_id(c.agent, i), f) for i, f in enumerate(c.findings, 1)]
    findings, merge_dropped = merge_candidates(cands)
    tagged = [
        Dropped(draft=d.draft, reason=d.reason, detail=f"[{c.agent}] {d.detail}")
        for c in parts
        for d in c.dropped
    ]
    by_tool: Counter[str] = Counter()
    for c in parts:
        by_tool.update(c.tool_calls_by_tool)
    duration = round(duration, 3)
    wait_sum, wait_wall = round(wait_sum, 3), round(wait_wall, 3)
    return ReviewResult(
        status=status,
        agent=parent.name,
        notes=notes,
        findings=findings,
        dropped=[*tagged, *merge_dropped],
        prompt_version=version,
        limits=coord.limits,
        evaluation=router.evaluation,
        item_id=item_id,
        model_calls=sum(c.model_calls for c in parts),
        tool_calls=sum(c.tool_calls for c in parts),
        tool_calls_by_tool=dict(sorted(by_tool.items())),
        input_tokens=sum(c.input_tokens for c in parts),
        output_tokens=sum(c.output_tokens for c in parts),
        own_model_calls=cross.model_calls if cross else 0,
        own_tool_calls=cross.tool_calls if cross else 0,
        own_tool_calls_by_tool=dict(cross.tool_calls_by_tool) if cross else {},
        own_input_tokens=cross.input_tokens if cross else 0,
        own_output_tokens=cross.output_tokens if cross else 0,
        duration_s=duration,
        wait_s=wait_sum,
        wait_s_sum=wait_sum,
        wait_s_wall=wait_wall,
        net_duration_s=round(duration - wait_wall, 3),
        waits_by_reason={k: round(v, 3) for k, v in sorted(by_reason.items())},
        retries_by_reason=retries,
        langsmith_run_id=cross.langsmith_run_id if cross else None,
        attempts=list(cross.attempts) if cross else [],
        children=children,
        spawn=info,
        **out,
    )
