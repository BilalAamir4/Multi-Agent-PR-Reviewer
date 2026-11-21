"""The review graph (stage 4B, decision 0013): ``START -> review -> judge -> present -> END``.

    review   runs the orchestrator exactly as ``--agent orchestrator`` does (plan, specialists as
             tools, spawning, D11 and the merge) and stores its ``OrchestratorResult``.
    judge    keeps or drops the merged findings: the verifier (one check per finding), the
             orchestrator as judge (one run for all findings) or nothing (``none``: the findings
             pass through unchanged). Neither judge edits a finding.
    present  code only: sorts the findings (severity, then confidence, then file and line) and
             builds ``approval_payload``, one entry per finding with a stable id and the judge's
             verdict and reason.

Every node reads the state and returns only its own keys, so a checkpointer can save the state
after any node and a resumed run goes on from there; recomputing a node gives the same update
(``judge`` rebuilds its notes from ``review_result``, not from ``notes``). Everything in the state
is JSON-serializable. The runtime pieces that are not state (router, agent functions, clock) live
in ``GraphRuntime``.

The whole graph is one LangSmith root run named ``review-graph``; the orchestrator, the verifier's
fan-out and the judge are nested in it through the node's ``RunnableConfig``.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from reviewer.agents.judge import run_judge
from reviewer.agents.judging import JudgeStepResult, assign_refs
from reviewer.agents.orchestrator import OrchestratorResult, orchestrate
from reviewer.agents.verifier import run_verifier
from reviewer.config import ReviewConfig, load_review_config
from reviewer.models import QuotaRouter
from reviewer.schemas import Finding, ReviewState
from reviewer.tools import ReviewContext

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
ABORTED = ("quota_paused", "error")  # review statuses after which the judge step is skipped


def make_review_key(
    diff_text: str,
    *,
    repo: str | None = None,
    pr: str | None = None,
    head_sha: str | None = None,
) -> str:
    """``<repo or "local">#<pr or item_id>@<head sha or diff sha256[:12]>``. Stage 6 uses it as
    the graph's ``thread_id`` (one review per pull request and commit, so a webhook delivered
    twice resumes the same thread instead of starting another)."""
    sha = head_sha or hashlib.sha256(diff_text.encode("utf-8")).hexdigest()[:12]
    return f"{repo or 'local'}#{pr or 'adhoc'}@{sha}"


@dataclass
class GraphRuntime:
    """What the nodes use besides the state. The defaults are the real agents; tests replace the
    agent functions. ``config`` replaces ``config/review.yaml`` for the run (tests)."""

    router_factory: Callable[[bool], QuotaRouter] = lambda evaluation: QuotaRouter(
        evaluation=evaluation
    )
    orchestrate: Callable[..., OrchestratorResult] = orchestrate
    verify: Callable[..., JudgeStepResult] = run_verifier
    judge: Callable[..., JudgeStepResult] = run_judge
    clock: Callable[[], float] = time.monotonic
    config: ReviewConfig | None = None
    _routers: dict[bool, QuotaRouter] = field(default_factory=dict)

    def router(self, evaluation: bool) -> QuotaRouter:
        """One router per mode for the whole run: its rate limiters and quota tracker are shared
        by every node."""
        if evaluation not in self._routers:
            self._routers[evaluation] = self.router_factory(evaluation)
        return self._routers[evaluation]


# --- Helpers ---------------------------------------------------------------------------------


def _context(state: ReviewState) -> ReviewContext:
    sandbox = None
    if state.get("sandbox_repo"):
        from reviewer.cli import sandbox_for_repo  # imported here: cli imports this module

        sandbox = sandbox_for_repo(state["sandbox_repo"])
    return ReviewContext.from_paths(Path(state["workspace"]), Path(state["diff_path"]), sandbox)


def _review_config(state: ReviewState, runtime: GraphRuntime) -> ReviewConfig:
    config = runtime.config or load_review_config()
    if state.get("security_mode"):
        config = config.with_security_mode(state["security_mode"])
    return config.with_spawn(enabled=state["spawn_enabled"])


def sort_findings(findings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Severity (high first), then confidence (highest first), then file and line."""
    return sorted(
        findings,
        key=lambda f: (
            SEVERITY_ORDER[f["severity"]],
            -f["confidence"],
            f["file"],
            f["line_start"],
            f["line_end"],
        ),
    )


# --- The nodes -------------------------------------------------------------------------------


def make_nodes(runtime: GraphRuntime) -> dict[str, Callable[..., dict[str, Any]]]:
    def review(state: ReviewState, config: RunnableConfig) -> dict[str, Any]:
        evaluation = state["evaluation"]
        result = runtime.orchestrate(
            _context(state),
            state["pr_title"],
            state["pr_body"],
            runtime.router(evaluation),
            clock=runtime.clock,
            config=_review_config(state, runtime),
            parent_config=config,
        )
        return {
            "review_result": result.model_dump(mode="json"),
            "findings": [f.model_dump(mode="json") for f in result.findings],
            "status": result.status,
            "notes": list(result.notes),
        }

    def judge(state: ReviewState, config: RunnableConfig) -> dict[str, Any]:
        review_result = state["review_result"]
        base_notes = list(review_result.get("notes", []))
        review_status = review_result["status"]
        kind = state["judge"]
        if kind == "none" or review_status in ABORTED:
            if kind != "none":
                base_notes.append(f"judge skipped: the review ended {review_status}")
            return {
                "judge_result": None,
                "final_findings": list(state["findings"]),
                "status": review_status,
                "notes": base_notes,
            }
        findings = [Finding.model_validate(d) for d in state["findings"]]
        ctx = _context(state)
        router = runtime.router(state["evaluation"])
        rc = _review_config(state, runtime)
        if kind == "verifier":
            step = runtime.verify(
                ctx,
                state["pr_title"],
                findings,
                router,
                config=rc,
                clock=runtime.clock,
                parent_config=config,
            )
        else:
            step = runtime.judge(
                ctx,
                review_result.get("plan", []),
                review_result.get("summary"),
                findings,
                router,
                config=rc,
                clock=runtime.clock,
                parent_config=config,
            )
        notes = [*base_notes, *step.notes]
        status = review_status
        if step.status in ABORTED:  # infrastructure: the review aborts and is re-run
            status = step.status
            notes.append(f"{step.judge} step ended the review: {step.status}")
        elif step.status != "complete":  # anything else: the findings are kept, with a note
            if status == "complete":
                status = "partial"
            notes.append(f"{step.judge} step ended {step.status}; findings it could not judge kept")
        return {
            "judge_result": step.model_dump(mode="json"),
            "final_findings": [f.model_dump(mode="json") for f in step.findings],
            "status": status,
            "notes": notes,
        }

    def present(state: ReviewState) -> dict[str, Any]:
        final = sort_findings(state["final_findings"])
        step = state.get("judge_result")
        verdicts = {v["finding_ref"]: v for v in (step or {}).get("verifications", [])}
        by_key = {
            (f["file"], f["line_start"], f["line_end"], f["category"], f["agent"], f["title"]): ref
            for f, ref in zip(
                state["findings"],
                assign_refs([Finding.model_validate(d) for d in state["findings"]]),
                strict=True,
            )
        }
        payload = []
        for f in final:
            ref = by_key[(f["file"], f["line_start"], f["line_end"], f["category"],
                          f["agent"], f["title"])]  # fmt: skip
            v = verdicts.get(ref)
            payload.append(
                {
                    "id": ref,
                    "file": f["file"],
                    "line_start": f["line_start"],
                    "line_end": f["line_end"],
                    "category": f["category"],
                    "severity": f["severity"],
                    "confidence": f["confidence"],
                    "title": f["title"],
                    "explanation": f["explanation"],
                    "evidence": f["evidence"],
                    "agent": f["agent"],
                    "verdict": v["verdict"] if v else "unchecked",
                    "verdict_reason": v["reason"] if v else None,
                }
            )
        return {"final_findings": final, "approval_payload": payload}

    return {"review": review, "judge": judge, "present": present}


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None, runtime: GraphRuntime | None = None
):
    """Compile ``START -> review -> judge -> present -> END``.

    Stage 6 (not built yet): a person approves before anything is posted. That is one more node
    after ``present`` (``present -> approve -> post -> END``) whose body calls
    ``langgraph.types.interrupt(state["approval_payload"])``, and the compile call below gets the
    checkpointer (the outermost graph only), with ``review_key`` as the ``thread_id``. No node
    above has to change: each already returns only its own keys and can resume after a
    checkpoint."""
    nodes = make_nodes(runtime or GraphRuntime())
    builder = StateGraph(ReviewState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "review")
    builder.add_edge("review", "judge")
    builder.add_edge("judge", "present")
    builder.add_edge("present", END)
    return builder.compile(checkpointer=checkpointer)


def run_graph(
    state: ReviewState,
    runtime: GraphRuntime | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> ReviewState:
    """Run the graph once and return the final state. The run is one LangSmith root run named
    ``review-graph``; ``review_key`` is the ``thread_id``."""
    graph = build_graph(checkpointer, runtime)
    run_config: dict[str, Any] = {
        "run_name": "review-graph",
        "tags": ["review-graph", f"judge:{state['judge']}"],
        "metadata": {
            "review_key": state["review_key"],
            "judge": state["judge"],
            "evaluation": state["evaluation"],
        },
        "configurable": {"thread_id": state["review_key"]},
        "recursion_limit": 25,
    }
    if config:
        run_config.update(config)
    return graph.invoke(state, run_config)


# --- Totals ----------------------------------------------------------------------------------

_SUMS = ("model_calls", "tool_calls", "input_tokens", "output_tokens", "duration_s", "wait_s_sum",
         "wait_s_wall", "net_duration_s")  # fmt: skip


def _stage(result: dict[str, Any] | None) -> dict[str, float]:
    result = result or {}
    return {k: result.get(k, 0) or 0 for k in _SUMS}


def stage_totals(state: ReviewState) -> dict[str, dict[str, float]]:
    """Calls, tokens, time and waits of the ``review`` stage, the ``judge`` stage and ``total``.
    The stages run one after the other and each counts only its own agents, so the total is their
    plain sum: nothing is counted twice, the wall-clock waits add up (they do not overlap), and
    net time is the sum of each stage's net time."""
    review = _stage(state.get("review_result"))
    judge = _stage(state.get("judge_result"))
    total = {k: round(review[k] + judge[k], 3) for k in _SUMS}
    return {"review": review, "judge": judge, "total": total}
