"""The review graph with human approval (decision 0016):
``START -> review -> judge -> present -> approve -> post -> END``.

``review``, ``judge`` and ``present`` are the node functions of ``reviewer.graph`` (stage 4B),
used unchanged; ``reviewer.graph.build_graph`` is not touched. ``approve`` calls ``interrupt`` with
the approval payload and stores the validated decisions; ``post`` creates one GitHub review, once.

Checkpoints are written after every node (``thread_id`` = ``review_key``). A node that was running
when the process died re-runs from its start; the nodes are idempotent (decision 0013), and
``post`` checks for its marker first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from approval.decisions import apply_decisions, validate_decisions
from approval.github import GitHubClient
from approval.posting import build_review, marker_for
from reviewer.graph import ABORTED, GraphRuntime, make_nodes
from reviewer.schemas import ReviewState

NODE_ORDER = ("review", "judge", "present", "approve", "post")


class ApprovalState(ReviewState, total=False):
    """``ReviewState`` plus the pull request and the approval results (all JSON-serializable; no
    token)."""

    repo: str  # "owner/name"
    pr_number: int
    head_sha: str
    base_sha: str
    pr_url: str
    decisions: list[dict[str, Any]]
    approved_findings: list[dict[str, Any]]
    posted: dict[str, Any]


@dataclass
class ApprovalRuntime:
    """What ``post`` uses besides the state: a factory for the GitHub client (reads the token from
    the environment when called)."""

    github: Callable[[], GitHubClient]


def make_approval_nodes(runtime: ApprovalRuntime) -> dict[str, Callable[..., dict[str, Any]]]:
    def approve(state: ApprovalState) -> dict[str, Any]:
        if state.get("status") in ABORTED:  # nothing trustworthy to approve
            return {"decisions": [], "approved_findings": []}
        payload = state["approval_payload"]
        if not payload:
            return {"decisions": [], "approved_findings": []}
        decisions = validate_decisions(interrupt(payload), payload)
        return {"decisions": decisions, "approved_findings": apply_decisions(decisions, payload)}

    def post(state: ApprovalState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("status") in ABORTED:
            return {"posted": {"skipped": f"the review ended {state['status']}"}}
        approved = state.get("approved_findings") or []
        if not approved:
            return {"posted": {"skipped": "no approved findings"}}
        owner, repo = state["repo"].split("/", 1)
        number = state["pr_number"]
        marker = marker_for(state["review_key"])
        client = runtime.github()
        try:
            for r in client.list_reviews(owner, repo, number):
                if marker in (r.get("body") or ""):
                    return {
                        "posted": {
                            "review_id": r["id"],
                            "url": r.get("html_url"),
                            "existed": True,
                        }
                    }
            diff_text = Path(state["diff_path"]).read_bytes().decode("utf-8")
            plan = build_review(approved, diff_text, marker)
            made = client.create_review(
                owner, repo, number, state["head_sha"], plan.body, plan.comments
            )
        finally:
            client.close()
        return {
            "posted": {
                "review_id": made["id"],
                "url": made.get("html_url"),
                "existed": False,
                "inline": plan.inline_ids,
                "in_summary": plan.summary_ids,
            }
        }

    return {"approve": approve, "post": post}


def build_approval_graph(
    checkpointer: BaseCheckpointSaver | None,
    runtime: GraphRuntime | None,
    approval_runtime: ApprovalRuntime,
):
    nodes = make_nodes(runtime or GraphRuntime())
    nodes.update(make_approval_nodes(approval_runtime))
    builder = StateGraph(ApprovalState)
    for name in NODE_ORDER:
        builder.add_node(name, nodes[name])
    builder.add_edge(START, "review")
    for a, b in zip(NODE_ORDER, NODE_ORDER[1:], strict=False):
        builder.add_edge(a, b)
    builder.add_edge("post", END)
    return builder.compile(checkpointer=checkpointer)


def thread_config(review_key: str, **extra: Any) -> dict[str, Any]:
    """Run config of one review: only ``thread_id`` is in ``configurable`` (so only that reaches a
    checkpoint)."""
    return {
        "run_name": "review-graph",
        "tags": ["review-graph", "approval"],
        "metadata": {"review_key": review_key},
        "configurable": {"thread_id": review_key},
        "recursion_limit": 25,
        **extra,
    }
