"""The security specialist as its own service (stage 3, decision 0011).

A LangGraph graph with ``MessagesState`` and one node. ``langgraph dev`` (LangGraph Agent Server)
serves it, and its A2A endpoint ``/a2a/{assistant_id}`` turns an A2A ``message/send`` text part
into the last human message of the state (``langgraph_api/api/a2a.py``,
``_process_a2a_message_parts``) and returns the last AI message's text in the task's artifacts.

The human message holds the JSON of a ``SecurityReviewRequest``: the PR and a snapshot of its
workspace. The node validates it, checks the archive's SHA-256, unpacks it safely into a new
temporary directory, runs ``run_security_review`` on that directory and answers with the JSON
response plus a ``service`` block. The temporary directory is always deleted. An invalid request
is an ``error`` response (same shape), never an exception.

Quota: this process has its own ``QuotaTracker`` (``quota_state.a2a.json``) and rate limiters, so
two processes never write the same state file; together they can exceed a model's RPM, which the
per-call retries absorb (decision 0011).

The service listens on 127.0.0.1 only and has no authentication.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langsmith.run_helpers import tracing_context
from langsmith.utils import tracing_is_enabled
from pydantic import ValidationError

from reviewer.a2a_schema import PROTOCOL_VERSION, SERVICE_NAME, SecurityReviewRequest
from reviewer.agents.specialists import SECURITY, run_security_review, specialist_spec
from reviewer.config import load_models_config, repo_root
from reviewer.models import QuotaRouter, QuotaTracker
from reviewer.snapshot import ArchiveError, check_sha256, unpack
from reviewer.test_hooks import fake_model_script, script_builder

STATE_FILE = "quota_state.a2a.json"
_TRACE_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# Fail at start-up, not at the first request, when the test-only fake model is asked for outside
# test mode.
fake_model_script()

_tracker_lock = threading.Lock()
_tracker: QuotaTracker | None = None


def prompt_version() -> str:
    return specialist_spec(SECURITY).prompt_version()


def _service_block() -> dict[str, Any]:
    return {"name": SERVICE_NAME, "pid": os.getpid(), "version": prompt_version()}


def _quota_tracker() -> QuotaTracker:
    """This process's own tracker, saved to ``<repo root>/quota_state.a2a.json``."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            config = load_models_config()
            providers = {m: s.provider for m, s in config.models.items()}
            _tracker = QuotaTracker(repo_root() / STATE_FILE, providers)
        return _tracker


def make_router(evaluation: bool) -> QuotaRouter:
    router = QuotaRouter(evaluation=evaluation, tracker=_quota_tracker())
    script = fake_model_script()  # raises TestHookRejected outside test mode
    if script is not None:
        router.build = script_builder(script)
    return router


def error_response(error_class: str, message: str, *, evaluation: bool = False) -> dict[str, Any]:
    """A ``SecurityReviewResponse`` for a request that could not be served."""
    return {
        "agent": SECURITY,
        "status": "error",
        "findings": [],
        "dropped": [],
        "notes": [],
        "summary": None,
        "error_class": error_class,
        "error": message,
        "limit": None,
        "reset_at": None,
        "model_id": None,
        "prompt_version": prompt_version(),
        "evaluation": evaluation,
        "counts": {
            "model_calls": 0,
            "tool_calls": 0,
            "tool_calls_by_tool": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_s": 0.0,
            "wait_s": 0.0,
            "net_duration_s": 0.0,
        },
        "retries_by_reason": {},
        "waits_by_reason": {},
        "wait_intervals": [],
        "service": _service_block(),
    }


def _rmtree(path: Path) -> None:
    def onexc(func, p, exc):  # read-only files on Windows
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    shutil.rmtree(path, onexc=onexc)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _trace_id(headers: dict[str, str] | None) -> str | None:
    """The trace id in a ``langsmith-trace`` header (the first UUID of its dotted order)."""
    m = _TRACE_ID.search((headers or {}).get("langsmith-trace", ""))
    return m.group(0) if m else None


def serve(text: str) -> dict[str, Any]:
    """The whole job for one request text: the response dict, always (never raises for a bad
    request or a failed review)."""
    try:
        request = SecurityReviewRequest.model_validate_json(text)
    except ValidationError as e:
        reasons = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'request'}: {err['msg']}"
            for err in e.errors(include_url=False, include_input=False)
        )
        return error_response("InvalidRequest", f"invalid request: {reasons}"[:1000])
    if request.protocol_version != PROTOCOL_VERSION:  # the model's Literal makes this unreachable
        return error_response("InvalidRequest", "unsupported protocol_version")
    tmp: Path | None = None
    try:
        try:
            archive = request.archive_bytes()
            check_sha256(archive, request.workspace_sha256)
        except (ArchiveError, ValueError) as e:
            return error_response("ArchiveError", str(e), evaluation=request.evaluation)
        tmp = Path(tempfile.mkdtemp(prefix="mapr-security-"))
        try:
            unpack(archive, tmp)
        except ArchiveError as e:
            return error_response("ArchiveError", str(e), evaluation=request.evaluation)
        metadata: dict[str, Any] = {"a2a_service": SERVICE_NAME, "a2a_pid": os.getpid()}
        if trace_id := _trace_id(request.trace_parent):
            metadata["orchestrator_trace_id"] = trace_id
        payload = {
            "workspace": str(tmp),
            "diff_text": request.diff_text,
            "pr_title": request.pr_title,
            "pr_body": request.pr_body,
            "focus": request.focus,
            "evaluation": request.evaluation,
        }
        router = make_router(request.evaluation)
        if request.trace_parent and tracing_is_enabled():
            # The review's runs become children of the orchestrator's tool-call run.
            with tracing_context(parent=request.trace_parent):
                response = run_security_review(
                    payload, router=router, parent_config={"metadata": metadata}
                )
        else:
            response = run_security_review(
                payload, router=router, parent_config={"metadata": metadata}
            )
        response["service"] = _service_block()
        return response
    except Exception as e:  # a bug here must still be an answer the caller can read
        return error_response(
            "InternalError", f"{type(e).__name__}: {e}"[:1000], evaluation=request.evaluation
        )
    finally:
        if tmp is not None:
            _rmtree(tmp)


def review(state: MessagesState) -> dict[str, Any]:
    """The graph's only node: the last human message in, the JSON response as an AI message."""
    humans = [m for m in state["messages"] if m.type == "human"]
    if not humans:
        response = error_response("InvalidRequest", "no human message in the input")
    else:
        response = serve(_message_text(humans[-1]))
    return {"messages": [AIMessage(content=json.dumps(response, separators=(",", ":")))]}


builder = StateGraph(MessagesState)
builder.add_node("review", review)
builder.add_edge(START, "review")
builder.add_edge("review", END)
graph = builder.compile()
