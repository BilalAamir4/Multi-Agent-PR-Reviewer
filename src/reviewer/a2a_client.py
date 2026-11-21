"""The A2A client of the orchestrator (stage 3, decision 0011): calls the security specialist
service with one JSON-RPC 2.0 ``message/send`` request.

The request is one text part holding the JSON of a ``SecurityReviewRequest``: the PR's title,
body and diff, and a snapshot of the workspace (a deterministic base64 ``tar.gz`` of the files
``run_tests`` would send, with its SHA-256), so the service never reads this machine's
filesystem. The installed ``langgraph-api`` (0.18.3, ``langgraph_api/api/a2a.py``) turns a text
part into a human message, runs the graph and returns a task whose ``artifacts`` hold the last AI
message as a text part; the graph puts the JSON response there. ``configuration.historyLength: 0``
keeps the server from echoing the (large) request back in the task history.

Failures are exceptions with a stable shape: ``RemoteUnavailable`` (nothing is listening) and
``RemoteError(code, detail)`` (HTTP error, JSON-RPC error, failed task, timeout or an invalid
response).
"""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from typing import Any

import httpx
from pydantic import ValidationError

from reviewer.a2a_schema import (
    PROTOCOL_VERSION,
    SERVICE_NAME,
    TRACE_HEADER_KEYS,
    SecurityReviewRequest,
    SecurityReviewResponse,
)
from reviewer.snapshot import pack, sha256_hex
from reviewer.tools import ReviewContext
from reviewer.tools.testing import workspace_files

TIMEOUT_EVALUATION_S = 900.0  # reviews wait on rate limits
TIMEOUT_NORMAL_S = 300.0
CONNECT_TIMEOUT_S = 10.0
EXCERPT_CHARS = 500


class RemoteUnavailable(Exception):
    """Nothing answers at the service's address (connection refused or unreachable)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        super().__init__(f"security service is not running at {base_url}")


class RemoteError(Exception):
    """The service was reached but the call failed. ``code`` is the HTTP status, the JSON-RPC
    error code, or one of ``"timeout"``, ``"invalid response"``, ``"task failed"``."""

    def __init__(self, code: int | str, detail: Any) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def default_timeout(evaluation: bool) -> float:
    return TIMEOUT_EVALUATION_S if evaluation else TIMEOUT_NORMAL_S


# --- The request -----------------------------------------------------------------------------


def build_request(
    ctx: ReviewContext,
    *,
    pr_title: str,
    pr_body: str,
    focus: str | None,
    evaluation: bool,
    spawn_enabled: bool,
    trace_parent: dict[str, str] | None = None,
) -> SecurityReviewRequest:
    archive = pack(workspace_files(ctx))
    return SecurityReviewRequest(
        protocol_version=PROTOCOL_VERSION,
        pr_title=pr_title,
        pr_body=pr_body,
        focus=focus,
        diff_text=ctx.diff_text,
        workspace_archive=base64.b64encode(archive).decode("ascii"),
        workspace_sha256=sha256_hex(archive),
        evaluation=evaluation,
        spawn_enabled=spawn_enabled,
        trace_parent=trace_parent,
    )


def jsonrpc_body(request: SecurityReviewRequest) -> str:
    """The JSON-RPC 2.0 ``message/send`` body for ``request``."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": request.model_dump_json()}],
                "messageId": str(uuid.uuid4()),
            },
            "configuration": {"historyLength": 0},
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def trace_headers(config: dict[str, Any] | None) -> dict[str, str] | None:
    """LangSmith distributed-tracing headers for the run that is current in ``config`` (the
    orchestrator's tool call), or None when tracing is off. Only the two header keys the service
    accepts are returned."""
    try:
        from langsmith.run_helpers import get_current_run_tree
        from langsmith.run_trees import RunTree
        from langsmith.utils import tracing_is_enabled

        if not tracing_is_enabled():
            return None
        tree = RunTree.from_runnable_config(config) if config else None
        tree = tree or get_current_run_tree()
        if tree is None:
            return None
        headers = {k: v for k, v in tree.to_headers().items() if k in TRACE_HEADER_KEYS}
        return headers or None
    except Exception:  # tracing must never break a review
        return None


# --- The response ----------------------------------------------------------------------------

_FAILED_STATES = {"failed", "TASK_STATE_FAILED", "rejected", "TASK_STATE_REJECTED", "canceled",
                  "TASK_STATE_CANCELED", "auth-required", "TASK_STATE_AUTH_REQUIRED"}  # fmt: skip


def _text_of(parts: Any) -> list[str]:
    out: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
    return out


def extract_response_text(body: Any) -> str:
    """The JSON text the graph returned, from a ``message/send`` response body. The legacy method
    name makes the server answer with ``result`` = the task; the v1 name wraps it as
    ``result.task``; both are accepted. Raises ``RemoteError`` for a JSON-RPC error, a task that
    did not complete, or a response with no text."""
    if not isinstance(body, dict):
        raise RemoteError("invalid response", f"expected a JSON object, got {type(body).__name__}")
    if "error" in body:
        err = body["error"] if isinstance(body["error"], dict) else {}
        raise RemoteError(err.get("code", "unknown"), err.get("message", str(body["error"])))
    result = body.get("result")
    if not isinstance(result, dict):
        raise RemoteError("invalid response", "no 'result' in the JSON-RPC response")
    task = result.get("task") if isinstance(result.get("task"), dict) else result
    status = task.get("status") if isinstance(task.get("status"), dict) else {}
    state = status.get("state")
    if state in _FAILED_STATES:
        texts = _text_of((status.get("message") or {}).get("parts"))
        raise RemoteError("task failed", " ".join(texts)[:EXCERPT_CHARS] or str(state))
    texts: list[str] = []
    for artifact in task.get("artifacts") or []:
        if isinstance(artifact, dict):
            texts += _text_of(artifact.get("parts"))
    if not texts:
        raise RemoteError("invalid response", f"the task has no text artifact (state {state})")
    return "".join(texts)


def parse_response(body: Any) -> SecurityReviewResponse:
    text = extract_response_text(body)
    try:
        return SecurityReviewResponse.model_validate_json(text)
    except ValidationError as e:
        raise RemoteError(
            "invalid response", f"{e.error_count()} errors: {e}"[:EXCERPT_CHARS]
        ) from e


# --- Calling the service ---------------------------------------------------------------------

_assistant_ids: dict[tuple[str, str], str] = {}
_assistant_lock = threading.Lock()


def discover_assistant_id(
    base_url: str, *, graph: str = SERVICE_NAME, client: httpx.Client | None = None
) -> str:
    """The assistant ID of graph ``graph``, found with ``POST /assistants/search`` and cached per
    process (decision 0011)."""
    key = (base_url.rstrip("/"), graph)
    with _assistant_lock:
        if key in _assistant_ids:
            return _assistant_ids[key]
    own = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=CONNECT_TIMEOUT_S))
    try:
        r = http.post(f"{key[0]}/assistants/search", json={"graph_id": graph, "limit": 10})
        r.raise_for_status()
        found = [a for a in r.json() if isinstance(a, dict) and a.get("graph_id") == graph]
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise RemoteUnavailable(key[0]) from e
    except httpx.HTTPStatusError as e:
        raise RemoteError(e.response.status_code, e.response.text[:EXCERPT_CHARS]) from e
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise RemoteError("timeout", 30.0) from e
    except ValueError as e:
        raise RemoteError("invalid response", f"assistants/search is not JSON: {e}") from e
    finally:
        if own:
            http.close()
    if not found:
        raise RemoteError("invalid response", f"the service has no graph named {graph!r}")
    with _assistant_lock:
        _assistant_ids[key] = str(found[0]["assistant_id"])
        return _assistant_ids[key]


def forget_assistant_ids() -> None:
    """Clear the per-process assistant ID cache (tests, and after restarting a service)."""
    with _assistant_lock:
        _assistant_ids.clear()


def call_remote_security(
    ctx: ReviewContext,
    *,
    pr_title: str,
    pr_body: str,
    focus: str | None,
    evaluation: bool,
    spawn_enabled: bool,
    base_url: str,
    assistant_id: str | None = None,
    timeout_s: float | None = None,
    trace_parent: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Run the security review of ``ctx`` on the service at ``base_url`` and return its response
    (the shape of ``run_security_review``'s, validated) plus ``"transport"``: ``{"name": "a2a",
    "base_url", "round_trip_s", "request_bytes"}``. ``assistant_id`` is discovered by graph name
    when not given. ``timeout_s`` defaults to 900 s in evaluation mode and 300 s otherwise.
    ``transport`` replaces the HTTP transport (tests)."""
    base = base_url.rstrip("/")
    timeout = timeout_s if timeout_s is not None else default_timeout(evaluation)
    request = build_request(
        ctx, pr_title=pr_title, pr_body=pr_body, focus=focus, evaluation=evaluation,
        spawn_enabled=spawn_enabled, trace_parent=trace_parent,
    )  # fmt: skip
    body = jsonrpc_body(request)
    payload = body.encode("utf-8")
    started = time.monotonic()
    timeouts = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_S)
    with httpx.Client(timeout=timeouts, transport=transport) as client:
        aid = assistant_id or discover_assistant_id(base, client=client)
        try:
            r = client.post(
                f"{base}/a2a/{aid}",
                content=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise RemoteUnavailable(base) from e
        except httpx.TimeoutException as e:
            raise RemoteError("timeout", timeout) from e
        except httpx.TransportError as e:
            raise RemoteError("transport", f"{type(e).__name__}: {e}"[:EXCERPT_CHARS]) from e
    if r.status_code >= 400:
        raise RemoteError(r.status_code, r.text[:EXCERPT_CHARS])
    try:
        parsed = r.json()
    except ValueError as e:
        raise RemoteError("invalid response", f"the body is not JSON: {e}") from e
    response = parse_response(parsed)
    out = response.model_dump(mode="json")
    out["transport"] = {
        "name": "a2a",
        "base_url": base,
        "round_trip_s": round(time.monotonic() - started, 3),
        "request_bytes": len(payload),
    }
    return out
