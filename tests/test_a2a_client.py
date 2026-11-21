"""The A2A client and the orchestrator's remote mode (stage 3, decision 0011).

Fake HTTP transports and fake responses only: no network, model or Docker call, no sleeping.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from reviewer import a2a_client as client
from reviewer.a2a_client import RemoteError, RemoteUnavailable, call_remote_security
from reviewer.a2a_schema import SecurityReviewRequest
from reviewer.agents import orchestrator as orch
from reviewer.agents.orchestrator import orchestrate
from reviewer.config import load_models_config, load_review_config
from reviewer.models import QuotaRouter, QuotaTracker, review_waits
from reviewer.snapshot import pack, sha256_hex
from tests.dev_pr import BODY, TITLE, make_dev_workspace
from tests.test_orchestrator import BUG, SEC, Fakes, call, fnd, full_plan, orch_answer, script
from tests.test_single_agent import FakeTime, ai

BASE = "http://127.0.0.1:2025"
AID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def ctx(tmp_path):
    return make_dev_workspace(tmp_path)


@pytest.fixture
def fake_time():
    return FakeTime()


@pytest.fixture
def router(tmp_path, fake_time):
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota_state.json", providers)
    return QuotaRouter(
        config, evaluation=True, tracker=tracker, limiters={}, clock=fake_time.clock,
        sleep=fake_time.sleep,
    )  # fmt: skip


def response_dict(**over) -> dict:
    base = {
        "agent": "security_reviewer", "status": "complete", "findings": [], "dropped": [],
        "notes": [], "summary": "ok", "error_class": None, "error": None, "limit": None,
        "reset_at": None, "model_id": "gemini-3.1-flash-lite", "prompt_version": "6f396484c1bb",
        "evaluation": True,
        "counts": {"model_calls": 3, "tool_calls": 2, "tool_calls_by_tool": {"read_diff": 2},
                   "input_tokens": 3000, "output_tokens": 150, "duration_s": 12.0, "wait_s": 4.0,
                   "net_duration_s": 8.0},
        "retries_by_reason": {"retry_server": 1}, "waits_by_reason": {"retry_server": 4.0},
        "wait_intervals": [], "service": {"name": "security_reviewer", "pid": 4242,
                                          "version": "6f396484c1bb"},
    }  # fmt: skip
    base.update(over)
    return base


def a2a_body(response: dict, *, wrapped=False, state="completed") -> dict:
    task = {
        "kind": "task", "id": "t", "contextId": "c", "status": {"state": state},
        "artifacts": [{"artifactId": "a", "name": "Assistant Response",
                       "parts": [{"kind": "text", "text": json.dumps(response)}]}],
    }  # fmt: skip
    return {"jsonrpc": "2.0", "id": "1", "result": {"task": task} if wrapped else task}


class Server:
    """An ``httpx.MockTransport`` handler that plays the service and records the requests."""

    def __init__(self, reply=None, *, search=None, raises=None):
        self.reply = reply
        self.search = (
            search
            if search is not None
            else [{"assistant_id": AID, "graph_id": "security_reviewer"}]
        )
        self.raises = raises
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if request.url.path == "/assistants/search":
            return httpx.Response(200, json=self.search)
        reply = self.reply(request) if callable(self.reply) else self.reply
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    @property
    def transport(self):
        return httpx.MockTransport(self)

    @property
    def rpc(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.startswith("/a2a/")]


@pytest.fixture(autouse=True)
def _fresh_cache():
    client.forget_assistant_ids()
    yield
    client.forget_assistant_ids()


def call_it(ctx, server, **kw):
    args = dict(pr_title=TITLE, pr_body=BODY, focus="look at the store", evaluation=True,
                base_url=BASE, transport=server.transport)  # fmt: skip
    args.update(kw)
    return call_remote_security(ctx, **args)


# --- The request -----------------------------------------------------------------------------


def test_request_is_one_jsonrpc_message_send_with_a_json_text_part(ctx):
    server = Server(a2a_body(response_dict()))
    out = call_it(ctx, server)
    (req,) = server.rpc
    assert req.url.path == f"/a2a/{AID}" and req.method == "POST"
    assert req.headers["accept"] == "application/json"
    rpc = json.loads(req.content)
    assert rpc["jsonrpc"] == "2.0" and rpc["method"] == "message/send" and rpc["id"]
    msg = rpc["params"]["message"]
    assert msg["role"] == "user" and msg["messageId"] and len(msg["parts"]) == 1
    assert rpc["params"]["configuration"] == {"historyLength": 0}
    part = msg["parts"][0]
    assert part["kind"] == "text"
    sent = SecurityReviewRequest.model_validate_json(part["text"])
    assert (sent.protocol_version, sent.pr_title, sent.focus) == ("1", TITLE, "look at the store")
    assert sent.diff_text == ctx.diff_text and sent.evaluation is True and sent.trace_parent is None
    assert sha256_hex(sent.archive_bytes()) == sent.workspace_sha256
    assert out["transport"]["request_bytes"] == len(req.content)


def test_the_archive_holds_the_files_run_tests_would_send_and_is_deterministic(ctx):
    from reviewer.tools.testing import workspace_files

    first, second = Server(a2a_body(response_dict())), Server(a2a_body(response_dict()))
    call_it(ctx, first)
    call_it(ctx, second)
    shas = []
    for server in (first, second):
        text = json.loads(server.rpc[0].content)["params"]["message"]["parts"][0]["text"]
        shas.append(json.loads(text)["workspace_sha256"])
    assert shas[0] == shas[1] == sha256_hex(pack(workspace_files(ctx)))


def test_trace_parent_is_sent_when_given(ctx):
    server = Server(a2a_body(response_dict()))
    call_it(
        ctx, server, trace_parent={"langsmith-trace": "20260101T000000000000Zabc", "baggage": "x=y"}
    )
    text = json.loads(server.rpc[0].content)["params"]["message"]["parts"][0]["text"]
    assert json.loads(text)["trace_parent"]["baggage"] == "x=y"


def test_assistant_id_is_discovered_by_graph_name_and_cached(ctx):
    server = Server(a2a_body(response_dict()))
    call_it(ctx, server)
    call_it(ctx, server)
    searches = [r for r in server.requests if r.url.path == "/assistants/search"]
    assert len(searches) == 1
    assert json.loads(searches[0].content)["graph_id"] == "security_reviewer"
    assert len(server.rpc) == 2


def test_a_given_assistant_id_skips_the_search(ctx):
    server = Server(a2a_body(response_dict()))
    call_it(ctx, server, assistant_id="given-id")
    assert [r.url.path for r in server.requests] == ["/a2a/given-id"]


def test_no_such_graph_is_an_invalid_response(ctx):
    with pytest.raises(RemoteError, match="no graph named") as e:
        call_it(ctx, Server(None, search=[]))
    assert e.value.code == "invalid response"


@pytest.mark.parametrize(("evaluation", "seconds"), [(True, 900.0), (False, 300.0)])
def test_default_timeout_depends_on_the_mode(ctx, evaluation, seconds):
    def reply(request):
        assert request.extensions["timeout"]["read"] == seconds
        assert request.extensions["timeout"]["connect"] == client.CONNECT_TIMEOUT_S
        return a2a_body(response_dict(evaluation=evaluation))

    call_it(ctx, Server(reply), evaluation=evaluation)


def test_explicit_timeout_wins(ctx):
    def reply(request):
        assert request.extensions["timeout"]["read"] == 7.0
        return a2a_body(response_dict())

    call_it(ctx, Server(reply), timeout_s=7.0)


# --- The response ----------------------------------------------------------------------------


@pytest.mark.parametrize("wrapped", [False, True], ids=["legacy_result_is_task", "v1_result_task"])
def test_both_result_shapes_are_read(ctx, wrapped):
    finding = fnd("security_reviewer", category="hardcoded_secret").model_dump(mode="json")
    server = Server(a2a_body(response_dict(findings=[finding]), wrapped=wrapped))
    out = call_it(ctx, server)
    assert out["findings"][0]["category"] == "hardcoded_secret"
    assert out["service"]["pid"] == 4242 and out["counts"]["model_calls"] == 3
    assert out["retries_by_reason"] == {"retry_server": 1}
    assert out["transport"]["name"] == "a2a" and out["transport"]["base_url"] == BASE
    assert out["transport"]["round_trip_s"] >= 0


def test_connection_refused_is_remote_unavailable(ctx):
    server = Server(raises=httpx.ConnectError("refused"))
    with pytest.raises(RemoteUnavailable, match=f"security service is not running at {BASE}"):
        call_it(ctx, server)


def test_http_error_is_remote_error_with_status_and_excerpt(ctx):
    server = Server(httpx.Response(502, text="bad gateway " * 100))
    with pytest.raises(RemoteError) as e:
        call_it(ctx, server)
    assert e.value.code == 502 and e.value.detail.startswith("bad gateway")
    assert len(e.value.detail) <= client.EXCERPT_CHARS


def test_jsonrpc_error_is_remote_error_with_code_and_message(ctx):
    body = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32602, "message": "Invalid params"}}
    with pytest.raises(RemoteError) as e:
        call_it(ctx, Server(body))
    assert (e.value.code, e.value.detail) == (-32602, "Invalid params")


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="not json"),
        [1, 2],
        {"jsonrpc": "2.0", "id": "1"},
        {"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "status": {"state": "completed"}}},
        {"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "artifacts": [
            {"parts": [{"kind": "text", "text": "this is not json"}]}]}},
        {"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "artifacts": [
            {"parts": [{"kind": "text", "text": json.dumps({"status": "complete"})}]}]}},
    ],
)  # fmt: skip
def test_invalid_responses(ctx, reply):
    with pytest.raises(RemoteError) as e:
        call_it(ctx, Server(reply))
    assert e.value.code == "invalid response"


def test_a_failed_task_is_a_remote_error(ctx):
    message = {"parts": [{"kind": "text", "text": "Error executing assistant: boom"}]}
    status = {"state": "failed", "message": message}
    body = {"jsonrpc": "2.0", "id": "1", "result": {"kind": "task", "status": status}}
    with pytest.raises(RemoteError, match="boom") as e:
        call_it(ctx, Server(body))
    assert e.value.code == "task failed"


def test_timeout_is_remote_error(ctx):
    with pytest.raises(RemoteError) as e:
        call_it(ctx, Server(raises=httpx.ReadTimeout("slow")), timeout_s=12.5, assistant_id=AID)
    assert (e.value.code, e.value.detail) == ("timeout", 12.5)


def test_an_error_response_from_the_service_is_returned_not_raised(ctx):
    resp = response_dict(status="error", error_class="ArchiveError", error="bad", model_id=None)
    out = call_it(ctx, Server(a2a_body(resp)))
    assert out["status"] == "error" and out["error_class"] == "ArchiveError"


# --- The orchestrator in remote mode ---------------------------------------------------------


def remote_config(mode="remote"):
    return load_review_config().with_security_mode(mode)


def orchestrate_remote(ctx, router, fake_time, monkeypatch, remote, *, fakes=None, mode="remote"):
    """Run the orchestrator with the real default runner; bug and test reviewers are replaced
    by fake results, the security one goes through ``remote`` (a fake ``call_remote_security``)."""
    fakes = fakes or Fakes()
    monkeypatch.setattr(orch, "call_remote_security", remote)

    def runner_factory(ctx_, title, body, router_, *, item_id, clock, config):
        real = orig(ctx_, title, body, router_, item_id=item_id, clock=clock, config=config)

        def run(name, focus, cfg):
            return real(name, focus, cfg) if name == SEC else fakes(name, focus, cfg)

        return run

    orig = orch.default_runner
    monkeypatch.setattr(orch, "default_runner", runner_factory)
    return orchestrate(ctx, TITLE, BODY, router, clock=fake_time.clock, config=remote_config(mode))


def test_remote_mode_uses_the_client_and_records_the_transport(ctx, router, fake_time, monkeypatch):
    sent = {}

    def remote(ctx_, **kw):
        sent.update(kw)
        timing = {"name": "a2a", "base_url": BASE, "round_trip_s": 1.5, "request_bytes": 321}
        return {**response_dict(), "transport": timing}

    script(router, [ai(call(SEC, "the store")), orch_answer(*full_plan((SEC,)))])
    res = orchestrate_remote(ctx, router, fake_time, monkeypatch, remote)
    assert res.status == "complete"
    (c,) = res.children
    assert (c.transport, c.agent) == ("a2a", SEC)
    d = c.transport_detail
    assert (d.service_pid, d.service_version, d.round_trip_s, d.request_bytes) == (
        4242,
        "6f396484c1bb",
        1.5,
        321,
    )
    assert d.base_url == BASE
    assert sent["base_url"] == BASE and sent["focus"] == "the store" and sent["evaluation"] is True
    # The child's own figures are counted once, as a child.
    assert (c.model_calls, c.input_tokens) == (3, 3000)
    assert (res.own_model_calls, res.model_calls) == (2, 5)
    assert c.retries_by_reason == {"retry_server": 1}


def test_local_mode_records_transport_local(ctx, router, fake_time, monkeypatch):
    script(router, [ai(call(SEC)), orch_answer(*full_plan((SEC,)))])
    monkeypatch.setattr(orch, "run_security_review", lambda request, **kw: response_dict())
    res = orchestrate(
        ctx, TITLE, BODY, router, clock=fake_time.clock, config=remote_config("local")
    )
    (c,) = res.children
    assert c.transport == "local" and c.transport_detail is None


def test_service_not_running_makes_the_child_an_error_with_a_note(
    ctx, router, fake_time, monkeypatch
):
    def remote(ctx_, **kw):
        raise RemoteUnavailable(kw["base_url"])

    script(router, [ai(call(BUG), call(SEC)), orch_answer(*full_plan((BUG, SEC)))])
    res = orchestrate_remote(ctx, router, fake_time, monkeypatch, remote)
    sec = next(c for c in res.children if c.agent == SEC)
    assert sec.status == "error" and sec.error_class == "RemoteUnavailable"
    assert sec.notes == [f"security service unavailable: {BASE}"]
    assert res.status == "partial"  # the bug reviewer's findings are kept
    from eval.run import decide_scoring

    assert decide_scoring(res) == ("rerun_required", f"RemoteUnavailable ({SEC})")


def test_other_remote_failures_are_errors_too_and_rerun_required(
    ctx, router, fake_time, monkeypatch
):
    def remote(ctx_, **kw):
        raise RemoteError(502, "bad gateway")

    script(router, [ai(call(SEC)), orch_answer(*full_plan((SEC,)))])
    res = orchestrate_remote(ctx, router, fake_time, monkeypatch, remote)
    (sec,) = res.children
    assert sec.status == "error" and sec.error_class == "RemoteError"
    from eval.run import decide_scoring

    assert decide_scoring(res)[0] == "rerun_required"


def test_remote_waits_join_the_orchestrators_union_and_sum(ctx, router, fake_time, monkeypatch):
    """The service's wait intervals (Unix timestamps) overlap an orchestrator-side wait in time;
    the union counts the overlap once, the sum counts it twice (D7 across processes)."""
    now = time.time()
    # fake_time.clock() reads 1000.0 at the start; the remote waits end "now".
    intervals = [
        {"start": now - 10.0, "end": now - 6.0, "reason": "rate_limiter"},
        {"start": now - 8.0, "end": now - 2.0, "reason": "retry_server"},
    ]

    def remote(ctx_, **kw):
        return {**response_dict(retries_by_reason={"retry_server": 1},
                                waits_by_reason={"rate_limiter": 4.0, "retry_server": 6.0},
                                wait_intervals=intervals),
                "transport": {"name": "a2a", "base_url": BASE, "round_trip_s": 1.0,
                              "request_bytes": 10}}  # fmt: skip

    script(router, [ai(call(SEC)), orch_answer(*full_plan((SEC,)))])
    res = orchestrate_remote(ctx, router, fake_time, monkeypatch, remote)
    assert res.wait_s_sum == pytest.approx(10.0, abs=0.01)
    assert res.wait_s_wall == pytest.approx(8.0, abs=0.01)  # [-10, -2] is one stretch
    assert res.waits_by_reason == {"rate_limiter": 4.0, "retry_server": 6.0}
    assert res.retries_by_reason == {"retry_server": 1}  # once, as the child's
    assert res.net_duration_s == pytest.approx(res.duration_s - 8.0, abs=0.01)


def test_join_remote_waits_without_a_review_log_is_a_noop():
    orch.join_remote_waits(
        [{"start": 1.0, "end": 2.0, "reason": "rate_limiter"}], "m", time.monotonic
    )


def test_join_remote_waits_maps_unix_time_onto_the_clock():
    clock = FakeTime()
    now = time.time()
    with review_waits() as log:
        orch.join_remote_waits(
            [{"start": now - 5.0, "end": now - 1.0, "reason": "retry_minute"}], "m", clock.clock
        )
    (w,) = log.waits
    assert w.seconds == pytest.approx(4.0) and w.nested and w.reason == "retry_minute"
    assert w.end == pytest.approx(clock.clock() - 1.0, abs=0.1)


# --- Config ----------------------------------------------------------------------------------


def test_config_default_is_local_and_the_override_copies():
    cfg = load_review_config()
    assert cfg.a2a.security_reviewer.mode == "local"
    assert cfg.a2a.security_reviewer.base_url == "http://127.0.0.1:2025"
    remote = cfg.with_security_mode("remote")
    assert remote.a2a.security_reviewer.mode == "remote"
    assert remote.a2a.security_reviewer.base_url == cfg.a2a.security_reviewer.base_url
    assert cfg.a2a.security_reviewer.mode == "local"  # the original is untouched


def test_trace_headers_are_none_when_tracing_is_off():
    assert client.trace_headers({}) is None
