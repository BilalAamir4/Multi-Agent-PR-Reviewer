"""Stage 3 integration tests (decision 0011): the real HTTP path to a real ``langgraph dev``
service. No model call and no Docker: the service answers with the scripted fake model, switched
on by ``REVIEWER_TEST_MODE=1`` plus ``REVIEWER_TEST_FAKE_MODEL``. They start ``langgraph dev`` on
port 2026 (2025 is the real service, 2024 the stage 0B check), so they skip when ``uv`` is
missing or the port is in use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from reviewer import a2a_client
from reviewer.a2a_client import RemoteUnavailable, call_remote_security
from reviewer.a2a_schema import SecurityReviewResponse
from reviewer.agents.specialists import run_security_review
from reviewer.config import load_models_config
from reviewer.models import QuotaRouter, QuotaTracker
from reviewer.test_hooks import FAKE_MODEL_VAR, TEST_MODE_VAR, script_builder
from scripts import run_a2a_security as svc
from tests.a2a_helpers import FOCUS, comparable, request_for, write_script
from tests.dev_pr import make_fixture_workspace, pr_text

pytestmark = pytest.mark.integration
PORT = 2026  # the test-mode service
REFUSE_PORT = 2027  # a service that must refuse to start
CLOSED_PORT = 2028  # nothing listens here
URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module", autouse=True)
def _need_uv_and_port():
    """Also skips while the real service (2025) runs: both would share ``.langgraph_api``."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    for port in (svc.PORT, PORT, REFUSE_PORT, CLOSED_PORT):
        if svc.port_open(port):
            pytest.skip(f"port {port} is in use")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("a2a_parity")


@pytest.fixture(scope="module")
def ctx(workdir):
    return make_fixture_workspace("dev_security_pr", workdir)


@pytest.fixture(scope="module")
def script(ctx, workdir):
    return write_script(ctx, workdir / "script.json")


@pytest.fixture(scope="module")
def service(workdir, script):
    """A test-mode service on port 2026 whose model is the scripted fake."""
    env = {TEST_MODE_VAR: "1", FAKE_MODEL_VAR: str(script)}
    pid_file = workdir / "service.pid"
    try:
        proc, assistant_id = svc.start(PORT, env, workdir / "service.log", pid_file)
    except RuntimeError as e:
        pytest.fail(str(e))
    yield proc, assistant_id
    assert svc.stop(PORT, pid_file), "the test service did not stop"


def test_the_service_refuses_the_fake_model_outside_test_mode(tmp_path, script):
    """``REVIEWER_TEST_FAKE_MODEL`` without ``REVIEWER_TEST_MODE=1``: the service must not start."""
    saved = os.environ.pop(TEST_MODE_VAR, None)
    log = tmp_path / "refused.log"
    try:
        with pytest.raises(RuntimeError) as e:
            svc.start(REFUSE_PORT, {FAKE_MODEL_VAR: str(script)}, log, tmp_path / "refused.pid")
    finally:
        if saved is not None:
            os.environ[TEST_MODE_VAR] = saved
    assert "TestHookRejected" in str(e.value) or "TestHookRejected" in log.read_text(
        encoding="utf-8", errors="replace"
    )
    assert not svc.port_open(REFUSE_PORT)


def test_a_stopped_service_is_remote_unavailable(tmp_path, ctx):
    url = f"http://127.0.0.1:{CLOSED_PORT}"
    a2a_client.forget_assistant_ids()
    with pytest.raises(RemoteUnavailable, match=f"not running at {url}"):
        call_remote_security(
            ctx, pr_title="t", pr_body="b", focus=None, evaluation=True, base_url=url
        )


def test_remote_response_equals_the_local_one(service, ctx, script, tmp_path):
    """The same scripted model, once in this process and once through HTTP: the responses are
    identical after removing the service block, the timings, the waits and the PIDs."""
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    router = QuotaRouter(evaluation=True, tracker=QuotaTracker(tmp_path / "q.json", providers))
    router.build = script_builder(script)
    local = run_security_review(request_for(ctx), router=router)
    assert local["status"] == "complete" and len(local["findings"]) == 2

    title, body = pr_text("dev_security_pr")
    a2a_client.forget_assistant_ids()
    remote = call_remote_security(
        ctx, pr_title=title, pr_body=body, focus=FOCUS, evaluation=True, base_url=URL
    )
    SecurityReviewResponse.model_validate({k: v for k, v in remote.items() if k != "transport"})
    assert remote["service"]["name"] == "security_reviewer"
    assert remote["service"]["pid"] != os.getpid()  # a different process answered
    assert remote["transport"]["round_trip_s"] > 0 and remote["transport"]["request_bytes"] > 0
    assert comparable(remote) == comparable(local)


def test_a_bad_request_over_http_is_a_clean_error_response(service, ctx):
    import json

    import httpx

    aid = a2a_client.discover_assistant_id(URL)
    req = a2a_client.build_request(ctx, pr_title="t", pr_body="b", focus=None, evaluation=True)
    bad = req.model_copy(update={"workspace_sha256": "0" * 64})
    r = httpx.post(
        f"{URL}/a2a/{aid}",
        content=a2a_client.jsonrpc_body(bad),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60,
    )
    assert r.status_code == 200
    resp = a2a_client.parse_response(r.json())
    assert (resp.status, resp.error_class) == ("error", "ArchiveError")
    assert "does not match" in (resp.error or "") and resp.counts.model_calls == 0
    json.dumps(resp.model_dump(mode="json"))


def test_service_pid_file_and_log_are_not_committed():
    out = subprocess.run(
        ["git", "check-ignore", "a2a_security/service.pid", "a2a_security/service.log",
         "quota_state.a2a.json"],
        capture_output=True, text=True, check=False, cwd=svc.REPO,
    )  # fmt: skip
    assert len(out.stdout.split()) == 3, out.stdout or sys.argv
