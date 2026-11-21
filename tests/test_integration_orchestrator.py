"""Integration (stage 2B): the routing test, the done criterion of stage 2 (decision 0009).

Real calls to the evaluation model (evaluation mode, one pinned model), real ``run_tests`` in the
``dev_fixture`` sandbox image. Only the synthetic dev fixtures are used; never an evaluation item.

- ``dev_docs_pr`` (README and docs only): the orchestrator must not call the security reviewer
  (calling no specialist at all is acceptable and expected).
- ``dev_risky_pr`` (SQL, a credential, logic, tests): the orchestrator must call all three
  specialists and the merged review must be ``complete``.
- ``dev_logic_pr``: informational only, nothing is asserted about routing.

Set ``DEV_RUN_DIR`` to also save each run's full ``OrchestratorResult`` as JSON.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from dotenv import load_dotenv

from eval.repos import get_repo, sandbox_spec
from reviewer.agents.orchestrator import OrchestratorResult, orchestrate
from reviewer.config import repo_root
from reviewer.models import QuotaRouter
from reviewer.sandbox import SandboxUnavailable, get_client
from tests.dev_pr import make_fixture_workspace, pr_text

pytestmark = pytest.mark.integration


def _require_image() -> None:
    image = get_repo("dev_fixture").image
    try:
        client = get_client()
    except SandboxUnavailable as e:
        pytest.skip(str(e))
    try:
        client.images.get(image)
    except Exception:
        pytest.skip(f"image {image} not built: docker build -t {image} sandbox_images/dev_fixture")
    finally:
        client.close()


def _run(fixture: str, tmp_path, record_property) -> OrchestratorResult:
    load_dotenv(repo_root() / ".env")
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        pytest.skip("GOOGLE_API_KEY missing from .env")
    _require_image()
    ctx = make_fixture_workspace(fixture, tmp_path, sandbox_spec("dev_fixture"))
    title, body = pr_text(fixture)
    result = orchestrate(ctx, title, body, QuotaRouter(evaluation=True))
    if result.status == "quota_paused":
        pytest.skip(f"evaluation model quota paused until {result.reset_at}")
    info = {
        "fixture": fixture,
        "status": result.status,
        "called": [c.agent for c in result.children],
        "plan": {e.specialist: [e.called, e.reason] for e in result.plan},
        "findings": [
            f"{f.agent}: {f.file}:{f.line_start}-{f.line_end} {f.category} "
            f"{f.confidence}: {f.title}"
            for f in result.findings
        ],
        "children": {c.agent: [c.status, len(c.findings), c.notes] for c in result.children},
        "notes": result.notes,
        "dropped": [f"{d.reason}: {d.detail}" for d in result.dropped],
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls,
        "own": [result.own_model_calls, result.own_tool_calls],
        "tokens": [result.input_tokens, result.output_tokens],
        "wait_s_wall": result.wait_s_wall,
        "wait_s_sum": result.wait_s_sum,
        "duration_s": result.duration_s,
        "net_s": result.net_duration_s,
        "prompt_version": result.prompt_version,
        "langsmith_run_id": result.langsmith_run_id,
    }
    for k, v in info.items():
        record_property(k, v)
    print(f"\nORCHESTRATOR RUN {fixture}:", json.dumps(info, indent=2))
    out_dir = os.environ.get("DEV_RUN_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{fixture}-{int(time.time())}.json")
        with open(path, "wb") as fh:
            fh.write(result.model_dump_json(indent=2).encode("utf-8"))
    return result


def test_docs_only_change_skips_the_security_reviewer(tmp_path, record_property):
    result = _run("dev_docs_pr", tmp_path, record_property)
    called = {c.agent for c in result.children}
    assert "security_reviewer" not in called, result.plan
    assert result.status == "complete", f"{result.status}: {result.error_class}: {result.error}"
    assert not [n for n in result.notes if n.startswith("plan_mismatch")], result.notes


def test_risky_change_calls_all_three_specialists(tmp_path, record_property):
    result = _run("dev_risky_pr", tmp_path, record_property)
    assert {c.agent for c in result.children} == {
        "bug_reviewer",
        "security_reviewer",
        "test_reviewer",
    }, result.plan
    assert result.status == "complete", (
        f"{result.status}: {result.error_class}: {result.error}; {result.notes}"
    )
    assert not [n for n in result.notes if n.startswith("plan_mismatch")], result.notes
    assert all(f.agent in {c.agent for c in result.children} for f in result.findings)


def test_logic_only_change_informational(tmp_path, record_property):
    """Records whether the security reviewer was skipped; asserts nothing about routing."""
    result = _run("dev_logic_pr", tmp_path, record_property)
    record_property(
        "security_skipped", "security_reviewer" not in {c.agent for c in result.children}
    )
    assert result.status in ("complete", "partial"), result.error
