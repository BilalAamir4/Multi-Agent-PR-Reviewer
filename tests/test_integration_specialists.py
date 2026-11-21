"""Integration (stage 2A): ``run_tests`` for real in the ``dev_fixture`` image (Docker, no model
call), and each specialist once on its matching dev fixture in evaluation mode (real calls to
the evaluation model, at most ``limits.<name>.max_model_calls`` each).

Never run on the evaluation items: specialists are first run on ``eval/dataset/`` in the stage 5
evaluation. Whether a fixture's expected defects were found is printed and recorded as test
properties, for information only; no test asserts it.
"""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

from eval.repos import get_repo, sandbox_spec
from reviewer.agents.base import ReviewResult
from reviewer.agents.specialists import run_specialist
from reviewer.config import repo_root
from reviewer.models import QuotaRouter
from reviewer.sandbox import SandboxUnavailable, get_client
from reviewer.schemas import Finding
from reviewer.tools import run_tests
from tests.dev_pr import expected_defects, make_fixture_workspace, pr_text

pytestmark = pytest.mark.integration

FIXTURE_OF = {
    "bug_reviewer": "dev_pr",
    "security_reviewer": "dev_security_pr",
    "test_reviewer": "dev_tests_pr",
}


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


def test_run_tests_in_the_dev_fixture_sandbox(tmp_path):
    _require_image()
    ctx = make_fixture_workspace("dev_tests_pr", tmp_path, sandbox_spec("dev_fixture"))
    out = run_tests(ctx, ["tests/test_summary.py"])
    lines = out.splitlines()
    assert lines[0] == lines[0].strip() and lines[0].startswith("pytest exit code 1: "), out
    assert "1 failed, 4 passed" in lines[0], out
    assert lines[1] == "Failed: tests/test_summary.py::test_clamp", out
    assert "assert clamp(5, 0, 10) == 5" in out
    passing = run_tests(ctx, ["tests/test_summary.py::test_median"])
    assert passing.startswith("pytest exit code 0: 1 passed"), passing


def _found(defect: dict, findings: list[Finding]) -> bool:
    return any(
        f.file == defect["file"]
        and f.line_start <= defect["line_end"] + 3
        and f.line_end >= defect["line_start"] - 3
        for f in findings
    )


@pytest.mark.parametrize("name", list(FIXTURE_OF))
def test_specialist_on_its_dev_fixture(name, tmp_path, record_property):
    load_dotenv(repo_root() / ".env")
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        pytest.skip("GOOGLE_API_KEY missing from .env")
    fixture = FIXTURE_OF[name]
    sandbox = None
    if name == "test_reviewer":
        _require_image()
        sandbox = sandbox_spec("dev_fixture")  # as --sandbox-repo dev_fixture
    ctx = make_fixture_workspace(fixture, tmp_path, sandbox)
    title, body = pr_text(fixture)
    result = run_specialist(
        name,
        ctx,
        pr_title=title,
        pr_body=body,
        router=QuotaRouter(evaluation=True),
        evaluation=True,
    )
    if result.status == "quota_paused":
        pytest.skip(f"evaluation model quota paused until {result.reset_at}")
    info = {
        "agent": name,
        "fixture": fixture,
        "status": result.status,
        "found": {
            f"{d['file']}:{d['line_start']} {d['kind']}": _found(d, result.findings)
            for d in expected_defects(fixture)
        },
        "findings": [
            f"{f.file}:{f.line_start}-{f.line_end} {f.category} {f.confidence}: {f.title}"
            for f in result.findings
        ],
        "dropped": [f"{d.reason}: {d.detail}" for d in result.dropped],
        "notes": result.notes,
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls_by_tool,
        "tokens": [result.input_tokens, result.output_tokens],
        "net_s": result.net_duration_s,
        "wait_s": result.wait_s,
        "prompt_version": result.prompt_version,
        "langsmith_run_id": result.langsmith_run_id,
    }
    for k, v in info.items():
        record_property(k, v)
    print(f"\nDEV RUN {name} (information only):", json.dumps(info, indent=2))
    out_dir = os.environ.get("DEV_RUN_DIR")
    if out_dir:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "wb") as fh:
            fh.write(result.model_dump_json(indent=2).encode("utf-8"))

    assert result.status == "complete", f"{result.status}: {result.error_class}: {result.error}"
    assert result.agent == name
    for f in result.findings:
        assert f.agent == name
        Finding.model_validate(f.model_dump())
    assert ReviewResult.model_validate_json(result.model_dump_json()) == result
