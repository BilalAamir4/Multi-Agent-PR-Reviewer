"""Integration: one real review of the synthetic dev fixture in evaluation mode (stage 1B).

Spends real calls to the evaluation model (at most ``limits.single.max_model_calls``). It does
not assert that the fixture's two defects are found; whether they were is printed and recorded
as test properties, for information only.
"""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

from reviewer.agents.single import ReviewResult, review
from reviewer.config import repo_root
from reviewer.models import QuotaRouter
from reviewer.schemas import Finding
from tests.dev_pr import BODY, TITLE, expected_defects, make_dev_workspace

pytestmark = pytest.mark.integration


def _found(defect: dict, findings: list[Finding]) -> bool:
    return any(
        f.file == defect["file"]
        and f.line_start <= defect["line_end"] + 3
        and f.line_end >= defect["line_start"] - 3
        for f in findings
    )


def test_review_dev_fixture(tmp_path, record_property):
    load_dotenv(repo_root() / ".env")
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        pytest.skip("GOOGLE_API_KEY missing from .env")
    ctx = make_dev_workspace(tmp_path)
    result = review(ctx, TITLE, BODY, QuotaRouter(evaluation=True))
    if result.status == "quota_paused":
        pytest.skip(f"evaluation model quota paused until {result.reset_at}")
    assert result.status == "complete", f"{result.status}: {result.error_class}: {result.error}"
    for f in result.findings:
        assert isinstance(f, Finding)
        Finding.model_validate(f.model_dump())
    text = result.model_dump_json()
    assert ReviewResult.model_validate_json(text) == result
    assert json.loads(text)["status"] == "complete"

    found = {
        f"{d['file']}:{d['line_start']}": _found(d, result.findings) for d in expected_defects()
    }
    info = {
        "found": found,
        "findings": [
            f"{f.file}:{f.line_start}-{f.line_end} {f.category} {f.confidence}: {f.title}"
            for f in result.findings
        ],
        "dropped": [f"{d.reason}: {d.detail}" for d in result.dropped],
        "model_calls": result.model_calls,
        "tool_calls": result.tool_calls_by_tool,
        "tokens": [result.input_tokens, result.output_tokens],
        "net_s": result.net_duration_s,
        "wait_s": result.wait_s,
        "langsmith_run_id": result.langsmith_run_id,
    }
    for k, v in info.items():
        record_property(k, v)
    print("\nDEV FIXTURE (information only):", json.dumps(info, indent=2))
