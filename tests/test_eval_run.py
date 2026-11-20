"""eval/run.py on the synthetic scorer fixture dataset, with ``review`` and ``item_inputs``
replaced by fakes. No models, no Docker, no git."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval import run as runner
from reviewer.agents.output import Dropped
from reviewer.agents.single import ReviewResult
from reviewer.config import AgentLimits
from reviewer.schemas import Finding

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIMITS = AgentLimits(max_model_calls=25, max_tool_calls=40)


def finding(**kw) -> Finding:
    base = {
        "file": "pkg/util.py",
        "line_start": 2,
        "line_end": 2,
        "category": "off_by_one",
        "severity": "high",
        "confidence": 0.9,
        "title": "t",
        "explanation": "e",
        "evidence": "x",
        "agent": "single",
        "model": "gemini-3.1-flash-lite",
    }
    base.update(kw)
    return Finding(**base)


def result(status: str, findings=(), **kw) -> ReviewResult:
    return ReviewResult(
        status=status,
        findings=list(findings),
        prompt_version="abc123def456",
        limits=LIMITS,
        evaluation=True,
        model_id="gemini-3.1-flash-lite",
        model_calls=4,
        tool_calls=3,
        input_tokens=2000,
        output_tokens=100,
        duration_s=12.0,
        wait_s=2.0,
        net_duration_s=10.0,
        **kw,
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "dataset", dataset)
    results = tmp_path / "results"
    calls: list[str] = []
    outcomes: dict[str, ReviewResult] = {}

    def fake_inputs(item_id):
        return object(), "title", "body"

    def fake_review(ctx, title, body, router, item_id=None):
        calls.append(item_id)
        return outcomes[item_id]

    monkeypatch.setattr(runner, "item_inputs", fake_inputs)
    monkeypatch.setattr(runner, "review", fake_review)
    monkeypatch.setattr(runner, "QuotaRouter", lambda evaluation: None)
    monkeypatch.setattr(runner, "load_dotenv", lambda *a, **kw: None)
    argv = ["--config", "single", "--dataset", str(dataset), "--results", str(results)]
    return argv, results / "single", calls, outcomes


def test_full_run_scores_and_summarizes(env, capsys):
    argv, out, calls, outcomes = env
    dropped = Dropped(draft={"file": "x"}, reason="evidence", detail="d")
    outcomes["item_001"] = result("complete", [finding()], dropped=[dropped])
    outcomes["item_002"] = result("limit_reached", limit="max_tool_calls")
    assert runner.main(argv) == 0
    assert calls == ["item_001", "item_002"]
    scored = json.loads((out / "item_001.json").read_bytes())
    assert scored["item_id"] == "item_001" and scored["config"] == "single"
    assert len(scored["findings"]) == 1
    assert json.loads((out / "item_002.json").read_bytes())["findings"] == []
    run_file = json.loads((out / "runs" / "item_001.json").read_bytes())
    assert run_file["result"]["status"] == "complete"
    assert [r["reason"] for r in run_file["run_log"]] == ["first run"]
    report = json.loads((out / "score.json").read_bytes())
    assert report["aggregates"]["strict"]["all"]["tp"] == 1
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| strict | all |" in summary and "| location_only | large |" in summary
    assert "| evidence | 1 |" in summary
    assert "median wait beside it: 2.0 s" in summary
    # (2000 * 0.25 + 100 * 1.50) / 1e6
    assert "$0.00065 at $0.25 / $1.50" in summary


def test_runs_subfolder_is_not_read_by_the_scorer(env):
    argv, out, _, outcomes = env
    outcomes["item_001"] = result("complete", [finding()])
    outcomes["item_002"] = result("complete")
    assert runner.main(argv) == 0
    # A stray runs/item_003.json (not in the dataset) would be an error if the scorer read it.
    (out / "runs" / "item_003.json").write_bytes((out / "runs" / "item_001.json").read_bytes())
    report = runner.score_config(Path(argv[3]), out)
    assert [it["item_id"] for it in report["items"]] == ["item_001", "item_002"]


def test_resume_skips_scored_items_and_force_reruns(env):
    argv, out, calls, outcomes = env
    outcomes["item_001"] = result("complete")
    outcomes["item_002"] = result("error", error_class="RetriesExhausted", error="x")
    assert runner.main(argv) == 1
    assert not (out / "item_002.json").exists()
    assert not (out / "summary.md").exists()
    outcomes["item_002"] = result("complete")
    assert runner.main([*argv, "--reason", "RetriesExhausted on the first run"]) == 0
    assert calls == ["item_001", "item_002", "item_002"]
    log = json.loads((out / "runs" / "item_002.json").read_bytes())["run_log"]
    assert [(r["status"], r["reason"]) for r in log] == [
        ("error", "first run"),
        ("complete", "RetriesExhausted on the first run"),
    ]
    assert runner.main([*argv, "--items", "item_001", "--force"]) == 0
    assert calls[-1] == "item_001"
    log = json.loads((out / "runs" / "item_001.json").read_bytes())["run_log"]
    assert [r["reason"] for r in log] == ["first run", "re-run (--force)"]


def test_quota_paused_stops_with_exit_3(env, capsys):
    argv, out, calls, outcomes = env
    outcomes["item_001"] = result("complete")
    reset = datetime(2026, 11, 22, 8, tzinfo=UTC)
    outcomes["item_002"] = result("quota_paused", reset_at=reset)
    assert runner.main(argv) == 3
    assert (out / "item_001.json").is_file()
    assert not (out / "item_002.json").exists()
    assert (out / "runs" / "item_002.json").is_file()
    assert "2026-11-22 13:00 PKT" in capsys.readouterr().out
    # Resuming leaves the completed item alone.
    outcomes["item_002"] = result("complete")
    assert runner.main(argv) == 0
    assert calls == ["item_001", "item_002", "item_002"]


def test_unknown_item_rejected(env):
    argv, *_ = env
    assert runner.main([*argv, "--items", "item_999"]) == 2
