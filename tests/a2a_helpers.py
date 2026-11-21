"""Helpers for the stage 3 transport-parity tests (decision 0011): a scripted security review of
``dev_security_pr`` and the comparison that ignores what legitimately differs between a local and
a remote run (the service block, timings, waits, process ids)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reviewer.tools import ReviewContext
from tests.dev_pr import expected_defects, pr_text

FIXTURE = "dev_security_pr"
FOCUS = "the new search query and the admin login"
TIMING_COUNTS = ("duration_s", "wait_s", "net_duration_s")


def finding(ctx: ReviewContext, index: int, **over: Any) -> dict[str, Any]:
    """A valid finding at the fixture's ``index``-th expected defect, evidence from the file."""
    d = expected_defects(FIXTURE)[index]
    lines = (ctx.workspace / d["file"]).read_text(encoding="utf-8").splitlines()
    out = {
        "file": d["file"],
        "line_start": d["line_start"],
        "line_end": d["line_end"],
        "category": d["kind"],
        "severity": "high",
        "confidence": 0.9 - 0.1 * index,
        "title": f"Defect {index}",
        "explanation": "Why it is wrong.",
        "evidence": "\n".join(lines[d["line_start"] - 1 : d["line_end"]]),
    }
    out.update(over)
    return out


def security_script(ctx: ReviewContext) -> list[dict[str, Any]]:
    """The model's answers, as ``reviewer.test_hooks`` reads them: list the files, read the diff,
    read a file, then answer with two findings and one that is not in the diff."""
    out_of_diff = finding(ctx, 0, file="vault/nowhere.py", title="Not in the diff")
    answer = {
        "summary": "Adds a search query and an admin login.",
        "findings": [finding(ctx, 0), finding(ctx, 1), out_of_diff],
    }
    return [
        {"tool_calls": [{"name": "list_changed_files", "args": {}, "id": "c1"}]},
        {"tool_calls": [{"name": "read_diff", "args": {"path": "vault/db.py"}, "id": "c2"}]},
        {"tool_calls": [{"name": "read_file", "args": {"path": "vault/admin.py"}, "id": "c3"}]},
        {"tool_calls": [{"name": "ReviewOutput", "args": answer, "id": "c4"}]},
    ]


def write_script(ctx: ReviewContext, path: Path) -> Path:
    path.write_bytes(json.dumps(security_script(ctx)).encode("utf-8"))
    return path


def request_for(ctx: ReviewContext) -> dict[str, Any]:
    title, body = pr_text(FIXTURE)
    return {
        "workspace": str(ctx.workspace),
        "diff_text": ctx.diff_text,
        "pr_title": title,
        "pr_body": body,
        "focus": FOCUS,
        "evaluation": True,
        "spawn_enabled": False,
    }


TIMING_KEYS = frozenset({
    "duration_s", "wait_s", "net_duration_s", "wait_s_sum", "wait_s_wall", "start_s", "end_s",
    "wall_s", "net_s", "waits_by_reason", "wait_intervals", "langsmith_run_id",
})  # fmt: skip


def _strip(value: Any) -> Any:
    """``value`` without the timing and wait fields at any depth (children, spawn records)."""
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def comparable(response: dict[str, Any]) -> dict[str, Any]:
    """``response`` without the service block, the timings, the waits and the PIDs, at any depth."""
    out = {k: v for k, v in response.items() if k not in ("service", "transport")}
    out = _strip(json.loads(json.dumps(out)))  # a copy, plain JSON
    for key in TIMING_COUNTS:
        out["counts"].pop(key, None)
    return out


def spawn_script(ctx: ReviewContext, subagents: int = 2) -> list[dict[str, Any]]:
    """Scripted answers for a spawning security review run one sub-agent at a time: each sub-agent
    lists the files and answers (sub-agent ``k`` reports the ``k``-th expected defect), then the
    specialist's cross-file pass answers with nothing."""
    out: list[dict[str, Any]] = []
    for k in range(subagents):
        answer = {"summary": f"batch {k + 1}", "findings": [finding(ctx, k)]}
        out.append({"tool_calls": [{"name": "list_changed_files", "args": {}, "id": f"s{k}a"}]})
        out.append({"tool_calls": [{"name": "ReviewOutput", "args": answer, "id": f"s{k}b"}]})
    cross = {"summary": "cross-file pass", "findings": []}
    out.append({"tool_calls": [{"name": "ReviewOutput", "args": cross, "id": "x1"}]})
    return out
