"""Rule 1 of stage 2A: the single agent is frozen. Refactoring ``single.py`` onto the shared
machinery in ``agents/base.py`` must not change what it builds.

``FINGERPRINT`` was computed by ``single_fingerprint`` against the stage 1B code (commit
1a869be, before the refactor). The test captures everything ``review`` passes to
``create_agent`` and to ``agent.invoke`` (system prompt, tools with their names, descriptions
and argument schemas, response format, middleware and their limits, agent name, first message,
run name, tags, metadata keys and recursion limit) and checks the digest still matches.
No model, network or Docker call is made.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.utils.function_calling import convert_to_openai_tool

from reviewer.agents import single
from reviewer.agents.output import ReviewOutput
from reviewer.config import load_models_config
from reviewer.models import QuotaRouter, QuotaTracker
from tests.dev_pr import BODY, TITLE, make_dev_workspace

# SHA-256 of src/reviewer/prompts/single.md at 1a869be (prompt_version aa94f1ae2cf6).
SINGLE_MD_SHA256 = "aa94f1ae2cf6"
FINGERPRINT = "ffe5ab2eda9ff536013141f3b6c9eba4d82fe9a30be6fa119e22c806713d1ffd"


class _Captured:
    def __init__(self) -> None:
        self.create: dict[str, Any] = {}
        self.invoke: list[tuple[Any, dict[str, Any]]] = []


def _fake_create_agent(captured: _Captured):
    def create_agent(model, **kwargs):
        captured.create = kwargs

        class _Agent:
            def invoke(self, inputs, config=None):
                captured.invoke.append((inputs, config or {}))
                return {"structured_response": ReviewOutput(summary="s", findings=[])}

        return _Agent()

    return create_agent


def _middleware_view(m: Any) -> list[Any]:
    if isinstance(m, ModelCallLimitMiddleware):
        return ["model_limit", m.run_limit, m.thread_limit, m.exit_behavior]
    if isinstance(m, ToolCallLimitMiddleware):
        names = sorted(getattr(m, "tool_names", None) or [])
        return ["tool_limit", m.run_limit, m.thread_limit, m.exit_behavior, m.tool_name, names]
    return ["other", type(m).__name__]


def single_fingerprint(captured: _Captured) -> dict[str, Any]:
    kw = captured.create
    rf = kw["response_format"]
    (inputs, config) = captured.invoke[0]
    return {
        "system_prompt_sha256": hashlib.sha256(kw["system_prompt"].encode("utf-8")).hexdigest(),
        "tools": [convert_to_openai_tool(t) for t in kw["tools"]],
        "response_format": [
            type(rf).__name__,
            rf.schema.__name__,
            rf.schema.model_json_schema(),
        ],
        "middleware": [_middleware_view(m) for m in kw["middleware"]],
        "name": kw.get("name"),
        "other_create_kwargs": sorted(
            k for k in kw if k not in {"tools", "system_prompt", "response_format", "middleware"}
        ),
        "messages": [[type(m).__name__, m.content] for m in inputs["messages"]],
        "run_name": config["run_name"],
        "tags": config["tags"],
        "metadata_keys": sorted(config["metadata"]),
        "metadata_prompt_version": config["metadata"]["prompt_version"],
        "recursion_limit": config["recursion_limit"],
    }


def digest(fp: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode("utf-8")).hexdigest()


def capture_single(monkeypatch, tmp_path: Path) -> _Captured:
    captured = _Captured()
    fake = _fake_create_agent(captured)
    # Before the refactor create_agent was called from single.py, after it from base.py.
    for name in ("reviewer.agents.single", "reviewer.agents.base"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, "create_agent"):
            monkeypatch.setattr(module, "create_agent", fake)
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota_state.json", providers)
    router = QuotaRouter(config, evaluation=True, tracker=tracker, limiters={})
    router.build = lambda model_id: object()
    ctx = make_dev_workspace(tmp_path)
    res = single.review(ctx, TITLE, BODY, router)
    assert res.status == "complete", res
    return captured


def test_single_prompt_file_is_frozen():
    assert single.prompt_version() == SINGLE_MD_SHA256
    data = single.PROMPT_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest()[:12] == SINGLE_MD_SHA256
    assert single.system_prompt() == data.decode("utf-8")


def test_single_agent_builds_the_identical_agent(monkeypatch, tmp_path):
    captured = capture_single(monkeypatch, tmp_path)
    fp = single_fingerprint(captured)
    assert fp["system_prompt_sha256"][:12] == SINGLE_MD_SHA256
    assert [t["function"]["name"] for t in fp["tools"]] == [
        "list_changed_files",
        "read_diff",
        "read_file",
        "search_code",
        "run_ruff",
        "run_bandit",
    ]
    assert fp["middleware"][:2] == [
        ["model_limit", 25, None, "end"],
        [
            "tool_limit",
            40,
            None,
            "end",
            None,
            [
                "list_changed_files",
                "read_diff",
                "read_file",
                "run_bandit",
                "run_ruff",
                "search_code",
            ],
        ],
    ]
    assert digest(fp) == FINGERPRINT, json.dumps(fp, indent=1)[:4000]


if __name__ == "__main__":  # pragma: no cover - used once to record FINGERPRINT
    import sys
    import tempfile

    mp = pytest.MonkeyPatch()
    with tempfile.TemporaryDirectory() as d:
        fp = single_fingerprint(capture_single(mp, Path(d)))
    mp.undo()
    sys.stdout.write(json.dumps(fp, indent=1)[:3000] + "\n" + digest(fp) + "\n")
