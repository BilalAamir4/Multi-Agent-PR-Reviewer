"""Regression test for the double-counted calls and tokens (decision 0011, Task 0).

Real ``create_agent`` graphs with deterministic fake chat models; the specialists run through the
real tool path (``SpecialistCollector`` -> ``run_specialist`` with ``parent_config``) on the real
``ToolNode``, three tool calls in one turn, in parallel threads. LangSmith tracing is switched on
with a client that makes no network call: with tracing on, a model call's ``parent_run_id`` is the
id of a LangSmith run-tree node that no callback announced, which broke the parent-link counting
of decision 0009 and counted every specialist model call in the orchestrator's own total too.
"""

from __future__ import annotations

import json
import threading
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import Client
from langsmith.run_helpers import tracing_context

from reviewer.agents import orchestrator as orch
from reviewer.agents.specialists import run_specialist
from reviewer.config import load_models_config
from reviewer.models import QuotaRouter, QuotaTracker
from tests.dev_pr import BODY, TITLE, make_dev_workspace
from tests.test_single_agent import USAGE, FakeTime, ai, answer, tool_call

SPECIALISTS = ("bug_reviewer", "security_reviewer", "test_reviewer")


class RoleModel(BaseChatModel):
    """Answers by role, so the order of the parallel specialists does not matter: the orchestrator
    lists the files, calls the three specialists in one turn and answers; a specialist lists the
    files once and answers. Every specialist waits for the other two on its first call, which
    proves the three run at the same time."""

    barrier: threading.Barrier
    threads: set[int] = set()

    @property
    def _llm_type(self) -> str:
        return "role-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tools_seen = sum(1 for m in messages if m.type == "tool")
        if "lead reviewer" in messages[0].content:
            if not any(m.type == "ai" for m in messages):
                msg = ai(tool_call("list_changed_files"))
            elif tools_seen == 1:
                msg = ai(*[tool_call(f"call_{n}", focus="look") for n in SPECIALISTS])
            else:
                plan = [{"specialist": n, "called": True, "reason": "r"} for n in SPECIALISTS]
                msg = ai(tool_call("OrchestratorOutput", plan=plan, summary="ok"))
        elif tools_seen == 0:
            self.threads.add(threading.get_ident())
            self.barrier.wait(timeout=20)
            msg = ai(tool_call("list_changed_files"))
        else:
            msg = answer([])
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def fake_tracing():
    """Tracing enabled, no network: the client's run posts do nothing."""
    client = Client(api_key="test", api_url="http://127.0.0.1:9", auto_batch_tracing=False)
    with (
        mock.patch.object(Client, "create_run"),
        mock.patch.object(Client, "update_run"),
        tracing_context(enabled=True, client=client),
    ):
        yield


def run_real_flow(tmp_path):
    ctx = make_dev_workspace(tmp_path)
    ft = FakeTime()
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota_state.json", providers)
    router = QuotaRouter(
        config, evaluation=True, tracker=tracker, limiters={}, clock=ft.clock, sleep=ft.sleep
    )
    model = RoleModel(barrier=threading.Barrier(3), threads=set())
    router.build = lambda model_id: model

    def runner(name, focus, tool_config):
        return run_specialist(
            name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=router,
            evaluation=True, clock=ft.clock, parent_config=tool_config,
        )  # fmt: skip

    res = orch.orchestrate(ctx, TITLE, BODY, router, clock=ft.clock, runner=runner)
    return res, model


@pytest.mark.parametrize("tracing", [False, True], ids=["tracing_off", "tracing_on"])
def test_parallel_real_specialists_are_counted_once(tmp_path, request, tracing):
    if tracing:
        request.getfixturevalue("fake_tracing")
    res, model = run_real_flow(tmp_path)
    assert res.status == "complete", (res.error_class, res.error, res.notes)
    assert len(model.threads) == 3  # the three specialists ran in three threads at once
    # The orchestrator alone: 3 model calls (list files, call the specialists, answer).
    assert (res.own_model_calls, res.own_tool_calls) == (3, 4)
    assert res.own_input_tokens == 3 * USAGE["input_tokens"]
    assert res.own_output_tokens == 3 * USAGE["output_tokens"]
    assert res.own_tool_calls_by_tool == {
        "call_bug_reviewer": 1,
        "call_security_reviewer": 1,
        "call_test_reviewer": 1,
        "list_changed_files": 1,
    }
    # Each specialist: its own two model calls and one tool call, nothing of the others'.
    assert [c.agent for c in res.children] == list(SPECIALISTS)
    for c in res.children:
        assert (c.model_calls, c.tool_calls) == (2, 1)
        assert c.input_tokens == 2 * USAGE["input_tokens"]
        assert c.output_tokens == 2 * USAGE["output_tokens"]
    # Totals are own plus children, nothing twice.
    assert res.model_calls == 3 + 3 * 2
    assert res.tool_calls == 4 + 3 * 1
    assert res.input_tokens == 9 * USAGE["input_tokens"]
    assert res.output_tokens == 9 * USAGE["output_tokens"]


def test_eval_run_files_and_summary_use_the_corrected_totals(tmp_path, monkeypatch, fake_tracing):
    """``eval.run`` writes what the result holds: totals are own plus children, once."""
    from eval import run as runner_mod

    ctx = make_dev_workspace(tmp_path)
    ft = FakeTime()
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    router = QuotaRouter(
        config, evaluation=True, tracker=QuotaTracker(tmp_path / "q.json", providers),
        limiters={}, clock=ft.clock, sleep=ft.sleep,
    )  # fmt: skip
    model = RoleModel(barrier=threading.Barrier(3), threads=set())
    router.build = lambda model_id: model
    monkeypatch.setattr(runner_mod, "item_inputs", lambda item_id: (ctx, TITLE, BODY))
    out = tmp_path / "results"
    result = runner_mod.run_item("item_001", "orchestrator", router, out, "test")
    assert (result.own_model_calls, result.model_calls) == (3, 9)
    data = json.loads((out / "runs" / "item_001.json").read_bytes())
    assert data["result"]["model_calls"] == 9 and data["result"]["own_model_calls"] == 3
    assert data["result"]["input_tokens"] == 9 * USAGE["input_tokens"]
    entry = data["run_log"][-1]
    assert (entry["model_calls"], entry["own_model_calls"]) == (9, 3)
    assert (entry["input_tokens"], entry["own_input_tokens"]) == (9000, 3000)
    assert (entry["output_tokens"], entry["own_output_tokens"]) == (450, 150)
    assert data["security_mode"] == entry["security_mode"] == "local"
    assert entry["security_transport"] == "local" and entry["security_transport_detail"] is None
