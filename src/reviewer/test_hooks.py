"""Test-only switch that lets the A2A service answer with a scripted fake model (stage 3,
decision 0011), so the transport can be compared deterministically without a real model.

``REVIEWER_TEST_FAKE_MODEL`` names a JSON file: a list of AI messages, each ``{"content": "",
"tool_calls": [{"name": ..., "args": {...}, "id": ...}]}``, handed out one per model call. The
variable is read **only** when ``REVIEWER_TEST_MODE=1``; with the variable set and the mode not
``1`` the service refuses to run (``TestHookRejected``), so a stray variable can never replace the
real model in a normal deployment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

TEST_MODE_VAR = "REVIEWER_TEST_MODE"
FAKE_MODEL_VAR = "REVIEWER_TEST_FAKE_MODEL"
USAGE = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}


class TestHookRejected(RuntimeError):
    """``REVIEWER_TEST_FAKE_MODEL`` is set but ``REVIEWER_TEST_MODE`` is not ``1``."""

    __test__ = False  # not a pytest class


def fake_model_script(env: Mapping[str, str] | None = None) -> Path | None:
    """The script file the environment asks for, or None when the hook is not set. Raises
    ``TestHookRejected`` when the hook is set outside test mode."""
    env = os.environ if env is None else env
    value = env.get(FAKE_MODEL_VAR)
    if value is None:
        return None
    if env.get(TEST_MODE_VAR) != "1":
        raise TestHookRejected(
            f"{FAKE_MODEL_VAR} is set but {TEST_MODE_VAR} is not 1: the fake model is test-only"
        )
    return Path(value)


class ScriptedChatModel(BaseChatModel):
    """Hands out ``messages`` one per call, with fixed token usage. ``bind_tools`` returns itself
    (``create_agent`` binds the tools)."""

    messages: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if self.position >= len(self.messages):
            raise RuntimeError("the fake model script has run out of messages")
        msg = self.messages[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def load_script(path: Path) -> list[AIMessage]:
    items = json.loads(Path(path).read_bytes().decode("utf-8"))
    out = []
    for i, item in enumerate(items):
        calls = [
            {"name": c["name"], "args": c["args"], "id": c.get("id", f"call_{i}_{n}"),
             "type": "tool_call"}
            for n, c in enumerate(item.get("tool_calls", []))
        ]  # fmt: skip
        out.append(
            AIMessage(content=item.get("content", ""), tool_calls=calls, usage_metadata=USAGE)
        )
    return out


def script_builder(path: Path) -> Callable[[str], BaseChatModel]:
    """A ``QuotaRouter.build`` replacement: a fresh scripted model, from the start of the file."""

    def build(model_id: str) -> BaseChatModel:
        return ScriptedChatModel(messages=load_script(path))

    return build
