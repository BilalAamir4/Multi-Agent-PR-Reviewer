"""Check tool calling on the stage 0 models with one ``create_agent`` run each.

Each run makes at most 2 model requests (the tool call, then the answer): SDK retries are off and
``ModelCallLimitMiddleware`` caps the run. Only synthetic content is sent.

    uv run python scripts/check_tool_calls.py                      # all five models
    uv run python scripts/check_tool_calls.py gemma-4-31b-it       # a subset, by exact ID
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reviewer.config import load_models_config  # noqa: E402

# Gemma 4 31B, Gemma 4 26B, Gemini 4 Hydrogen, Gemini 3.1 Flash Lite, Groq gpt-oss-120b
MODELS = [
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-4-hydrogen",
    "gemini-3.1-flash-lite",
    "openai/gpt-oss-120b",
]
KEY_FOR = {"google_genai": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}

SYNTHETIC_FILE = """\
import math

RATE = 0.175


def price_with_tax(amount):
    total = amount * (1 + RATE)
    return math.ceil(total * 100) / 100
""".splitlines()
LINE = 7
EXPECTED = SYNTHETIC_FILE[LINE - 1].strip()
QUESTION = (
    f"What is on line {LINE} of the file? Use the get_line tool. "
    "Answer with the exact text of the line."
)


@tool
def get_line(n: int) -> str:
    """Return line n (1-based) of the file under review."""
    if not 1 <= n <= len(SYNTHETIC_FILE):
        return f"error: line {n} is out of range 1..{len(SYNTHETIC_FILE)}"
    return SYNTHETIC_FILE[n - 1]


@dataclass
class ToolCheck:
    model: str
    provider: str
    tool_called: bool = False
    args_correct: bool = False
    answer_correct: bool = False
    args: object = None
    answer: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.tool_called and self.args_correct and self.answer_correct and not self.error


def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    parts = []
    for block in msg.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def check_model(model_id: str, provider: str, model_calls: int = 2) -> ToolCheck:
    res = ToolCheck(model=model_id, provider=provider)
    try:
        llm = init_chat_model(model_id, model_provider=provider, temperature=0, max_retries=0)
        agent = create_agent(
            llm,
            tools=[get_line],
            middleware=[ModelCallLimitMiddleware(run_limit=model_calls, exit_behavior="end")],
            name=f"tool_check_{model_id.replace('/', '_')}",
        )
        out = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    except Exception as e:
        res.error = f"{type(e).__module__}.{type(e).__qualname__}: {e}"
        return res

    messages = out["messages"]
    calls = [c for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    calls = [c for c in calls if c["name"] == "get_line"]
    res.tool_called = bool(calls)
    if calls:
        res.args = calls[0]["args"]
        res.args_correct = res.args.get("n") == LINE
    finals = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    res.answer = _text(finals[-1]).strip() if finals else ""
    res.answer_correct = EXPECTED in res.answer
    return res


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="exact model IDs (default: all five)")
    parser.add_argument(
        "--model-calls",
        type=int,
        default=2,
        help="cap on model requests per run; 1 checks only the tool call and its arguments",
    )
    args = parser.parse_args(argv)
    load_dotenv(REPO / ".env")
    cfg = load_models_config()
    wanted = args.models or MODELS
    results = []
    for model_id in wanted:
        spec = cfg.models[model_id]
        if not os.environ.get(KEY_FOR[spec.provider], "").strip():
            print(f"{model_id}: skipped, {KEY_FOR[spec.provider]} missing from .env")
            continue
        r = check_model(model_id, spec.provider, args.model_calls)
        results.append(r)
        print(
            f"{model_id} ({spec.provider}): tool_call={r.tool_called} args_correct={r.args_correct}"
            f" answer_correct={r.answer_correct} args={r.args!r}"
        )
        print(f"    answer: {r.answer[:200]!r}")
        if r.error:
            print(f"    error: {r.error[:1500]}")
    print(f"expected line {LINE}: {EXPECTED!r}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
