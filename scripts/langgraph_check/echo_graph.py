"""Minimal graph for checking ``langgraph dev`` on Windows. It echoes the last user message and
makes no model calls."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, MessagesState, StateGraph


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def echo(state: MessagesState) -> dict:
    last = state["messages"][-1] if state["messages"] else None
    text = _text(last) if last is not None else ""
    return {"messages": [AIMessage(content=f"echo: {text}")]}


builder = StateGraph(MessagesState)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile()
