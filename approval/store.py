"""Where stage 6 keeps its files (``.reviewer/``, gitignored): the SQLite checkpoints, the
decision progress of each review, and the review workspaces."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from reviewer.config import repo_root


def default_state_dir() -> Path:
    return repo_root() / ".reviewer"


def safe_name(review_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", review_key)


def open_checkpointer(state_dir: Path) -> SqliteSaver:
    state_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_dir / "checkpoints.sqlite", check_same_thread=False)
    return SqliteSaver(conn)


def close_checkpointer(saver: SqliteSaver) -> None:
    saver.conn.close()


def workspace_dir(state_dir: Path, review_key: str) -> Path:
    return state_dir / "workspaces" / safe_name(review_key)


def progress_path(state_dir: Path, review_key: str) -> Path:
    return state_dir / "decisions" / f"{safe_name(review_key)}.json"


def load_progress(state_dir: Path, review_key: str) -> dict[str, dict[str, Any]]:
    """Decisions made so far, by finding id."""
    path = progress_path(state_dir, review_key)
    if not path.is_file():
        return {}
    return json.loads(path.read_bytes().decode("utf-8")).get("decisions", {})


def save_progress(state_dir: Path, review_key: str, decisions: dict[str, dict[str, Any]]) -> None:
    path = progress_path(state_dir, review_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"review_key": review_key, "decisions": decisions}
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
