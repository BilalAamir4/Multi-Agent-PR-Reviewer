"""Searching the workspace with ripgrep (stage 1A, D2a: runs on the host, only reads)."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess

from reviewer.tools.context import ReviewContext

SEARCH_TIMEOUT_S = 20
MAX_MATCHES = 50
MAX_PER_FILE = 20
MAX_COLUMNS = 300


def _rg_text(obj: dict) -> str:
    if "text" in obj:
        return obj["text"]
    return base64.b64decode(obj.get("bytes", "")).decode("utf-8", errors="replace")


def _relative(path_text: str, root: str) -> str:
    rel = path_text[len(root) :] if path_text.startswith(root) else path_text
    return rel.replace("\\", "/").lstrip("/")


def search_code(
    ctx: ReviewContext, pattern: str, glob: str | None = None, fixed_string: bool = False
) -> str:
    """Matches of ``pattern`` in the workspace as ``path:line: text``, at most 20 per file and
    50 in total. Returns ``ERROR: ...`` when ripgrep is missing, times out or rejects the
    pattern."""
    if not pattern:
        return "ERROR: pattern must not be empty"
    rg = shutil.which("rg")
    if rg is None:
        return "ERROR: ripgrep not installed"
    root = str(ctx.workspace)
    cmd = [
        rg,
        "--json",
        "--no-config",
        "--max-count",
        str(MAX_PER_FILE),
        "--max-columns",
        str(MAX_COLUMNS),
    ]
    if fixed_string:
        cmd.append("-F")
    if glob:
        cmd += ["--glob", glob]
    cmd += ["--", pattern, root]  # "--": a pattern starting with "-" is not read as a flag
    try:
        res = subprocess.run(
            cmd, capture_output=True, shell=False, timeout=SEARCH_TIMEOUT_S, cwd=ctx.workspace
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: search timed out after {SEARCH_TIMEOUT_S} seconds; narrow the pattern"
    matches: list[str] = []
    more = False
    for raw in res.stdout.splitlines():
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if event.get("type") != "match":
            continue
        if len(matches) >= MAX_MATCHES:
            more = True
            break
        data = event["data"]
        path = _relative(_rg_text(data["path"]), root)
        text = _rg_text(data["lines"]).rstrip("\r\n")
        if len(text) > MAX_COLUMNS:
            text = text[:MAX_COLUMNS] + " [line cut]"
        matches.append(f"{path}:{data['line_number']}: {text}")
    if res.returncode == 2 and not matches:
        detail = res.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return f"ERROR: ripgrep: {detail[-1] if detail else 'failed'}"
    if not matches:
        return "No matches."
    if more:
        matches.append(f"[stopped at {MAX_MATCHES} matches; narrow the pattern or the glob]")
    return "\n".join(matches)
