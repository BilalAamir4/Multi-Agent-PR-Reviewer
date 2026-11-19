"""Check that ``langgraph dev`` runs on Windows and serves the echo graph over HTTP and A2A (D1).

Starts ``langgraph dev`` in the background, checks /ok, the assistant listing and an A2A
``message/send`` call, then stops the server and checks that nothing listens on the port.
No model calls are made.

    uv run python scripts/check_langgraph.py
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "scripts" / "langgraph_check" / "langgraph.json"
HOST, PORT = "127.0.0.1", 2024
BASE = f"http://{HOST}:{PORT}"
ECHO_TEXT = "ping from check_langgraph"
READY_TIMEOUT_S = 180


@dataclass
class LanggraphCheck:
    ok_status: int | None = None
    assistant_id: str | None = None
    a2a_request: dict | None = None
    a2a_status: int | None = None
    a2a_response: object = None
    a2a_text: str = ""
    a2a_ok: bool = False
    port_free_after: bool = False
    errors: list[str] = field(default_factory=list)
    log_tail: str = ""


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, PORT)) == 0


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            # uv -> langgraph -> uvicorn (+ reloader): kill the whole tree.
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _a2a_texts(obj: object) -> list[str]:
    """Collect every text part found anywhere in an A2A JSON-RPC result."""
    found: list[str] = []
    if isinstance(obj, dict):
        if obj.get("kind") == "text" and isinstance(obj.get("text"), str):
            found.append(obj["text"])
        if "text" in obj and isinstance(obj.get("text"), str) and "kind" not in obj:
            found.append(obj["text"])
        for v in obj.values():
            found.extend(_a2a_texts(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_a2a_texts(v))
    return found


def _checks(client: httpx.Client, res: LanggraphCheck) -> None:
    r = client.get(f"{BASE}/ok")
    res.ok_status = r.status_code

    r = client.post(f"{BASE}/assistants/search", json={"graph_id": "echo", "limit": 10})
    r.raise_for_status()
    assistants = [a for a in r.json() if a.get("graph_id") == "echo"]
    if not assistants:
        res.errors.append(f"assistants/search returned no 'echo' assistant: {r.text[:500]}")
        return
    res.assistant_id = assistants[0]["assistant_id"]

    res.a2a_request = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": ECHO_TEXT}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    r = client.post(
        f"{BASE}/a2a/{res.assistant_id}",
        json=res.a2a_request,
        headers={"Accept": "application/json"},
    )
    res.a2a_status = r.status_code
    try:
        res.a2a_response = r.json()
    except json.JSONDecodeError:
        res.a2a_response = r.text
    if isinstance(res.a2a_response, dict) and "result" in res.a2a_response:
        # The graph replies "echo: <text>"; the bare input may also appear in the task history.
        expected = f"echo: {ECHO_TEXT}"
        texts = _a2a_texts(res.a2a_response["result"])
        res.a2a_text = next((t for t in texts if expected in t), " | ".join(texts))
        res.a2a_ok = expected in res.a2a_text
    else:
        res.errors.append("A2A response has no 'result'")


def run_check() -> LanggraphCheck:
    res = LanggraphCheck()
    if _port_open():
        res.errors.append(f"port {PORT} is already in use before starting")
        return res
    uv = shutil.which("uv")
    if uv is None:
        res.errors.append("uv is not on PATH")
        return res
    # langgraph dev resolves graph paths ("./echo_graph.py") against the working directory, not
    # the config file's folder, so it runs from the config folder (Windows check, stage 0B).
    cmd = [uv, "run", "--project", str(REPO), "langgraph", "dev", "--config", CONFIG.name]
    cmd += ["--no-browser", "--port", str(PORT)]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "langgraph-dev.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, cwd=CONFIG.parent, stdout=log, stderr=subprocess.STDOUT)
            try:
                with httpx.Client(timeout=30) as client:
                    deadline = time.monotonic() + READY_TIMEOUT_S
                    ready = False
                    while time.monotonic() < deadline and proc.poll() is None:
                        try:
                            if client.get(f"{BASE}/ok", timeout=2).status_code == 200:
                                ready = True
                                break
                        except httpx.TransportError:
                            pass
                        log_text = log_path.read_text(encoding="utf-8", errors="replace")
                        if "Application startup failed" in log_text:
                            break  # the reloader keeps the process alive; stop waiting
                        time.sleep(1)
                    if ready:
                        _checks(client, res)
                    else:
                        res.errors.append(
                            f"server not ready after {READY_TIMEOUT_S}s (exit code {proc.poll()})"
                        )
            except Exception as e:
                res.errors.append(f"{type(e).__qualname__}: {e}")
            finally:
                _stop(proc)
        res.log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    for _ in range(10):
        if not _port_open():
            res.port_free_after = True
            break
        time.sleep(1)
    return res


def main() -> int:
    res = run_check()
    print(f"{'PASS' if res.ok_status == 200 else 'FAIL'} GET /ok -> {res.ok_status}")
    print(
        f"{'PASS' if res.assistant_id else 'FAIL'} assistants/search lists 'echo': "
        f"assistant_id={res.assistant_id}"
    )
    print(f"{'PASS' if res.a2a_ok else 'FAIL'} A2A message/send -> HTTP {res.a2a_status}")
    print(f"    echoed text: {res.a2a_text!r}")
    if not res.a2a_ok:
        print(f"    request: {json.dumps(res.a2a_request)}")
        print(f"    response: {json.dumps(res.a2a_response, default=str)[:3000]}")
    print(f"{'PASS' if res.port_free_after else 'FAIL'} nothing listening on {PORT} after stop")
    for e in res.errors:
        print(f"error: {e}")
    if res.errors or not res.a2a_ok:
        print("---- server log tail ----")
        print(res.log_tail)
    ok = res.ok_status == 200 and res.assistant_id and res.a2a_ok and res.port_free_after
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
