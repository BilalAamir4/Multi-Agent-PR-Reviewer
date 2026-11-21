"""Start or stop the security specialist's A2A service (stage 3, decision 0011).

    uv run python scripts/run_a2a_security.py            # start it on 127.0.0.1:2025
    uv run python scripts/run_a2a_security.py --stop     # stop it

Starting runs ``langgraph dev`` for ``a2a_security/langgraph.json`` in the background (its output
goes to ``a2a_security/service.log``), waits until ``GET /ok`` answers and prints the assistant ID
of the ``security_reviewer`` graph. The service listens on 127.0.0.1 only and has no
authentication. Port 2024 stays free for ``scripts/check_langgraph.py``. The process id is kept
in ``a2a_security/service.pid`` so ``--stop`` can end the whole process tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO / "a2a_security"
CONFIG = "langgraph.json"
PID_FILE = SERVICE_DIR / "service.pid"
LOG_FILE = SERVICE_DIR / "service.log"
HOST = "127.0.0.1"
PORT = 2025
GRAPH = "security_reviewer"
READY_TIMEOUT_S = 180
CREATE_BREAKAWAY_FROM_JOB = 0x01000000  # Windows only


def base_url(port: int = PORT) -> str:
    return f"http://{HOST}:{port}"


def port_open(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((HOST, port)) == 0


def assistant_id(port: int = PORT) -> str | None:
    r = httpx.post(
        f"{base_url(port)}/assistants/search", json={"graph_id": GRAPH, "limit": 10}, timeout=30
    )
    r.raise_for_status()
    found = [a for a in r.json() if a.get("graph_id") == GRAPH]
    return found[0]["assistant_id"] if found else None


def kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        # uv -> langgraph -> uvicorn: kill the whole tree.
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def start(
    port: int = PORT,
    env: dict[str, str] | None = None,
    log_file: Path = LOG_FILE,
    pid_file: Path = PID_FILE,
) -> tuple[subprocess.Popen, str]:
    """Start the service and return ``(process, assistant_id)``. Raises ``RuntimeError`` (after
    stopping the process) when it does not come up. ``env`` adds to this process's environment
    (the integration tests pass the test-only fake model switch)."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    # langgraph dev resolves graph paths ("./graph.py") against the working directory, not the
    # config file's folder, so it runs from the service folder (stage 0B, Windows).
    cmd = [uv, "run", "--project", str(REPO), "langgraph", "dev", "--config", CONFIG]
    cmd += ["--no-browser", "--no-reload", "--host", HOST, "--port", str(port)]
    child_env = {**os.environ, "PYTHONUTF8": "1", **(env or {})}
    flags = 0
    if sys.platform == "win32":
        # Detached from this console, in its own process group, and out of this process's job
        # object when the job allows it (a terminal or tool that ends a job would otherwise take
        # the service down with the script that started it).
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        flags |= CREATE_BREAKAWAY_FROM_JOB
    with log_file.open("wb") as log:
        popen = dict(
            cwd=SERVICE_DIR, env=child_env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )  # fmt: skip
        try:
            proc = subprocess.Popen(cmd, creationflags=flags, **popen)
        except OSError:  # the job does not allow breakaway
            flags &= ~CREATE_BREAKAWAY_FROM_JOB
            proc = subprocess.Popen(cmd, creationflags=flags, **popen)
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            if httpx.get(f"{base_url(port)}/ok", timeout=2).status_code == 200:
                aid = assistant_id(port)
                if aid:
                    return proc, aid
        except httpx.TransportError:
            pass
        if "Application startup failed" in log_file.read_text(encoding="utf-8", errors="replace"):
            break
        time.sleep(1)
    tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
    kill_tree(proc.pid)
    pid_file.unlink(missing_ok=True)
    raise RuntimeError(f"the service did not come up (exit code {proc.poll()}):\n{tail}")


def stop(port: int = PORT, pid_file: Path = PID_FILE) -> bool:
    """Stop the service started by ``start``; True when nothing listens afterwards."""
    if pid_file.is_file():
        kill_tree(int(pid_file.read_text(encoding="utf-8").strip()))
        pid_file.unlink(missing_ok=True)
    for _ in range(15):
        if not port_open(port):
            return True
        time.sleep(1)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stop", action="store_true", help="stop the running service")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    if args.stop:
        if stop(args.port):
            print(f"stopped; nothing listens on {HOST}:{args.port}")
            return 0
        print(
            f"error: something still listens on {HOST}:{args.port} (not started by this script?)",
            file=sys.stderr,
        )
        return 1
    if port_open(args.port):
        try:
            aid = assistant_id(args.port)
        except (httpx.HTTPError, ValueError):
            aid = None
        if aid:
            print(f"already running at {base_url(args.port)}")
            print(f"assistant_id: {aid}")
            return 0
        print(f"error: port {args.port} is in use by something else", file=sys.stderr)
        return 1
    try:
        _, aid = start(args.port)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"security specialist service running at {base_url(args.port)} (127.0.0.1 only)")
    print(f"assistant_id: {aid}")
    print(f"log: {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
