"""Stage 0 done criterion (b): a container with no network prints something, started from Python.

Pulls python:3.12-slim and builds ``mapr-sandbox-base:check`` from it, pinned by digest, adding
only the /work directory the sandbox image contract needs (network is on only while pulling and
building, D2). Then runs a probe that checks the D2 settings from inside the container, and
checks that a timeout kills and removes the container.

    uv run python scripts/check_sandbox.py
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from reviewer.sandbox import (  # noqa: E402
    LABEL,
    SandboxResult,
    SandboxUnavailable,
    get_client,
    run_in_sandbox,
)

BASE = "python:3.12-slim"
IMAGE = "mapr-sandbox-base:check"
CHECKS = ("network_blocked", "rootfs_read_only", "tmp_writable", "work_writable", "not_root")

PROBE = b"""\
import errno
import os
import socket

print("hello from the sandbox")


def report(name, ok, detail):
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {detail}")


try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    report("network_blocked", False, "connected to 1.1.1.1:53")
except OSError as e:
    report("network_blocked", True, f"{type(e).__name__}: {e}")

try:
    with open("/probe_test", "w") as f:
        f.write("x")
    report("rootfs_read_only", False, "wrote /probe_test")
except OSError as e:
    report("rootfs_read_only", e.errno == errno.EROFS, f"errno {e.errno} {e.strerror}")

try:
    with open("/tmp/probe_test", "w") as f:
        f.write("x")
    report("tmp_writable", True, "wrote /tmp/probe_test")
except OSError as e:
    report("tmp_writable", False, f"errno {e.errno} {e.strerror}")

try:
    with open("/work/.write_test", "w") as f:
        f.write("x")
    report("work_writable", True, "wrote /work/.write_test")
except OSError as e:
    report("work_writable", False, f"errno {e.errno} {e.strerror}")

uid = os.getuid()
report("not_root", uid != 0, f"uid={uid} gid={os.getgid()}")
"""


@dataclass
class SandboxCheck:
    digest: str = ""
    image_id: str = ""
    probe: SandboxResult | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    hello: bool = False
    timeout: SandboxResult | None = None
    leftover_containers: int = -1
    leftover_volumes: int = 0

    @property
    def probe_ok(self) -> bool:
        return self.hello and all(self.checks.get(c) for c in CHECKS)

    @property
    def timeout_ok(self) -> bool:
        return bool(self.timeout and self.timeout.timed_out) and self.leftover_containers == 0


def _dangling_volumes(client) -> set[str]:
    return {v.name for v in client.volumes.list(filters={"dangling": True})}


def run_check() -> SandboxCheck:
    res = SandboxCheck()
    client = get_client()
    try:
        base = client.images.pull("python", tag="3.12-slim")
        digests = [d for d in base.attrs.get("RepoDigests") or [] if d.startswith("python@")]
        if not digests:
            raise RuntimeError(f"no repo digest for {BASE}")
        res.digest = digests[0]  # python@sha256:...
        pinned = f"{BASE}@{res.digest.split('@', 1)[1]}"
        dockerfile = f"FROM {pinned}\nRUN mkdir /work && chown 1000:1000 /work\n"
        image, _ = client.images.build(
            fileobj=io.BytesIO(dockerfile.encode()), tag=IMAGE, rm=True, pull=False
        )
        res.image_id = image.id
        volumes_before = _dangling_volumes(client)
    finally:
        client.close()

    res.probe = run_in_sandbox(IMAGE, {"probe.py": PROBE}, ["python", "/work/probe.py"])
    res.hello = "hello from the sandbox" in res.probe.stdout
    for line in res.probe.stdout.splitlines():
        if line.startswith("CHECK "):
            _, name, verdict, detail = (line.split(" ", 3) + [""])[:4]
            res.checks[name] = verdict == "PASS"
            res.details[name] = detail

    res.timeout = run_in_sandbox(
        IMAGE, {}, ["python", "-c", "import time; time.sleep(30)"], timeout_s=3
    )

    client = get_client()
    try:
        res.leftover_containers = len(
            client.containers.list(all=True, filters={"label": f"{LABEL}=1"})
        )
        res.leftover_volumes = len(_dangling_volumes(client) - volumes_before)
    finally:
        client.close()
    return res


def main() -> int:
    try:
        res = run_check()
    except SandboxUnavailable as e:
        print(f"FAIL: {e}")
        return 2
    print(f"base image: {BASE} digest {res.digest}")
    print(f"check image: {IMAGE} id {res.image_id}")
    p = res.probe
    print(f"probe: exit_code={p.exit_code} timed_out={p.timed_out} {p.duration_s:.2f}s")
    print(f"{'PASS' if res.hello else 'FAIL'} prints 'hello from the sandbox'")
    for name in CHECKS:
        ok = res.checks.get(name)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {res.details.get(name, 'no output')}")
    if p.stderr.strip():
        print(f"probe stderr: {p.stderr.strip()[:1000]}")
    t = res.timeout
    print(
        f"{'PASS' if res.timeout_ok else 'FAIL'} timeout: timed_out={t.timed_out} "
        f"exit_code={t.exit_code} {t.duration_s:.2f}s, "
        f"sandbox containers left={res.leftover_containers}, "
        f"new dangling volumes={res.leftover_volumes}"
    )
    ok = res.probe_ok and res.timeout_ok and res.leftover_volumes == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
