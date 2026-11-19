"""Run commands on PR code in a throwaway Docker container (D2, D2b).

The code is copied into an anonymous volume at ``/work`` with ``put_archive`` before the container
starts; nothing from the host is mounted. The container has no network, a read-only root
filesystem, resource limits and a non-root user, and it is removed with its volume afterwards
(docs/decisions/0003-sandbox-copy-into-volume.md).
"""

from __future__ import annotations

import io
import tarfile
import time
from dataclasses import dataclass
from pathlib import PurePosixPath

import docker
import requests
from docker.errors import DockerException
from docker.types import Mount

from reviewer.schemas import validate_posix_path

UID = GID = 1000
WORKDIR = "/work"
LABEL = "multi-agent-pr-reviewer.sandbox"


class SandboxUnavailable(Exception):
    """Docker Desktop cannot be reached (D2b)."""

    def __init__(self) -> None:
        super().__init__("Docker Desktop is not running")


@dataclass
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def get_client() -> docker.DockerClient:
    """Return a connected Docker client, or raise ``SandboxUnavailable`` (D2b)."""
    try:
        client = docker.from_env()
        client.ping()
    except (DockerException, requests.exceptions.RequestException, OSError) as e:
        raise SandboxUnavailable() from e
    return client


def _check_paths(files: dict[str, bytes]) -> dict[PurePosixPath, bytes]:
    checked: dict[PurePosixPath, bytes] = {}
    for raw, content in files.items():
        try:
            rel = validate_posix_path(raw)
        except ValueError as e:
            raise ValueError(f"invalid sandbox path {raw!r}: {e}") from None
        checked[PurePosixPath(rel)] = content
    return checked


def build_tar(files: dict[str, bytes]) -> bytes:
    """Build an in-memory tar of ``files``, owned by uid/gid 1000, rooted at a ``.`` entry."""
    checked = _check_paths(files)
    dirs = {PurePosixPath(".")}
    for path in checked:
        dirs.update(p for p in path.parents if str(p) != ".")
    now = time.time()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for d in sorted(dirs, key=lambda p: (len(p.parts), str(p))):
            info = tarfile.TarInfo("." if str(d) == "." else f"./{d}")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid, info.gid, info.mtime = UID, GID, now
            tar.addfile(info)
        for path, content in sorted(checked.items(), key=lambda kv: str(kv[0])):
            info = tarfile.TarInfo(f"./{path}")
            info.size = len(content)
            info.mode = 0o644
            info.uid, info.gid, info.mtime = UID, GID, now
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def run_in_sandbox(
    image: str, files: dict[str, bytes], command: list[str], timeout_s: int = 120
) -> SandboxResult:
    """Copy ``files`` into a fresh no-network container and run ``command`` there (D2).

    Image contract: ``image`` must contain ``/work`` owned by ``1000:1000``
    (``RUN mkdir /work && chown 1000:1000 /work``). Docker skips the tar's ``.`` entry, so the
    anonymous volume's root takes its ownership from the image's ``/work``; without it the
    non-root user cannot create files directly in ``/work`` (docs/decisions/0003).

    ``files`` maps POSIX relative paths to contents; absolute paths and ``..`` are rejected.
    On timeout the container is killed and ``timed_out`` is True. The container and its volume
    are always removed afterwards.
    """
    tar_bytes = build_tar(files)  # validates paths before touching Docker
    client = get_client()
    container = None
    try:
        container = client.containers.create(
            image,
            command=command,
            network_mode="none",
            read_only=True,
            tmpfs={"/tmp": "rw,size=256m"},
            mem_limit="1g",
            nano_cpus=1_000_000_000,
            pids_limit=256,
            user=f"{UID}:{GID}",
            working_dir=WORKDIR,
            # Anonymous volume (no source): not a host mount, removed with remove(v=True).
            mounts=[Mount(target=WORKDIR, source=None, type="volume")],
            labels={LABEL: "1"},
        )
        if not container.put_archive(WORKDIR, tar_bytes):
            raise RuntimeError(f"put_archive into {WORKDIR} was refused")
        start = time.monotonic()
        container.start()
        timed_out = False
        exit_code: int | None
        try:
            exit_code = container.wait(timeout=timeout_s).get("StatusCode")
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            timed_out = True
            try:
                container.kill()
            except DockerException:
                pass  # it may have exited between the timeout and the kill
            exit_code = None
        duration = time.monotonic() - start
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", "replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", "replace")
        return SandboxResult(exit_code, stdout, stderr, timed_out, duration)
    except (requests.exceptions.ConnectionError, OSError) as e:
        raise SandboxUnavailable() from e
    finally:
        if container is not None:
            try:
                container.remove(v=True, force=True)
            except DockerException:
                pass
        client.close()
