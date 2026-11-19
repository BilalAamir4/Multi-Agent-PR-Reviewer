import io
import tarfile

import docker
import pytest
from docker.errors import DockerException

from reviewer import sandbox
from reviewer.sandbox import SandboxUnavailable, build_tar, get_client, run_in_sandbox


def test_get_client_turns_docker_exception_into_sandbox_unavailable(monkeypatch):
    def boom():
        raise DockerException("Error while fetching server API version")

    monkeypatch.setattr(docker, "from_env", boom)
    with pytest.raises(SandboxUnavailable) as info:
        get_client()
    assert str(info.value) == "Docker Desktop is not running"


def test_get_client_turns_failed_ping_into_sandbox_unavailable(monkeypatch):
    class Client:
        def ping(self):
            raise DockerException("ping failed")

    monkeypatch.setattr(docker, "from_env", Client)
    with pytest.raises(SandboxUnavailable, match="^Docker Desktop is not running$"):
        get_client()


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.py", "a/../../x.py", "C:/x.py", "a\\b.py"])
def test_bad_paths_rejected_before_docker(monkeypatch, bad):
    def no_docker():
        raise AssertionError("Docker must not be contacted for invalid paths")

    monkeypatch.setattr(sandbox, "get_client", no_docker)
    with pytest.raises(ValueError, match="invalid sandbox path"):
        run_in_sandbox("img", {bad: b"x"}, ["true"])


def test_tar_members_owned_by_1000_with_modes():
    data = build_tar({"pkg/mod.py": b"x = 1\n", "top.py": b""})
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = {m.name: m for m in tar.getmembers()}
    assert set(members) == {".", "./pkg", "./pkg/mod.py", "./top.py"}
    for m in members.values():
        assert (m.uid, m.gid) == (1000, 1000)
        assert m.mode == (0o755 if m.isdir() else 0o644)
    assert members["./pkg/mod.py"].size == 6
