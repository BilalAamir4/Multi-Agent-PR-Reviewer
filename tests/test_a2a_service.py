"""The A2A security service (stage 3, decision 0011): snapshot packing, the safe archive reader,
and ``a2a_security.graph.serve`` (what the service's one node does).

No network, model or Docker call. Every bad request must come back as a clean ``error`` response
of the normal shape, never an exception, and the temporary directory must always be deleted.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import random
import tarfile
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import reviewer.a2a_schema as schema
import reviewer.snapshot as snapshot
from a2a_security import graph as service
from reviewer.a2a_schema import SecurityReviewRequest, SecurityReviewResponse
from reviewer.snapshot import ArchiveError, pack, sha256_hex, unpack
from reviewer.test_hooks import FAKE_MODEL_VAR, TEST_MODE_VAR, TestHookRejected, fake_model_script

FILES = {"pkg/a.py": b"x = 1\n", "pkg/b.py": b"y = 2\n", "README.md": b"hi\n"}
DIFF = (
    "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n"
    "@@ -1 +1 @@\n-x = 0\n+x = 1\n"
)


@pytest.fixture(autouse=True)
def _never_reach_a_model(monkeypatch):
    """A request that is accepted must not run a real review in these tests."""

    def refuse(*a, **k):
        raise AssertionError("a unit test reached run_security_review")

    monkeypatch.setattr(service, "run_security_review", refuse)


def request_text(archive: bytes, **over) -> str:
    data = {
        "protocol_version": "2",
        "pr_title": "t",
        "pr_body": "b",
        "focus": None,
        "diff_text": DIFF,
        "workspace_archive": base64.b64encode(archive).decode(),
        "workspace_sha256": sha256_hex(archive),
        "evaluation": True,
        "spawn_enabled": False,
        "trace_parent": None,
    }
    data.update(over)
    return json.dumps(data)


def raw_tar(members: list[tarfile.TarInfo], payloads: dict[str, bytes] | None = None) -> bytes:
    """A tar.gz built member by member (so a test can write what ``pack`` never would)."""
    payloads = payloads or {}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for m in members:
            data = payloads.get(m.name, b"")
            if m.isreg():
                m.size = len(data)
            tar.addfile(m, io.BytesIO(data) if m.isreg() else None)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def member(name: str, kind=tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info


def assert_clean_error(resp: dict, error_class: str, fragment: str = "") -> None:
    SecurityReviewResponse.model_validate(resp)  # the normal response shape
    assert resp["status"] == "error"
    assert resp["error_class"] == error_class
    assert fragment in resp["error"]
    assert resp["findings"] == [] and resp["counts"]["model_calls"] == 0
    assert resp["service"]["name"] == "security_reviewer"


@pytest.fixture
def temp_dirs(monkeypatch, tmp_path):
    """Record every directory ``serve`` creates, under ``tmp_path``."""
    made: list[Path] = []

    def mkdtemp(prefix="x"):
        d = tmp_path / f"{prefix}{len(made)}"
        d.mkdir()
        made.append(d)
        return str(d)

    monkeypatch.setattr(service.tempfile, "mkdtemp", mkdtemp)
    return made


# --- pack / unpack ---------------------------------------------------------------------------


def test_pack_is_deterministic_and_order_independent():
    a = pack(FILES)
    b = pack(dict(reversed(list(FILES.items()))))
    assert a == b and sha256_hex(a) == sha256_hex(b)
    assert pack({**FILES, "README.md": b"other\n"}) != a


def test_pack_unpack_round_trip(tmp_path):
    assert unpack(pack(FILES), tmp_path) == 3
    assert {
        p.relative_to(tmp_path).as_posix(): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    } == FILES


# --- Unsafe archives (each: a clean ArchiveError, nothing outside the directory) ---------------


@pytest.mark.parametrize(
    "name",
    ["../evil.txt", "a/../../evil.txt", "/abs.txt", "C:/evil.txt", "a\\..\\evil.txt", "a/./b.txt",
     "nul", "dir/CON.txt", "bad:stream", ""],
)  # fmt: skip
def test_unsafe_names_are_refused(tmp_path, name):
    data = raw_tar([member(name)], {name: b"x"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError):
        unpack(data, dest)
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize(
    ("kind", "link", "word"),
    [
        (tarfile.SYMTYPE, "../../etc/passwd", "symbolic link"),
        (tarfile.LNKTYPE, "pkg/a.py", "hard link"),
        (tarfile.CHRTYPE, "", "special"),
        (tarfile.BLKTYPE, "", "special"),
        (tarfile.FIFOTYPE, "", "special"),
    ],
)
def test_links_and_devices_are_refused(tmp_path, kind, link, word):
    data = raw_tar([member("pkg/a.py"), member("pkg/link", kind, link)], {"pkg/a.py": b"x"})
    with pytest.raises(ArchiveError, match=word):
        unpack(data, tmp_path)


def test_duplicate_entries_are_refused(tmp_path):
    data = raw_tar([member("a.py"), member("a.py")], {"a.py": b"x"})
    with pytest.raises(ArchiveError, match="twice"):
        unpack(data, tmp_path)


def test_not_an_archive_is_refused(tmp_path):
    for junk in (b"", b"not gzip at all", gzip.compress(b"not a tar file" * 100)):
        with pytest.raises(ArchiveError):
            unpack(junk, tmp_path)


def test_entry_and_size_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "MAX_ARCHIVE_ENTRIES", 3)
    with pytest.raises(ArchiveError, match="more than 3 entries"):
        unpack(pack({f"f{i}.py": b"x" for i in range(5)}), tmp_path)
    monkeypatch.setattr(snapshot, "MAX_ARCHIVE_ENTRIES", 20_000)
    monkeypatch.setattr(snapshot, "MAX_UNPACKED_BYTES", 1000)
    bomb = pack({"zeros.bin": b"\0" * 50_000})  # tiny compressed, large unpacked
    assert len(bomb) < 1000
    with pytest.raises(ArchiveError, match="unpacks to more than"):
        unpack(bomb, tmp_path)
    monkeypatch.setattr(snapshot, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ArchiveError, match="larger than 100 bytes"):
        unpack(pack({"a.py": random.Random(0).randbytes(4000)}), tmp_path)


# --- serve(): what the service's node answers --------------------------------------------------


def test_tampered_sha_is_a_clean_error(temp_dirs):
    resp = service.serve(request_text(pack(FILES), workspace_sha256="0" * 64))
    assert_clean_error(resp, "ArchiveError", "does not match")
    assert temp_dirs == []  # refused before anything was unpacked


def test_tampered_archive_bytes_are_a_clean_error(temp_dirs):
    archive = pack(FILES)
    bad = archive[:-1] + bytes([archive[-1] ^ 1])
    resp = service.serve(request_text(archive, workspace_archive=base64.b64encode(bad).decode()))
    assert_clean_error(resp, "ArchiveError", "does not match")


def test_path_traversal_archive_is_a_clean_error_and_cleans_up(temp_dirs, tmp_path):
    data = raw_tar(
        [member("ok.py"), member("../escape.txt")], {"ok.py": b"x", "../escape.txt": b"x"}
    )
    resp = service.serve(request_text(data))
    assert_clean_error(resp, "ArchiveError")
    assert not (tmp_path / "escape.txt").exists()
    assert len(temp_dirs) == 1 and not temp_dirs[0].exists()  # always deleted


def test_symlink_member_is_a_clean_error_and_cleans_up(temp_dirs):
    data = raw_tar([member("ok.py"), member("l", tarfile.SYMTYPE, "/etc/passwd")], {"ok.py": b"x"})
    resp = service.serve(request_text(data))
    assert_clean_error(resp, "ArchiveError", "symbolic link")
    assert not temp_dirs[0].exists()


def test_oversized_archive_is_a_clean_error(temp_dirs, monkeypatch):
    monkeypatch.setattr(schema, "MAX_ARCHIVE_BYTES", 500)
    monkeypatch.setattr(snapshot, "MAX_ARCHIVE_BYTES", 500)
    resp = service.serve(request_text(pack({"a.py": random.Random(0).randbytes(4000)})))
    assert_clean_error(resp, "InvalidRequest", "larger than 500 bytes")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("not json", "invalid"),
        ("[]", "invalid"),
        (json.dumps({"protocol_version": "2"}), "pr_title"),
        (request_text(pack(FILES), protocol_version="1"), "upgraded together"),
        (request_text(pack(FILES), protocol_version="3"), "protocol_version"),
        (request_text(pack(FILES), spawn_enabled="yes"), "spawn_enabled"),
        (request_text(pack(FILES), surprise=1), "surprise"),
        (request_text(pack(FILES), evaluation="yes"), "evaluation"),
        (request_text(pack(FILES), workspace_sha256="XYZ"), "workspace_sha256"),
        (request_text(pack(FILES), trace_parent={"authorization": "x"}), "trace_parent"),
    ],
)
def test_invalid_request_is_a_clean_error(temp_dirs, text, fragment):
    resp = service.serve(text)
    assert_clean_error(resp, "InvalidRequest", fragment)
    assert temp_dirs == []


def test_node_without_a_human_message_answers_an_error():
    out = service.review({"messages": [AIMessage(content="hi")]})
    resp = json.loads(out["messages"][-1].content)
    assert_clean_error(resp, "InvalidRequest", "no human message")


def test_node_reads_the_last_human_message_and_answers_json(monkeypatch, temp_dirs):
    seen = {}

    def fake_review(payload, *, router, parent_config, config):
        seen["payload"] = payload
        seen["files"] = sorted(p.name for p in Path(payload["workspace"]).rglob("*") if p.is_file())
        seen["router_eval"] = router.evaluation
        seen["metadata"] = parent_config["metadata"]
        return {**service.error_response("X", "y"), "status": "complete", "error_class": None,
                "error": None}  # fmt: skip

    monkeypatch.setattr(service, "run_security_review", fake_review)
    monkeypatch.setattr(
        service, "make_router", lambda evaluation: type("R", (), {"evaluation": evaluation})()
    )
    text = request_text(pack(FILES), focus="look at a.py")
    out = service.review({"messages": [HumanMessage(content="old"), HumanMessage(content=text)]})
    resp = SecurityReviewResponse.model_validate_json(out["messages"][-1].content)
    assert resp.status == "complete" and resp.service is not None
    assert (
        resp.service.name == "security_reviewer"
        and resp.service.version == service.prompt_version()
    )
    assert seen["files"] == ["README.md", "a.py", "b.py"]
    assert seen["payload"]["focus"] == "look at a.py" and seen["router_eval"] is True
    assert seen["metadata"]["a2a_service"] == "security_reviewer"
    assert len(temp_dirs) == 1 and not temp_dirs[0].exists()


def test_content_blocks_are_read_as_text(monkeypatch, temp_dirs):
    text = request_text(pack(FILES), workspace_sha256="0" * 64)
    blocks = [{"type": "text", "text": text[:10]}, {"type": "text", "text": text[10:]}]
    out = service.review({"messages": [HumanMessage(content=blocks)]})
    assert_clean_error(json.loads(out["messages"][-1].content), "ArchiveError")


def test_an_internal_failure_is_an_error_response_and_cleans_up(monkeypatch, temp_dirs):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(service, "run_security_review", boom)
    monkeypatch.setattr(service, "make_router", lambda evaluation: object())
    resp = service.serve(request_text(pack(FILES)))
    assert_clean_error(resp, "InternalError", "kaboom")
    assert not temp_dirs[0].exists()


def test_request_model_rejects_extra_and_checks_archive_size():
    with pytest.raises(ValueError):
        SecurityReviewRequest.model_validate_json(request_text(pack(FILES), extra_field=1))
    req = SecurityReviewRequest.model_validate_json(request_text(pack(FILES)))
    assert req.archive_bytes() == pack(FILES)
    bad = req.model_copy(update={"workspace_archive": "not base64!!"})
    with pytest.raises(ValueError, match="base64"):
        bad.archive_bytes()


# --- The test-only fake-model switch and its guard ---------------------------------------------


def test_fake_model_variable_is_ignored_when_unset():
    assert fake_model_script({}) is None
    assert fake_model_script({TEST_MODE_VAR: "1"}) is None


def test_fake_model_variable_is_rejected_outside_test_mode():
    for env in ({FAKE_MODEL_VAR: "script.json"}, {FAKE_MODEL_VAR: "s.json", TEST_MODE_VAR: "0"},
                {FAKE_MODEL_VAR: "s.json", TEST_MODE_VAR: "true"}):  # fmt: skip
        with pytest.raises(TestHookRejected, match="test-only"):
            fake_model_script(env)
    assert fake_model_script({FAKE_MODEL_VAR: "s.json", TEST_MODE_VAR: "1"}) == Path("s.json")


def test_service_refuses_the_fake_model_outside_test_mode(monkeypatch):
    monkeypatch.delenv(TEST_MODE_VAR, raising=False)
    monkeypatch.setenv(FAKE_MODEL_VAR, "script.json")
    with pytest.raises(TestHookRejected):
        service.make_router(True)


def test_service_uses_its_own_quota_state_file():
    assert service.STATE_FILE == "quota_state.a2a.json"
    assert service._quota_tracker().path.name == "quota_state.a2a.json"


# --- Transport parity, deterministic (in-process half) -------------------------------------------


def test_serve_matches_the_local_run_with_the_same_scripted_model(tmp_path, monkeypatch):
    """The same scripted model through ``serve`` (archive, unpack, run) and through
    ``run_security_review`` on the original workspace gives identical responses once the service
    block, the timings and the PIDs are removed. The HTTP half is in
    ``tests/test_integration_a2a.py``."""
    from reviewer.a2a_client import build_request
    from reviewer.agents.specialists import run_security_review
    from reviewer.config import load_models_config
    from reviewer.models import QuotaRouter, QuotaTracker
    from reviewer.test_hooks import script_builder
    from tests.a2a_helpers import FOCUS, comparable, request_for, write_script
    from tests.dev_pr import make_fixture_workspace, pr_text

    ctx = make_fixture_workspace("dev_security_pr", tmp_path)
    script = write_script(ctx, tmp_path / "script.json")
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota.json", providers)
    local_router = QuotaRouter(evaluation=True, tracker=tracker)
    local_router.build = script_builder(script)
    monkeypatch.undo()  # the autouse guard must not stop the real review below
    local = run_security_review(request_for(ctx), router=local_router)

    monkeypatch.setenv(TEST_MODE_VAR, "1")
    monkeypatch.setenv(FAKE_MODEL_VAR, str(script))
    monkeypatch.setattr(service, "_tracker", QuotaTracker(tmp_path / "quota.a2a.json", providers))
    title, body = pr_text("dev_security_pr")
    request = build_request(
        ctx, pr_title=title, pr_body=body, focus=FOCUS, evaluation=True, spawn_enabled=False
    )
    remote = service.serve(request.model_dump_json())

    assert local["status"] == "complete", local
    assert [f["category"] for f in local["findings"]] == ["sql_injection", "hardcoded_secret"]
    assert [d["reason"] for d in local["dropped"]] == ["not_in_diff"]
    assert remote["service"]["pid"] > 0 and "service" not in comparable(local)
    assert comparable(remote) == comparable(local)
    assert remote["counts"]["model_calls"] == 4 and remote["counts"]["input_tokens"] == 4000


def test_a_spawning_review_through_serve_matches_the_local_run(tmp_path, monkeypatch):
    """The parity rule with sub-agents (decision 0012): the same scripted model, one sub-agent at a
    time, through ``serve`` and through ``run_security_review`` gives identical responses (children
    and spawn record included) once timings, waits and the service block are removed."""
    from reviewer.a2a_client import build_request
    from reviewer.agents.specialists import run_security_review
    from reviewer.config import load_models_config, load_review_config
    from reviewer.models import QuotaRouter, QuotaTracker
    from reviewer.test_hooks import script_builder
    from tests.a2a_helpers import FOCUS, comparable, request_for, spawn_script
    from tests.dev_pr import make_fixture_workspace, pr_text

    real = load_review_config()
    spawn = real.spawn.model_copy(
        update={"min_py_files": 2, "max_files_per_subagent": 2, "max_concurrency": 1}
    )
    cfg = real.model_copy(update={"spawn": spawn})
    ctx = make_fixture_workspace("dev_security_pr", tmp_path)
    script = tmp_path / "script.json"
    script.write_bytes(json.dumps(spawn_script(ctx)).encode("utf-8"))
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    local_router = QuotaRouter(
        evaluation=True, tracker=QuotaTracker(tmp_path / "quota.json", providers)
    )
    local_router.build = script_builder(script)
    monkeypatch.undo()  # the autouse guard must not stop the real review below
    request = {**request_for(ctx), "spawn_enabled": True}
    local = run_security_review(request, router=local_router, config=cfg)

    monkeypatch.setenv(TEST_MODE_VAR, "1")
    monkeypatch.setenv(FAKE_MODEL_VAR, str(script))
    monkeypatch.setattr(service, "load_review_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(service, "_tracker", QuotaTracker(tmp_path / "quota.a2a.json", providers))
    title, body = pr_text("dev_security_pr")
    wire = build_request(
        ctx, pr_title=title, pr_body=body, focus=FOCUS, evaluation=True, spawn_enabled=True
    )
    remote = service.serve(wire.model_dump_json())

    assert local["status"] == "complete", local
    assert local["spawn"]["spawned"] is True and len(local["children"]) == 2
    assert [c["agent"] for c in local["children"]] == [
        "security_reviewer/sub1", "security_reviewer/sub2"
    ]  # fmt: skip
    assert {f["agent"] for f in local["findings"]} <= {
        "security_reviewer/sub1", "security_reviewer/sub2", "security_reviewer"
    }  # fmt: skip
    assert remote["spawn"]["spawned"] is True and remote["service"]["pid"] > 0
    assert comparable(remote) == comparable(local)
    # every agent (each sub-agent and the cross-file pass) builds its own scripted model, which
    # starts at the top of the script: two calls each, three agents
    assert remote["counts"]["model_calls"] == 6 and remote["own"]["own_model_calls"] == 2
    assert local["own"]["own_model_calls"] == 2 and remote["counts"]["tool_calls"] == 3
    # spawn_enabled false in the request turns it off in the service as well
    off = service.serve(
        build_request(
            ctx, pr_title=title, pr_body=body, focus=FOCUS, evaluation=True, spawn_enabled=False
        ).model_dump_json()
    )
    assert off.get("error") or off["spawn"] is None
