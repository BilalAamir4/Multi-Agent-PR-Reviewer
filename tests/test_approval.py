"""Stage 6 (decision 0016): approval, checkpointed resume and GitHub posting. Fakes only: a fake
review runtime, ``httpx.MockTransport`` for GitHub, ``InMemorySaver`` and ``SqliteSaver`` in a
temporary directory. No network, model, Docker or GitHub call."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from approval import cli as approval_cli
from approval.decisions import DecisionError, apply_decisions, validate_decisions
from approval.github import GitHubClient, GitHubError, scrub
from approval.graph import (
    ApprovalRuntime,
    build_approval_graph,
    thread_config,
)
from approval.posting import build_review, marker_for
from approval.store import close_checkpointer, open_checkpointer
from approval.workspace import checkout_head
from reviewer.agents.judging import assign_refs
from reviewer.graph import build_graph, run_graph
from tests.dev_pr import FIXTURES
from tests.test_graph import CART, CATALOG, DIFF, LOW, fake_runtime, initial

TOKEN = "github_pat_SECRET1234567890abcdef"
KEY = "o/r#7@abc"
DIFF_TEXT = Path(DIFF).read_bytes().decode("utf-8")
OFF_DIFF = CART.model_copy(update={"line_start": 1, "line_end": 1, "title": "off diff"})


# --- A fake GitHub ---------------------------------------------------------------------------


class FakeGitHub:
    def __init__(self):
        self.reviews: list[dict] = []
        self.requests: list[tuple[str, str]] = []
        self.diff = DIFF_TEXT

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        self.requests.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "GET" and path.endswith("/pulls/7"):
            if "diff" in request.headers["accept"]:
                return httpx.Response(200, content=self.diff.encode())
            return httpx.Response(
                200,
                json={
                    "title": "T",
                    "body": "B",
                    "html_url": "https://github.com/o/r/pull/7",
                    "state": "open",
                    "head": {"sha": "abc", "repo": {"clone_url": "https://github.com/o/r.git"}},
                    "base": {"sha": "base"},
                },
            )
        if request.method == "GET" and path.endswith("/reviews"):
            return httpx.Response(200, json=self.reviews)
        if request.method == "POST" and path.endswith("/reviews"):
            body = json.loads(request.content)
            review = {
                "id": 100 + len(self.reviews),
                "html_url": f"https://github.com/o/r/pull/7#pullrequestreview-{len(self.reviews)}",
                "body": body["body"],
                "comments": body["comments"],
                "commit_id": body["commit_id"],
            }
            self.reviews.append(review)
            return httpx.Response(200, json=review)
        return httpx.Response(404, json={"message": "not found"})

    def client(self) -> GitHubClient:
        return GitHubClient(TOKEN, transport=httpx.MockTransport(self.handler))

    def posts(self) -> int:
        return sum(1 for m, p in self.requests if m == "POST")


def approval_state(**kw):
    return {**initial(), "repo": "o/r", "pr_number": 7, "head_sha": "abc", "review_key": KEY, **kw}


def make_graph(saver, gh: FakeGitHub, found=(CATALOG, CART, LOW), rejected=(LOW,), calls=None):
    rt = fake_runtime(list(found), rejected=rejected, calls=calls)
    return build_approval_graph(saver, rt, ApprovalRuntime(github=gh.client))


def payload_of(graph, config):
    return approval_cli.pending_payload(graph.get_state(config))


def decisions_for(payload, actions):
    pairs = zip(payload, actions, strict=True)
    return [{"finding_id": e["id"], "action": a, "edited": {}} for e, a in pairs]


# --- Isolation (rule 2) ----------------------------------------------------------------------


def test_build_graph_without_approval_is_unchanged():
    g = build_graph().get_graph()
    assert set(g.nodes) == {"__start__", "review", "judge", "present", "__end__"}
    assert {(e.source, e.target) for e in g.edges} == {
        ("__start__", "review"),
        ("review", "judge"),
        ("judge", "present"),
        ("present", "__end__"),
    }
    final = run_graph(initial(), fake_runtime([CATALOG, CART]))
    assert set(final) == {
        "review_key", "pr_title", "pr_body", "workspace", "diff_path", "sandbox_repo",
        "security_mode", "spawn_enabled", "judge", "evaluation", "review_result", "findings",
        "judge_result", "final_findings", "approval_payload", "status", "notes",
    }  # fmt: skip


def test_approval_graph_adds_approve_and_post_after_present():
    g = make_graph(InMemorySaver(), FakeGitHub()).get_graph()
    assert {(e.source, e.target) for e in g.edges} == {
        ("__start__", "review"),
        ("review", "judge"),
        ("judge", "present"),
        ("present", "approve"),
        ("approve", "post"),
        ("post", "__end__"),
    }


# --- Interrupt and resume --------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_interrupt_then_resume_posts_only_approved(tmp_path, backend):
    gh = FakeGitHub()
    saver = InMemorySaver() if backend == "memory" else open_checkpointer(tmp_path)
    graph = make_graph(saver, gh)
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    snap = graph.get_state(config)
    assert snap.next == ("approve",) and "posted" not in snap.values
    payload = payload_of(graph, config)
    assert [e["title"] for e in payload] == [CART.title, CATALOG.title]  # LOW rejected by judge
    assert gh.posts() == 0
    final = graph.invoke(Command(resume=decisions_for(payload, ["approve", "reject"])), config)
    assert final["posted"]["existed"] is False and final["posted"]["inline"] == [payload[0]["id"]]
    assert gh.posts() == 1
    review = gh.reviews[0]
    assert marker_for(KEY) in review["body"] and review["commit_id"] == "abc"
    assert [(c["path"], c["line"], c["start_line"]) for c in review["comments"]] == [
        ("shop/cart.py", 23, 22)
    ]
    json.dumps(graph.get_state(config).values)  # JSON-serializable, and no token anywhere:
    assert TOKEN not in json.dumps(graph.get_state(config).values)
    if backend == "sqlite":
        close_checkpointer(saver)
        assert TOKEN.encode() not in (tmp_path / "checkpoints.sqlite").read_bytes()


def test_edit_changes_only_title_explanation_severity_and_is_marked():
    gh = FakeGitHub()
    graph = make_graph(InMemorySaver(), gh)
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    payload = payload_of(graph, config)
    d = decisions_for(payload, ["edit", "reject"])
    d[0]["edited"] = {"title": "Better title", "severity": "low"}
    final = graph.invoke(Command(resume=d), config)
    assert final["approved_findings"][0]["title"] == "Better title"
    assert final["approved_findings"][0]["line_start"] == 22  # lines untouched
    body = gh.reviews[0]["comments"][0]["body"]
    assert body.startswith("**Better title** · severity: low")
    assert "found by test_reviewer/sub1 · verified: confirmed · edited by reviewer" in body


def test_no_findings_means_no_interrupt_and_nothing_posted():
    gh = FakeGitHub()
    graph = make_graph(InMemorySaver(), gh, found=[], rejected=())
    final = graph.invoke(approval_state(), thread_config(KEY))
    assert final["posted"] == {"skipped": "no approved findings"} and gh.requests == []


def test_all_rejected_posts_nothing():
    gh = FakeGitHub()
    graph = make_graph(InMemorySaver(), gh)
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    payload = payload_of(graph, config)
    final = graph.invoke(Command(resume=decisions_for(payload, ["reject", "reject"])), config)
    assert final["posted"] == {"skipped": "no approved findings"}
    assert gh.requests == []  # not even list_reviews


def test_aborted_review_is_not_posted():
    gh = FakeGitHub()
    rt = fake_runtime([CATALOG], review_status="quota_paused")
    graph = build_approval_graph(InMemorySaver(), rt, ApprovalRuntime(github=gh.client))
    final = graph.invoke(approval_state(), thread_config(KEY))
    assert "quota_paused" in final["posted"]["skipped"] and gh.requests == []


# --- Decision validation ---------------------------------------------------------------------

PAYLOAD = [{"id": "a", **{k: 1 for k in ()}}, {"id": "b"}]


def test_validation_rules():
    ok = [
        {"finding_id": "a", "action": "approve"},
        {"finding_id": "b", "action": "edit", "edited": {"severity": "low"}},
    ]
    out = validate_decisions(ok, PAYLOAD)
    assert out[0]["edited"] == {} and out[1]["edited"] == {"severity": "low"}
    with pytest.raises(DecisionError, match="missing"):
        validate_decisions(ok[:1], PAYLOAD)
    with pytest.raises(DecisionError, match="duplicate"):
        validate_decisions([ok[0], ok[0], ok[1]], PAYLOAD)
    with pytest.raises(DecisionError, match="unknown"):
        validate_decisions([*ok, {"finding_id": "zzz", "action": "reject"}], PAYLOAD)
    for field in ("line_start", "line_end", "file", "category"):
        bad = [ok[0], {"finding_id": "b", "action": "edit", "edited": {field: 3}}]
        with pytest.raises(DecisionError, match="cannot edit"):
            validate_decisions(bad, PAYLOAD)
    with pytest.raises(DecisionError, match="severity"):
        validate_decisions([ok[0], {**ok[1], "edited": {"severity": "urgent"}}], PAYLOAD)
    with pytest.raises(DecisionError, match="blank"):
        validate_decisions([ok[0], {**ok[1], "edited": {"title": "  "}}], PAYLOAD)
    with pytest.raises(DecisionError, match="only for action 'edit'"):
        validate_decisions([{**ok[0], "edited": {"title": "x"}}, ok[1]], PAYLOAD)
    with pytest.raises(DecisionError, match="action"):
        validate_decisions([{"finding_id": "a", "action": "maybe"}, ok[1]], PAYLOAD)
    assert (
        apply_decisions(out, [{"id": "a", "title": "t"}, {"id": "b", "title": "u"}])[1]["edited"]
        is True
    )


def test_invalid_resume_does_not_post():
    gh = FakeGitHub()
    graph = make_graph(InMemorySaver(), gh)
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    payload = payload_of(graph, config)
    bad = decisions_for(payload, ["approve", "approve"])
    bad[0]["edited"] = {"line_end": 99}
    with pytest.raises(DecisionError):
        graph.invoke(Command(resume=bad), config)
    assert gh.posts() == 0


# --- The post node ---------------------------------------------------------------------------


def test_post_finds_existing_marker_and_does_not_post_twice():
    gh = FakeGitHub()
    graph = make_graph(InMemorySaver(), gh)
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    payload = payload_of(graph, config)
    graph.invoke(Command(resume=decisions_for(payload, ["approve", "approve"])), config)
    assert gh.posts() == 1
    redo = next(s for s in graph.get_state_history(config) if s.next == ("post",))
    final = graph.invoke(None, redo.config)  # post runs again
    assert final["posted"]["existed"] is True and final["posted"]["review_id"] == 100
    assert gh.posts() == 1


def test_off_diff_finding_goes_into_the_summary():
    f = {"id": "x", "file": "shop/cart.py", "line_start": 1, "line_end": 1, "title": "Old line",
         "explanation": "why", "evidence": "x = 1", "severity": "low", "agent": "a",
         "verdict": "unchecked"}  # fmt: skip
    g = {**f, "id": "y", "line_start": 22, "line_end": 22, "title": "In diff"}
    plan = build_review([f, g], DIFF_TEXT, "<!-- m -->")
    assert plan.summary_ids == ["x"] and plan.inline_ids == ["y"]
    assert len(plan.comments) == 1 and plan.comments[0]["line"] == 22
    assert "start_line" not in plan.comments[0]
    assert plan.body.startswith("<!-- m -->") and "`shop/cart.py:1` **Old line**" in plan.body
    only_off = build_review([f], DIFF_TEXT, "<!-- m -->")
    assert only_off.comments == []


def test_range_comments_and_hunks():
    diff = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -1,2 +1,4 @@\n a\n+b\n+c\n d\n"
        "@@ -20,2 +21,4 @@\n e\n+f\n+g\n h\n"
    )
    base = {"file": "m.py", "title": "t", "explanation": "e", "evidence": "x", "severity": "high",
            "agent": "a", "verdict": "confirmed"}  # fmt: skip
    one = {**base, "id": "1", "line_start": 2, "line_end": 3}
    cross = {**base, "id": "2", "line_start": 3, "line_end": 22}  # two hunks
    half = {**base, "id": "3", "line_start": 1, "line_end": 3}  # line 1 is context, not changed
    plan = build_review([one, cross, half], diff, "<!-- m -->")
    assert plan.inline_ids == ["1", "2"] and plan.summary_ids == ["3"]
    first, second = plan.comments
    assert (first["line"], first["start_line"]) == (3, 2)
    assert first["start_side"] == "RIGHT" and first["side"] == "RIGHT"
    assert second["line"] == 22 and "start_line" not in second  # ends in different hunks


def test_cross_hunk_range_falls_back_to_single_line():
    diff = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -1,2 +1,4 @@\n a\n+b\n+c\n d\n"
        "@@ -20,2 +21,4 @@\n e\n+f\n+g\n h\n"
    )
    f = {"id": "2", "file": "m.py", "line_start": 3, "line_end": 23, "title": "t",
         "explanation": "e", "evidence": "x", "severity": "high", "agent": "a",
         "verdict": "confirmed"}  # fmt: skip
    plan = build_review([f], diff, "<!-- m -->")
    assert plan.comments[0]["line"] == 23 and "start_line" not in plan.comments[0]


def test_evidence_with_backticks_gets_a_longer_fence():
    f = {"id": "1", "file": "shop/cart.py", "line_start": 22, "line_end": 22, "title": "t",
         "explanation": "e", "evidence": "x = '''```'''", "severity": "low", "agent": "a",
         "verdict": "confirmed"}  # fmt: skip
    body = build_review([f], DIFF_TEXT, "m").comments[0]["body"]
    assert "````\nx = '''```'''\n````" in body


# --- The GitHub client -----------------------------------------------------------------------


def client_for(handler, **kw):
    sleeps: list[float] = []
    c = GitHubClient(
        TOKEN, transport=httpx.MockTransport(handler), sleep=sleeps.append, now=lambda: 1000.0,
        **kw,
    )  # fmt: skip
    return c, sleeps


def test_retries_5xx_429_and_honours_retry_after_and_reset():
    seq = iter(
        [
            httpx.Response(503),
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(
                403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1030"}
            ),
            httpx.Response(403, json={"message": "You have exceeded a secondary rate limit"}),
            httpx.Response(200, json=[]),
        ]
    )
    c, sleeps = client_for(lambda r: next(seq))
    assert c.list_reviews("o", "r", 1) == []
    assert sleeps == [1.0, 7.0, 30.0, 8.0]  # backoff 2**0, retry-after, reset - now, 2**3


def test_gives_up_and_scrubs_the_token():
    def handler(request):
        return httpx.Response(500, json={"message": f"boom {TOKEN} Bearer {TOKEN}"})

    c, sleeps = client_for(handler, max_retries=2)
    with pytest.raises(GitHubError) as e:
        c.get_pr("o", "r", 7)
    assert TOKEN not in str(e.value) and "500" in str(e.value) and len(sleeps) == 2
    assert TOKEN not in repr(c)


def test_post_is_not_retried_on_5xx_but_is_on_429():
    seen: list[int] = []

    def handler(request):
        seen.append(1)
        return httpx.Response(502 if len(seen) == 1 else 200, json={"id": 1})

    c, _ = client_for(handler)
    with pytest.raises(GitHubError):
        c.create_review("o", "r", 7, "abc", "body", [])
    assert len(seen) == 1
    seq = iter(
        [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json={"id": 3})]
    )
    c, sleeps = client_for(lambda r: next(seq))
    assert c.create_review("o", "r", 7, "abc", "body", [])["id"] == 3 and sleeps == [1.0]


def test_transport_error_text_is_scrubbed():
    def handler(request):
        raise httpx.ConnectError(f"cannot reach host with {TOKEN}")

    c, _ = client_for(handler, max_retries=1)
    with pytest.raises(GitHubError) as e:
        c.get_pr("o", "r", 7)
    assert TOKEN not in str(e.value)


def test_get_pr_and_diff_and_scrub_function():
    gh = FakeGitHub()
    c = gh.client()
    pr = c.get_pr("o", "r", 7)
    assert pr["head_sha"] == "abc" and pr["clone_url"].endswith("o/r.git")
    assert c.get_pr_diff("o", "r", 7) == DIFF_TEXT
    assert scrub(f"x {TOKEN} ghp_abcDEF123", TOKEN) == "x *** ***"


# --- Crash recovery --------------------------------------------------------------------------


def test_crash_in_judge_resumes_without_rerunning_review(tmp_path):
    gh = FakeGitHub()
    calls: list = []
    crash = {"on": True}
    rt = fake_runtime([CATALOG, CART], calls=calls)
    real_verify = rt.verify

    def flaky(*a, **kw):
        if crash["on"]:
            crash["on"] = False
            calls.append(("crash",))
            raise RuntimeError("simulated crash")
        return real_verify(*a, **kw)

    rt.verify = flaky
    saver = open_checkpointer(tmp_path)
    config = thread_config(KEY)
    graph = build_approval_graph(saver, rt, ApprovalRuntime(github=gh.client))
    with pytest.raises(RuntimeError, match="simulated"):
        graph.invoke(approval_state(), config)
    snap = graph.get_state(config)
    assert snap.next == ("judge",) and approval_cli.last_completed(snap) == "review"
    close_checkpointer(saver)

    saver2 = open_checkpointer(tmp_path)  # a "new process": new saver, new graph object
    graph2 = build_approval_graph(saver2, rt, ApprovalRuntime(github=gh.client))
    graph2.invoke(None, config)
    names = [c[0] for c in calls]
    assert names.count("orchestrate") == 1  # review did not run again
    assert names.count("verify") == 1 and names.count("crash") == 1
    assert graph2.get_state(config).next == ("approve",)
    close_checkpointer(saver2)


# --- The CLI ---------------------------------------------------------------------------------


class Keys:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def cli_env(tmp_path, answers=(), gh=None, **rt_kw):
    gh = gh or FakeGitHub()
    out: list[str] = []
    err: list[str] = []
    keys = Keys(answers)
    git_calls: list[list[str]] = []
    deps = approval_cli.Deps(
        github=gh.client,
        runtime=lambda: fake_runtime([CATALOG, CART, LOW], rejected=[LOW], **rt_kw),
        git=lambda args, cwd: git_calls.append(list(args)),
        ask=keys,
        out=out.append,
        err=err.append,
    )

    def run(*argv):
        return approval_cli.main(["--state-dir", str(tmp_path / "st"), *argv], deps)

    return run, gh, out, err, keys, git_calls


def key_from(out):
    return next(line.split(": ", 1)[1] for line in out if line.startswith("review_key:"))


def test_cli_review_pr_stops_at_the_interrupt(tmp_path):
    run, gh, out, err, _, git_calls = cli_env(tmp_path)
    assert run("review-pr", "--repo", "o/r", "--pr", "7", "--eval-mode") == 0
    key = key_from(out)
    assert key.startswith("o/r#7@abc")
    assert any("2 waiting for approval" in line for line in out)
    assert any(f'approve --key "{key}"' in line for line in out)
    assert gh.posts() == 0
    assert [c[0] for c in git_calls] == ["init", "config", "config", "remote", "fetch", "checkout"]
    assert ["fetch", "-q", "--depth", "1", "origin", "abc"] in git_calls
    assert (tmp_path / "st" / "workspaces").is_dir()
    out.clear()
    assert run("status", "--key", key) == 0
    text = "\n".join(out)
    assert "last completed: present" in text and "next node:      approve" in text
    assert "waiting for approval of 2" in text and "decisions:      0/2" in text
    out.clear()
    assert run("resume", "--key", key) == 0 and "waiting for approval" in out[0]
    out.clear()
    assert run("review-pr", "--repo", "o/r", "--pr", "7", "--eval-mode") == 0
    assert "already exists" in out[0]
    assert run("review-pr", "--repo", "o/r", "--pr", "7", "--key-suffix", "demo") == 0


def test_cli_approve_saves_progress_resumes_and_posts_once(tmp_path):
    run, gh, out, err, keys, _ = cli_env(tmp_path, answers=["a", "q"])
    run("review-pr", "--repo", "o/r", "--pr", "7")
    key = key_from(out)
    out.clear()
    assert run("approve", "--key", key) == 0  # approve the first, then quit
    progress = json.loads(
        (tmp_path / "st" / "decisions")
        .joinpath(next(p.name for p in (tmp_path / "st" / "decisions").iterdir()))
        .read_text()
    )
    assert [d["action"] for d in progress["decisions"].values()] == ["approve"]
    assert gh.posts() == 0
    # come back: only the second finding is asked; edit it, then confirm
    keys.answers = ["e", "New title", "", "low", "y"]
    out.clear()
    assert run("approve", "--key", key) == 0
    assert sum("finding " in line and "---" in line for line in out) == 1
    assert gh.posts() == 1
    assert any(line.startswith("posted review 100") for line in out)
    assert [c["path"] for c in gh.reviews[0]["comments"]] == ["shop/cart.py", "shop/catalog.py"]
    assert "**New title** · severity: low" in gh.reviews[0]["comments"][1]["body"]
    # approve again: dry run shows the plan, a real run does not post twice
    out.clear()
    assert run("approve", "--key", key, "--dry-run") == 0
    assert "would post one review" in "\n".join(out) and gh.posts() == 1
    assert run("approve", "--key", key) == 0
    assert gh.posts() == 1 and any("not posted again" in line for line in out)
    out.clear()
    run("status", "--key", key)
    assert '"existed": true' in "\n".join(out) and "next node:      (finished)" in "\n".join(out)


def test_cli_dry_run_and_decline_do_not_call_github(tmp_path):
    run, gh, out, err, keys, _ = cli_env(tmp_path, answers=["a", "r"])
    run("review-pr", "--repo", "o/r", "--pr", "7")
    key = key_from(out)
    before = len(gh.requests)
    out.clear()
    assert run("approve", "--key", key, "--dry-run") == 0
    assert len(gh.requests) == before and "dry run" in out[-1]
    keys.answers = ["n"]
    assert run("approve", "--key", key) == 0
    assert gh.posts() == 0 and any("not posted" in line for line in out)


def test_cli_skip_leaves_findings_undecided(tmp_path):
    run, gh, out, err, keys, _ = cli_env(tmp_path, answers=["s", "r"])
    run("review-pr", "--repo", "o/r", "--pr", "7")
    key = key_from(out)
    out.clear()
    run("approve", "--key", key)
    assert any("1 finding(s) undecided" in line for line in out) and gh.posts() == 0


def test_cli_resume_after_crash_continues_from_checkpoint(tmp_path):
    calls: list = []
    crash = {"on": True}
    gh = FakeGitHub()
    out: list[str] = []
    rt = fake_runtime([CATALOG, CART], calls=calls)
    real = rt.verify

    def flaky(*a, **kw):
        if crash["on"]:
            crash["on"] = False
            raise RuntimeError("killed")
        return real(*a, **kw)

    rt.verify = flaky
    deps = approval_cli.Deps(
        github=gh.client, runtime=lambda: rt, git=lambda a, c: None, ask=Keys([]),
        out=out.append, err=out.append,
    )  # fmt: skip
    argv = ["--state-dir", str(tmp_path / "st")]
    assert approval_cli.main([*argv, "review-pr", "--repo", "o/r", "--pr", "7"], deps) == 1
    assert any("RuntimeError" in line for line in out)
    key = make_key()
    out.clear()
    approval_cli.main([*argv, "status", "--key", key], deps)
    assert "last completed: review" in "\n".join(out)
    out.clear()
    assert approval_cli.main([*argv, "resume", "--key", key], deps) == 0
    assert "resuming after review; running judge" in out[0]
    assert [c[0] for c in calls].count("orchestrate") == 1


def make_key():
    from reviewer.graph import make_review_key

    return make_review_key(DIFF_TEXT, repo="o/r", pr="7", head_sha="abc")


def test_main_scrubs_token_from_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    err: list[str] = []

    def boom():
        raise RuntimeError(f"bad {TOKEN}")

    deps = approval_cli.Deps(github=boom, err=err.append, out=lambda s: None)
    code = approval_cli.main(
        ["--state-dir", str(tmp_path), "review-pr", "--repo", "o/r", "--pr", "1"], deps
    )
    assert code == 1 and TOKEN not in err[0] and "***" in err[0]


def test_checkout_head_commands(tmp_path):
    calls = []
    ws, dp = checkout_head(
        "https://github.com/o/r.git", "abc", tmp_path / "w", "diff text",
        lambda args, cwd: calls.append(list(args)),
    )  # fmt: skip
    assert ws == tmp_path / "w" / "repo" and dp.read_text() == "diff text"
    assert ["config", "core.autocrlf", "false"] in calls
    assert ["remote", "add", "origin", "https://github.com/o/r.git"] in calls


def test_refs_match_payload_ids():
    graph = make_graph(InMemorySaver(), FakeGitHub())
    config = thread_config(KEY)
    graph.invoke(approval_state(), config)
    ids = {e["id"] for e in payload_of(graph, config)}
    assert ids <= set(assign_refs([CATALOG, CART, LOW]))
    assert FIXTURES.is_dir()
