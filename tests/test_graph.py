"""The review graph (stage 4B, decision 0013): flow, state, idempotent nodes, statuses, the
checkpointer slot, and the accounting through the real callback flow (the orchestrator, the
verifier's ``Send`` fan-out and the judge, tracing off and on). No network, model or Docker call."""

from __future__ import annotations

import json
import re
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from reviewer import graph as graph_mod
from reviewer.agents.judging import JudgeStepResult, VerificationRecord, assign_refs
from reviewer.agents.orchestrator import OrchestratorResult
from reviewer.config import load_models_config, load_review_config
from reviewer.graph import GraphRuntime, build_graph, make_review_key, run_graph, stage_totals
from tests.dev_pr import BODY, FIXTURES, TITLE
from tests.test_judging import CART, CATALOG, LOW, make_router
from tests.test_single_agent import USAGE, FakeTime, ai, answer, tool_call
from tests.test_usage_accounting import fake_tracing  # noqa: F401  (a fixture)

CONFIG = load_review_config()
EVAL_MODEL = load_models_config().evaluation.model
WORKSPACE = str(FIXTURES / "dev_pr" / "workspace")
DIFF = str(FIXTURES / "dev_pr" / "pr.diff")
LIMITS = CONFIG.limits.orchestrator


def initial(judge="verifier", **kw):
    state = {
        "review_key": "local#adhoc@abc",
        "pr_title": TITLE,
        "pr_body": BODY,
        "workspace": WORKSPACE,
        "diff_path": DIFF,
        "sandbox_repo": None,
        "security_mode": None,
        "spawn_enabled": False,
        "judge": judge,
        "evaluation": True,
    }
    return {**state, **kw}


# --- Fakes for the unit tests ----------------------------------------------------------------


def review_result(findings, status="complete", notes=(), **kw):
    return OrchestratorResult(
        status=status, findings=list(findings), notes=list(notes), prompt_version="p",
        limits=LIMITS, evaluation=True, summary="Adds coupons.", model_calls=9, tool_calls=7,
        input_tokens=9000, output_tokens=450, duration_s=20.0, wait_s=8.0, wait_s_sum=8.0,
        wait_s_wall=5.0, net_duration_s=15.0,
        plan=[{"specialist": "bug_reviewer", "called": True, "reason": "source"}], **kw,
    )  # fmt: skip


def step_result(findings, rejected=(), status="complete", judge="verifier", **kw):
    all_findings = list(findings)
    refs = assign_refs(all_findings)
    kept = [f for f in all_findings if f not in rejected]
    verifications = [
        VerificationRecord(
            finding_ref=r, verdict="rejected" if f in rejected else "confirmed", reason=f"why {i}"
        )
        for i, (r, f) in enumerate(zip(refs, all_findings, strict=True))
    ]
    dropped = [
        {"draft": {**f.model_dump(mode="json"), "finding_id": r}, "reason": "verifier_rejected",
         "detail": "why"}
        for r, f in zip(refs, all_findings, strict=True) if f in rejected
    ]  # fmt: skip
    return JudgeStepResult(
        status=status, agent=judge, judge=judge, prompt_version="p", limits=LIMITS, evaluation=True,
        findings=kept, dropped=dropped, verifications=verifications, input_findings=len(refs),
        model_calls=4, tool_calls=2, input_tokens=4000, output_tokens=200, duration_s=6.0,
        wait_s=2.0, wait_s_sum=2.0, wait_s_wall=1.0, net_duration_s=5.0, **kw,
    )  # fmt: skip


def fake_runtime(
    found, *, rejected=(), review_status="complete", step_status="complete", calls=None
):
    calls = calls if calls is not None else []

    def orchestrate(ctx, title, body, router, **kw):
        calls.append(("orchestrate", kw["parent_config"] is not None))
        return review_result(found, review_status, notes=["a note"])

    def verify(ctx, title, findings, router, **kw):
        calls.append(("verify", len(findings)))
        return step_result(findings, rejected, step_status)

    def judge(ctx, plan, summary, findings, router, **kw):
        calls.append(("judge", plan, summary, len(findings)))
        return step_result(findings, rejected, step_status, judge="orchestrator")

    return GraphRuntime(
        router_factory=lambda evaluation: object(), orchestrate=orchestrate, verify=verify,
        judge=judge, config=CONFIG,
    )  # fmt: skip


# --- Flow ------------------------------------------------------------------------------------


def test_verifier_flow_sorts_findings_and_builds_the_approval_payload():
    calls: list = []
    rt = fake_runtime([CATALOG, CART, LOW], rejected=[LOW], calls=calls)
    final = run_graph(initial(), rt)
    assert [c[0] for c in calls] == ["orchestrate", "verify"] and calls[0][1] is True
    assert final["status"] == "complete" and final["notes"] == ["a note"]
    # high before medium; the rejected low finding is gone
    assert [f["title"] for f in final["final_findings"]] == [CART.title, CATALOG.title]
    payload = final["approval_payload"]
    assert [e["title"] for e in payload] == [CART.title, CATALOG.title]
    refs = assign_refs([CATALOG, CART, LOW])
    first = payload[0]
    assert first["id"] == refs[1] and first["agent"] == "test_reviewer/sub1"
    assert (first["file"], first["line_start"], first["line_end"]) == ("shop/cart.py", 22, 23)
    assert (first["category"], first["severity"], first["confidence"]) == (
        "missing_none_check", "high", 0.8)  # fmt: skip
    assert first["evidence"] == CART.evidence and first["explanation"] == CART.explanation
    assert first["verdict"] == "confirmed" and first["verdict_reason"] == "why 1"
    assert final["judge_result"]["dropped"][0]["reason"] == "verifier_rejected"
    json.dumps(final)  # the whole state is JSON-serializable


def test_orchestrator_judge_gets_the_saved_plan_and_summary():
    calls: list = []
    final = run_graph(initial("orchestrator"), fake_runtime([CATALOG], calls=calls))
    assert calls[1][0] == "judge" and calls[1][2] == "Adds coupons." and calls[1][3] == 1
    assert calls[1][1][0]["specialist"] == "bug_reviewer"
    assert final["judge_result"]["judge"] == "orchestrator"


def test_judge_none_passes_findings_through():
    calls: list = []
    final = run_graph(initial("none"), fake_runtime([CATALOG, CART], calls=calls))
    assert [c[0] for c in calls] == ["orchestrate"]
    assert final["judge_result"] is None and len(final["final_findings"]) == 2
    assert {e["verdict"] for e in final["approval_payload"]} == {"unchecked"}
    assert all(e["verdict_reason"] is None for e in final["approval_payload"])


def test_same_severity_sorts_by_confidence_then_file_and_line():
    a = CATALOG.model_copy(update={"confidence": 0.5, "title": "a"})
    b = CATALOG.model_copy(update={"confidence": 0.9, "title": "b", "file": "shop/z.py"})
    c = CATALOG.model_copy(update={"confidence": 0.9, "title": "c", "file": "shop/a.py"})
    d = CATALOG.model_copy(update={"confidence": 0.9, "title": "d", "file": "shop/a.py",
                                   "line_start": 3, "line_end": 3})  # fmt: skip
    final = run_graph(initial("none"), fake_runtime([a, b, d, c]))
    assert [f["title"] for f in final["final_findings"]] == ["d", "c", "b", "a"]


def test_review_key_convention():
    assert make_review_key("diff text").startswith("local#adhoc@")
    assert len(make_review_key("diff text").split("@")[1]) == 12
    assert make_review_key("x", repo="o/r", pr="7", head_sha="abc123") == "o/r#7@abc123"
    assert make_review_key("x", pr="item_001") == make_review_key("x", pr="item_001")
    assert make_review_key("x") != make_review_key("y")


# --- Nodes: own keys only, idempotent ---------------------------------------------------------


def test_nodes_return_only_their_own_keys_and_are_idempotent():
    nodes = graph_mod.make_nodes(fake_runtime([CATALOG, CART, LOW], rejected=[LOW]))
    cfg = {"configurable": {}}
    state = initial()
    r1, r2 = nodes["review"](state, cfg), nodes["review"](state, cfg)
    assert r1 == r2 and set(r1) == {"review_result", "findings", "status", "notes"}
    state = {**state, **r1}
    j1, j2 = nodes["judge"](state, cfg), nodes["judge"](state, cfg)
    assert j1 == j2 and set(j1) == {"judge_result", "final_findings", "status", "notes"}
    assert j1["notes"] == r1["notes"]  # rebuilt from review_result: running twice adds nothing
    state = {**state, **j1}
    p1, p2 = nodes["present"](state), nodes["present"](state)
    assert p1 == p2 and set(p1) == {"final_findings", "approval_payload"}
    state = {**state, **p1}
    assert nodes["present"](state) == p1  # present on its own output changes nothing
    assert nodes["judge"]({**state, **p1}, cfg) == j1  # and judge after present too


# --- Statuses --------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["quota_paused", "error"])
def test_judge_is_skipped_when_the_review_aborted(status):
    calls: list = []
    final = run_graph(initial(), fake_runtime([CATALOG], review_status=status, calls=calls))
    assert [c[0] for c in calls] == ["orchestrate"] and final["status"] == status
    assert final["judge_result"] is None and any("judge skipped" in n for n in final["notes"])
    assert len(final["final_findings"]) == 1


@pytest.mark.parametrize("step_status", ["quota_paused", "error"])
def test_an_infrastructure_failure_in_the_judge_aborts_the_review(step_status):
    final = run_graph(initial(), fake_runtime([CATALOG], step_status=step_status))
    assert final["status"] == step_status
    assert any("step ended the review" in n for n in final["notes"])
    assert len(final["final_findings"]) == 1  # the step drops nothing when it aborts


def test_a_partial_judge_step_makes_the_review_partial_with_a_note():
    final = run_graph(initial(), fake_runtime([CATALOG], step_status="partial"))
    assert final["status"] == "partial"
    assert any("verifier step ended partial" in n for n in final["notes"])


def test_review_partial_stays_partial_and_limit_reached_is_kept():
    for status in ("partial", "limit_reached"):
        final = run_graph(initial(), fake_runtime([CATALOG], review_status=status))
        assert final["status"] == status


# --- Totals and the checkpointer slot --------------------------------------------------------


def test_stage_totals_are_the_plain_sum_of_the_two_stages():
    final = run_graph(initial(), fake_runtime([CATALOG, CART]))
    t = stage_totals(final)
    assert t["review"]["model_calls"] == 9 and t["judge"]["model_calls"] == 4
    assert t["total"]["model_calls"] == 13 and t["total"]["input_tokens"] == 13000
    assert t["total"]["net_duration_s"] == 20.0 and t["total"]["wait_s_wall"] == 6.0
    none = stage_totals(run_graph(initial("none"), fake_runtime([CATALOG])))
    assert none["judge"]["model_calls"] == 0 and none["total"] == none["review"]


def test_the_graph_compiles_without_a_checkpointer_and_accepts_one_later():
    assert build_graph() is not None
    saver = InMemorySaver()
    rt = fake_runtime([CATALOG, CART], rejected=[CART])
    final = run_graph(initial(), rt, checkpointer=saver)
    config = {"configurable": {"thread_id": final["review_key"]}}
    saved = build_graph(saver, rt).get_state(config)
    assert saved.values["approval_payload"] == final["approval_payload"]
    assert saved.next == ()  # finished: stage 6 puts an interrupt after "present"
    names = [s.metadata["step"] for s in build_graph(saver, rt).get_state_history(config)]
    assert len(names) >= 4  # one checkpoint per node


# --- The real callback flow ------------------------------------------------------------------


class GraphRoleModel(BaseChatModel):
    """Answers by role from the system prompt (so parallel agents need no ordering): the
    orchestrator calls its three specialists; the bug reviewer finds the catalog defect and the
    test reviewer the cart one; the verifier reads the file once and rejects the cart finding;
    the judge lists the files once and rejects the cart finding."""

    @property
    def _llm_type(self) -> str:
        return "graph-role"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system, user = messages[0].content, messages[1].content
        seen = sum(1 for m in messages if m.type == "tool")
        names = ("bug_reviewer", "security_reviewer", "test_reviewer")
        if "independent checker" in system:
            file = re.search(r"^file: (.+)$", user, re.MULTILINE).group(1)
            if seen == 0:
                msg = ai(tool_call("read_file", path=file))
            else:
                verdict = "rejected" if file == "shop/cart.py" else "confirmed"
                msg = ai(tool_call("Verdict", verdict=verdict, reason=f"{verdict} {file}"))
        elif "Judging your team's findings" in system:
            if seen == 0:
                msg = ai(tool_call("list_changed_files"))
            else:
                found = re.findall(r"id: (F\w+)\nfile: (.+)\n", user)
                verdicts = [
                    {"finding_id": i, "reason": "r",
                     "verdict": "rejected" if f == "shop/cart.py" else "confirmed"}
                    for i, f in found
                ]  # fmt: skip
                msg = ai(tool_call("JudgeOutput", verdicts=verdicts))
        elif "lead reviewer" in system:
            if not any(m.type == "ai" for m in messages):
                msg = ai(tool_call("list_changed_files"))
            elif seen == 1:
                msg = ai(*[tool_call(f"call_{n}", focus="look") for n in names])
            else:
                plan = [{"specialist": n, "called": True, "reason": "r"} for n in names]
                msg = ai(tool_call("OrchestratorOutput", plan=plan, summary="ok"))
        elif seen == 0:
            msg = ai(tool_call("list_changed_files"))
        else:
            from tests.test_judging import CART as C
            from tests.test_judging import CATALOG as K

            def draft(f):
                keys = ("file", "line_start", "line_end", "category", "severity", "confidence",
                        "title", "explanation", "evidence")  # fmt: skip
                return {k: getattr(f, k) for k in keys}

            if "its bug reviewer" in system:
                msg = answer([draft(K)])
            elif "its test reviewer" in system:
                msg = answer([draft(C)])
            else:
                msg = answer([])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def real_runtime(tmp_path, ft):
    router = make_router(tmp_path, ft)
    router.build = lambda model_id: GraphRoleModel()
    return GraphRuntime(router_factory=lambda evaluation: router, clock=ft.clock, config=CONFIG)


@pytest.mark.parametrize("judge", ["verifier", "orchestrator"])
@pytest.mark.parametrize("tracing", [False, True], ids=["tracing_off", "tracing_on"])
def test_graph_accounting_counts_every_call_once(tmp_path, request, tracing, judge):
    created: list[dict] = []
    if tracing:
        request.getfixturevalue("fake_tracing")
    from langsmith import Client

    if tracing:
        mock.patch.object(Client, "create_run", side_effect=lambda **kw: created.append(kw)).start()
        request.addfinalizer(mock.patch.stopall)
    ft = FakeTime()
    final = run_graph(initial(judge), real_runtime(tmp_path, ft))
    assert final["status"] == "complete", (final["status"], final["notes"])
    # The two reviewers' findings reach the judge; it drops the cart finding, keeps the other.
    assert [f["file"] for f in final["findings"]] == ["shop/catalog.py", "shop/cart.py"]
    assert [f["file"] for f in final["final_findings"]] == ["shop/catalog.py"]
    review, step = final["review_result"], final["judge_result"]
    # review: orchestrator 3 own + 3 specialists x 2; tools 1 + 3 calls, 1 per specialist
    assert (review["own_model_calls"], review["model_calls"]) == (3, 9)
    assert (review["own_tool_calls"], review["tool_calls"]) == (4, 7)
    assert [c["model_calls"] for c in review["children"]] == [2, 2, 2]
    if judge == "verifier":
        assert (step["model_calls"], step["tool_calls"]) == (4, 2)  # 2 checks x (read, answer)
        assert [c["model_calls"] for c in step["children"]] == [2, 2]
        assert {v["calls"] for v in step["verifications"]} == {2}
    else:
        assert (step["model_calls"], step["tool_calls"]) == (2, 1)  # one run for both findings
        assert [c["model_calls"] for c in step["children"]] == [2]
    assert (step["own_model_calls"], step["own_tool_calls"]) == (0, 0)
    t = stage_totals(final)
    assert t["total"]["model_calls"] == 9 + step["model_calls"]
    assert t["total"]["tool_calls"] == 7 + step["tool_calls"]
    assert t["total"]["input_tokens"] == t["total"]["model_calls"] * USAGE["input_tokens"]
    assert t["total"]["output_tokens"] == t["total"]["model_calls"] * USAGE["output_tokens"]
    # waits nest: the step's summed wait is its checks' waits (the router's wait before each
    # attempt included), nothing twice; the wall-clock union never exceeds the sum
    assert step["wait_s_sum"] >= sum(c["wait_s"] for c in step["children"]) - 1e-6
    assert 0 <= step["wait_s_wall"] <= step["wait_s_sum"] + 1e-6
    assert review["wait_s_wall"] <= review["wait_s_sum"] + 1e-6
    assert t["total"]["wait_s_sum"] == pytest.approx(review["wait_s_sum"] + step["wait_s_sum"])
    assert t["total"]["net_duration_s"] == pytest.approx(
        review["net_duration_s"] + step["net_duration_s"]
    )
    if tracing:  # one root run named review-graph; every other run is below it
        root = [c for c in created if c.get("parent_run_id") is None]
        assert [c["name"] for c in root] == ["review-graph"]
        assert {c.get("trace_id") for c in created} == {root[0]["trace_id"]}
        names = {c["name"] for c in created}
        assert {"orchestrator-review", "bug_reviewer-review"} <= names
        assert ("verifier-check" if judge == "verifier" else "judge-run") in names
