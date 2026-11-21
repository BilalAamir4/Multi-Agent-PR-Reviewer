"""The verifier and the orchestrator as judge (stage 4B, decision 0013): prompts, specs, the
keep-or-drop policy, the per-finding fan-out with real ``Send`` and the single judge run, on
deterministic fake chat models. No network, model or Docker call."""

from __future__ import annotations

import hashlib
import re
import threading

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from reviewer.agents import judge, judging, verifier
from reviewer.agents.base import ReviewResult, compose_prompt, fragment
from reviewer.agents.judging import (
    JudgeOutput,
    JudgeStepResult,
    assign_refs,
    finding_ref,
)
from reviewer.agents.orchestrator import TOOL_NAMES as ORCHESTRATOR_TOOLS
from reviewer.agents.orchestrator import orchestrator_spec
from reviewer.config import load_models_config, load_review_config
from reviewer.models import QuotaRouter, QuotaTracker
from reviewer.schemas import Finding
from tests.dev_pr import TITLE, make_dev_workspace
from tests.test_single_agent import USAGE, FakeTime, ai, tool_call

EVAL_MODEL = load_models_config().evaluation.model
CONFIG = load_review_config()

# Recorded when the prompts were written (stage 4B). A change needs a record in the ADR.
PROMPT_VERSIONS = {"verifier": "4fb90e13fd02", "judge": "ec54742e1b4c"}

CATALOG = Finding(
    file="shop/catalog.py", line_start=36, line_end=36, category="off_by_one", severity="medium",
    confidence=0.9, title="Each page is one product short",
    explanation="The slice end is start + size - 1.",
    evidence="return products[start : start + size - 1]", agent="bug_reviewer", model=EVAL_MODEL,
)  # fmt: skip
CART = Finding(
    file="shop/cart.py", line_start=22, line_end=23, category="missing_none_check",
    severity="high", confidence=0.8, title="apply_coupon reads percent of None",
    explanation="find() returns None for an unknown code.",
    evidence="if coupon.percent > 0:", agent="test_reviewer/sub1", model=EVAL_MODEL,
)  # fmt: skip
LOW = CART.model_copy(update={"file": "shop/coupons.py", "line_start": 1, "line_end": 1,
                              "severity": "low", "confidence": 0.3, "title": "other"})  # fmt: skip


def make_router(tmp_path, ft: FakeTime) -> QuotaRouter:
    providers = {m: s.provider for m, s in load_models_config().models.items()}
    return QuotaRouter(
        load_models_config(), evaluation=True,
        tracker=QuotaTracker(tmp_path / "quota_state.json", providers),
        limiters={}, clock=ft.clock, sleep=ft.sleep,
    )  # fmt: skip


_LOCK = threading.Lock()


class CheckerModel(BaseChatModel):
    """Plays the verifier and the judge. ``rule`` maps a finding's file to its verdict. A verifier
    reads the file once, then answers; the judge lists the files once, then answers for every
    finding id in its user message. ``active`` records how many verifications ran at once."""

    rule: dict[str, str]
    state: dict = {"active": 0, "max_active": 0}
    hold: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "checker"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system, user = messages[0].content, messages[1].content
        tools_seen = sum(1 for m in messages if m.type == "tool")
        if "independent checker" in system:
            file = re.search(r"^file: (.+)$", user, re.MULTILINE).group(1)
            if tools_seen == 0:
                with _LOCK:
                    self.state["active"] += 1
                    self.state["max_active"] = max(self.state["max_active"], self.state["active"])
                if self.hold:
                    threading.Event().wait(self.hold)
                msg = ai(tool_call("read_file", path=file))
            else:
                with _LOCK:
                    self.state["active"] -= 1
                verdict = self.rule.get(file, "confirmed")
                msg = ai(tool_call("Verdict", verdict=verdict, reason=f"{verdict} at {file}"))
        elif tools_seen == 0:
            msg = ai(tool_call("list_changed_files"))
        else:
            verdicts = []
            blocks = re.findall(r"id: (F\w+)\nfile: (.+)\n", user)
            for fid, file in blocks:
                verdicts.append({"finding_id": fid, "verdict": self.rule.get(file, "confirmed"),
                                 "reason": f"judged {file}"})  # fmt: skip
            msg = ai(tool_call("JudgeOutput", verdicts=verdicts))
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def ctx(tmp_path):
    return make_dev_workspace(tmp_path)


@pytest.fixture
def ft():
    return FakeTime()


@pytest.fixture
def router(tmp_path, ft):
    return make_router(tmp_path, ft)


def use(router, rule, hold=0.0):
    model = CheckerModel(rule=rule, state={"active": 0, "max_active": 0}, hold=hold)
    router.build = lambda model_id: model
    return model


# --- Prompts, specs and messages -------------------------------------------------------------


def verdict_block(text: str) -> str:
    return text.split("# Verdicts\n\n", 1)[1].split("\n\n", 1)[0].strip()


def test_verdict_definitions_are_identical_in_both_prompts():
    v = verdict_block(verifier.verifier_spec(CONFIG).system_prompt())
    j = verdict_block(judge.judge_spec(CONFIG).system_prompt())
    assert v == j
    for word in ("`confirmed`", "`rejected`", "`uncertain`", "category label"):
        assert word in v


@pytest.mark.parametrize("name", ["verifier", "judge"])
def test_prompts_mention_nothing_about_evaluation(name):
    spec = verifier.verifier_spec(CONFIG) if name == "verifier" else judge.judge_spec(CONFIG)
    text = spec.system_prompt().lower()
    if name == "judge":  # the orchestrator prompt is frozen; only the new fragment is checked
        text = fragment("judge").lower()
    for word in ("dataset", "planted", "plant", "evaluation", "scoring", "benchmark", "inserted"):
        assert word not in text


def test_verifier_prompt_says_the_finding_is_a_claim_to_check():
    text = verifier.verifier_spec(CONFIG).system_prompt()
    assert "a claim to check, not a fact" in text
    assert fragment("untrusted_input") in text  # the shared untrusted-input fragment


def test_verifier_spec():
    spec = verifier.verifier_spec(CONFIG)
    assert (spec.name, spec.role) == ("verifier", "verifier")
    assert spec.tools == ("read_diff", "read_file", "search_code")
    assert (spec.limits.max_model_calls, spec.limits.max_tool_calls) == (6, 8)
    assert spec.limits is CONFIG.limits.verifier_per_finding


def test_judge_is_the_orchestrator_plus_one_fragment_and_its_own_tools():
    spec = judge.judge_spec(CONFIG)
    orch = orchestrator_spec(CONFIG)
    assert (spec.role, spec.prompt_path) == ("orchestrator", orch.prompt_path)
    assert (
        spec.tools
        == tuple(ORCHESTRATOR_TOOLS)
        == ("list_changed_files", "read_diff", "search_code")
    )
    assert "read_file" not in spec.tools
    assert spec.system_prompt() == orch.system_prompt() + "\n\n" + fragment("judge")
    assert spec.system_prompt().startswith(compose_prompt(orch.prompt_path))
    assert "takes precedence over the planning instructions above" in fragment("judge")
    assert (spec.limits.max_model_calls, spec.limits.max_tool_calls) == (8, 12)


def test_new_prompt_hashes_are_recorded():
    assert verifier.verifier_spec(CONFIG).prompt_version() == PROMPT_VERSIONS["verifier"]
    assert judge.judge_spec(CONFIG).prompt_version() == PROMPT_VERSIONS["judge"]
    assert hashlib.sha256(fragment("judge").encode()).hexdigest()[:12]


def test_verifier_message_hides_the_agent_and_marks_the_claim_untrusted():
    msg = verifier.user_message(TITLE, CART)
    assert "untrusted" in msg and "claim to check, not a fact" in msg
    assert "file: shop/cart.py" in msg and "lines: 22-23" in msg
    assert "category: missing_none_check" in msg and "severity: high" in msg
    assert "evidence:\nif coupon.percent > 0:" in msg
    assert "test_reviewer" not in msg and "sub1" not in msg and "bug_reviewer" not in msg
    assert f"<pr_title>\n{TITLE}\n</pr_title>" in msg


def test_judge_message_has_plan_summary_ids_and_the_finding_agent():
    refs = assign_refs([CATALOG, CART])
    plan = [{"specialist": "bug_reviewer", "called": True, "reason": "source changed"}]
    msg = judge.user_message(plan, "Adds coupons.", [CATALOG, CART], refs)
    assert "- bug_reviewer: called: source changed" in msg and "Adds coupons." in msg
    for ref in refs:
        assert f"id: {ref}" in msg
    assert "found_by: bug_reviewer" in msg and "found_by: test_reviewer/sub1" in msg


def test_finding_ids_are_stable_and_unique():
    assert finding_ref(CATALOG) == finding_ref(CATALOG.model_copy())
    assert finding_ref(CATALOG) != finding_ref(CART)
    twin = CATALOG.model_copy()
    refs = assign_refs([CATALOG, twin])
    assert refs[0] == finding_ref(CATALOG) and refs[1] == refs[0] + "-2"


# --- The verifier ----------------------------------------------------------------------------


def run_v(ctx, router, ft, findings, **kw):
    return verifier.run_verifier(ctx, TITLE, findings, router, config=CONFIG, clock=ft.clock, **kw)


def test_verifier_keeps_confirmed_and_uncertain_and_drops_rejected(ctx, router, ft):
    use(router, {"shop/cart.py": "rejected", "shop/coupons.py": "uncertain"})
    res = run_v(ctx, router, ft, [CATALOG, CART, LOW])
    assert res.status == "complete" and res.judge == "verifier" and res.input_findings == 3
    assert [f.file for f in res.findings] == ["shop/catalog.py", "shop/coupons.py"]
    assert res.findings[0] == CATALOG  # unchanged: same agent, model and every field
    assert [d.reason for d in res.dropped] == ["verifier_rejected"]
    assert res.dropped[0].detail == "rejected at shop/cart.py"
    assert res.dropped[0].draft["file"] == "shop/cart.py"
    by = {v.finding_ref: v for v in res.verifications}
    refs = assign_refs([CATALOG, CART, LOW])
    assert [by[r].verdict for r in refs] == ["confirmed", "rejected", "uncertain"]
    assert all(v.model_id == EVAL_MODEL and v.calls == 2 for v in res.verifications)
    assert all(v.tokens == 2 * (USAGE["input_tokens"] + USAGE["output_tokens"])
               for v in res.verifications)  # fmt: skip


def test_verifier_counts_and_tokens_are_the_checks_sum(ctx, router, ft):
    use(router, {})
    res = run_v(ctx, router, ft, [CATALOG, CART])
    assert len(res.children) == 2 and all(c.agent == "verifier" for c in res.children)
    assert (res.model_calls, res.tool_calls) == (4, 2)
    assert res.tool_calls_by_tool == {"read_file": 2}
    assert res.input_tokens == 4 * USAGE["input_tokens"]
    assert res.output_tokens == 4 * USAGE["output_tokens"]
    assert (res.own_model_calls, res.own_tool_calls) == (0, 0)
    assert res.prompt_version == verifier.verifier_spec(CONFIG).prompt_version()


def test_verifier_with_no_findings_makes_no_model_call(ctx, router, ft):
    router.build = lambda model_id: pytest.fail("no model call expected")
    res = run_v(ctx, router, ft, [])
    assert res.status == "complete" and res.findings == [] and res.model_calls == 0
    assert res.children == [] and res.verifications == []


def test_verifier_runs_checks_in_parallel_up_to_max_concurrency(ctx, router, ft):
    model = use(router, {}, hold=0.2)
    many = [CATALOG.model_copy(update={"line_start": i, "line_end": i, "title": f"t{i}"})
            for i in range(1, 7)]  # fmt: skip
    res = run_v(ctx, router, ft, many)
    assert res.status == "complete" and len(res.findings) == 6
    assert 2 <= model.state["max_active"] <= CONFIG.verifier.max_concurrency == 3


def failing_check(kind):
    def check(index, finding, ref, config):
        status = {"limit": "limit_reached", "error": "error", "paused": "quota_paused",
                  "exhausted": "error"}[kind]  # fmt: skip
        cls = {"error": "NoStructuredResponse", "exhausted": "AllExhausted"}.get(kind)
        child = ReviewResult(
            status=status, agent="verifier", error_class=cls, error=cls, prompt_version="p",
            limits=CONFIG.limits.verifier_per_finding, evaluation=True, model_calls=6,
            input_tokens=10, output_tokens=1, model_id=EVAL_MODEL,
        )  # fmt: skip
        return child, None

    return check


@pytest.mark.parametrize("kind", ["limit", "error"])
def test_a_failed_check_counts_as_uncertain_and_the_finding_is_kept(ctx, router, ft, kind):
    res = run_v(ctx, router, ft, [CATALOG], check=failing_check(kind))
    assert res.findings == [CATALOG] and res.dropped == []
    rec = res.verifications[0]
    assert rec.verdict == "uncertain" and rec.source == (
        "limit_reached" if kind == "limit" else "error"
    )
    assert res.status == "partial" and any("kept as uncertain" in n for n in res.notes)


@pytest.mark.parametrize("kind", ["paused", "exhausted"])
def test_infrastructure_failures_abort_the_step_and_drop_nothing(ctx, router, ft, kind):
    res = run_v(ctx, router, ft, [CATALOG, CART], check=failing_check(kind))
    expected = "quota_paused" if kind == "paused" else "error"
    assert res.status == expected and res.findings == [CATALOG, CART] and res.dropped == []
    assert res.verifications == []
    if kind == "exhausted":
        assert res.error_class == "AllExhausted"


# --- The orchestrator as judge ---------------------------------------------------------------


def run_j(ctx, router, ft, findings, plan=(), summary="s", **kw):
    return judge.run_judge(ctx, list(plan), summary, findings, router, config=CONFIG,
                           clock=ft.clock, **kw)  # fmt: skip


def test_judge_drops_only_rejected_in_one_run(ctx, router, ft):
    use(router, {"shop/cart.py": "rejected", "shop/coupons.py": "uncertain"})
    res = run_j(ctx, router, ft, [CATALOG, CART, LOW])
    assert res.status == "complete" and res.judge == "orchestrator" and res.agent == "judge"
    assert [f.file for f in res.findings] == ["shop/catalog.py", "shop/coupons.py"]
    assert res.findings[0] == CATALOG
    assert [d.reason for d in res.dropped] == ["judge_rejected"]
    assert len(res.children) == 1 and res.children[0].agent == "judge"
    assert (res.model_calls, res.tool_calls) == (2, 1)  # one run for all three findings
    assert all(v.calls is None and v.tokens is None for v in res.verifications)
    assert res.prompt_version == judge.judge_spec(CONFIG).prompt_version()


def test_judge_missing_unknown_and_duplicate_ids(ctx, router, ft):
    refs = assign_refs([CATALOG, CART, LOW])
    out = JudgeOutput.model_validate({"verdicts": [
        {"finding_id": refs[0], "verdict": "rejected", "reason": "no"},
        {"finding_id": refs[0], "verdict": "confirmed", "reason": "yes"},  # duplicate: both ignored
        {"finding_id": "Fnope", "verdict": "rejected", "reason": "?"},  # unknown: ignored
        {"finding_id": refs[1], "verdict": "rejected", "reason": "wrong"},
    ]})  # fmt: skip

    def run(spec, message, parent_config):
        child = ReviewResult(status="complete", agent="judge", prompt_version="p",
                             limits=spec.limits, evaluation=True, model_calls=2)  # fmt: skip
        return child, out

    res = run_j(ctx, router, ft, [CATALOG, CART, LOW], run=run)
    assert res.status == "complete"  # notes, not a status change
    assert [f.title for f in res.findings] == [CATALOG.title, LOW.title]  # missing and dup: kept
    by = {v.finding_ref: v for v in res.verifications}
    assert by[refs[0]].verdict == "uncertain" and by[refs[0]].source == "missing"
    assert by[refs[1]].verdict == "rejected" and by[refs[2]].verdict == "uncertain"
    notes = " | ".join(res.notes)
    assert "unknown finding id 'Fnope'" in notes and "2 verdicts for" in notes
    assert "no valid verdict from the judge" in notes


def test_judge_that_hits_its_limit_keeps_every_finding(ctx, router, ft):
    def run(spec, message, parent_config):
        child = ReviewResult(status="limit_reached", agent="judge", limit="max_tool_calls",
                             prompt_version="p", limits=spec.limits, evaluation=True)  # fmt: skip
        return child, None

    res = run_j(ctx, router, ft, [CATALOG, CART], run=run)
    assert res.status == "limit_reached" and len(res.findings) == 2 and res.dropped == []
    assert {v.source for v in res.verifications} == {"limit_reached"}


def test_judge_infrastructure_failure_aborts(ctx, router, ft):
    def run(spec, message, parent_config):
        child = ReviewResult(
            status="error", agent="judge", error_class="RetriesExhausted", error="x",
            prompt_version="p", limits=spec.limits, evaluation=True,
        )  # fmt: skip
        return child, None

    res = run_j(ctx, router, ft, [CATALOG], run=run)
    assert res.status == "error" and res.error_class == "RetriesExhausted"
    assert res.findings == [CATALOG] and res.dropped == []


def test_judge_with_no_findings_makes_no_model_call(ctx, router, ft):
    router.build = lambda model_id: pytest.fail("no model call expected")
    res = run_j(ctx, router, ft, [])
    assert res.status == "complete" and res.model_calls == 0 and res.children == []


def test_step_result_round_trips_through_json(ctx, router, ft):
    use(router, {"shop/cart.py": "rejected"})
    res = run_j(ctx, router, ft, [CATALOG, CART])
    again = JudgeStepResult.model_validate(res.model_dump(mode="json"))
    assert again == res and judging.finding_ref(again.findings[0]) == finding_ref(CATALOG)
