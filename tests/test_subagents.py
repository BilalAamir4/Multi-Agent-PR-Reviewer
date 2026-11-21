"""A specialist's sub-agents (stage 4A, decision 0012): specs and prompts, the fan-out through a
real ``StateGraph`` with ``Send``, the specialist's flow (cross-file pass, merge, status), the
accounting through the real callback flow with parallel tasks and tracing on and off, and the
rule that with spawning off nothing changes.

No network, model or Docker call: ``create_agent`` runs on deterministic fake chat models.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from reviewer.agents import orchestrator as orch
from reviewer.agents import specialists, subagents
from reviewer.agents.base import (
    ReviewResult,
    compose_prompt,
    fragment,
    run_agent,
)
from reviewer.agents.orchestrator import SpecialistAbort, SpecialistCollector
from reviewer.agents.specialists import NAMES, run_specialist, specialist_spec, user_message
from reviewer.agents.subagents import (
    Batch,
    coordination_spec,
    plan_batches,
    should_spawn_for,
    subagent_spec,
)
from reviewer.config import ReviewConfig, load_models_config, load_review_config
from reviewer.models import QuotaRouter, QuotaTracker, review_waits
from reviewer.schemas import Finding
from tests.dev_pr import expected_defects, make_fixture_workspace, pr_text
from tests.test_single_agent import USAGE, FakeTime, ai, answer, tool_call
from tests.test_usage_accounting import fake_tracing  # noqa: F401  (a fixture)

REAL: ReviewConfig = load_review_config()  # spawning on, 4 files / 3 per batch / 5 batches
EVAL_MODEL = load_models_config().evaluation.model
FIXTURE = "dev_multi_pr"
TITLE, BODY = pr_text(FIXTURE)
BUG, SEC, TST = NAMES

# The prompt versions recorded when the specialists and the orchestrator were built (decisions
# 0008 and 0009). Rule 1 of stage 4A: the base prompts are unchanged.
PROMPT_VERSIONS = {
    "bug_reviewer": "0a0cf383d913",
    "security_reviewer": "6f396484c1bb",
    "test_reviewer": "c7c71887b22e",
    "orchestrator": "4eda79561a8c",
    "single": "aa94f1ae2cf6",
}


# --- Fixtures and a fake model ---------------------------------------------------------------


@pytest.fixture
def ft():
    return FakeTime()


@pytest.fixture
def router(tmp_path, ft):
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    tracker = QuotaTracker(tmp_path / "quota_state.json", providers)
    return QuotaRouter(
        config, evaluation=True, tracker=tracker, limiters={}, clock=ft.clock, sleep=ft.sleep
    )


@pytest.fixture
def ctx(tmp_path):
    return make_fixture_workspace(FIXTURE, tmp_path)


def finding(ctx, file: str, line: int, category: str, confidence: float = 0.8, **over) -> dict:
    """A valid draft at ``file:line``, its evidence copied from the file."""
    text = (ctx.workspace / file).read_text(encoding="utf-8").splitlines()[line - 1]
    out = {
        "file": file, "line_start": line, "line_end": line, "category": category,
        "severity": "medium", "confidence": confidence, "title": f"A problem at {file}:{line}",
        "explanation": "Why.", "evidence": text.strip(),
    }  # fmt: skip
    out.update(over)
    return out


class SpawnModel(BaseChatModel):
    """Answers by role, read from the system prompt: the orchestrator ("lead reviewer") lists the
    files, calls the three specialists in one turn and answers; a sub-agent ("Your assigned
    files") lists the files and answers with ``sub_answer(assigned files)``; a specialist's
    cross-file pass ("Coordination with your sub-agents") answers with ``cross_findings``; a plain
    specialist lists the files and answers with ``cross_findings``. Every sub-agent's first call
    sleeps a little and is counted, so ``max_active`` is how many ran at the same time."""

    sub_answer: object = None  # callable(files: list[str]) -> list[dict]
    cross_findings: list = []
    first_call_sleep: float = 0.15
    lock: object = None
    state: dict = {}

    @property
    def _llm_type(self) -> str:
        return "spawn-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _assigned(self, system: str) -> list[str]:
        block = system.split("Your assigned files are:")[1].split("Call `read_diff`")[0]
        return [ln.strip()[3:-1] for ln in block.splitlines() if ln.strip().startswith("- `")]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        tools_seen = sum(1 for m in messages if m.type == "tool")
        if "lead reviewer" in system:
            if not any(m.type == "ai" for m in messages):
                msg = ai(tool_call("list_changed_files"))
            elif tools_seen == 1:
                msg = ai(*[tool_call(f"call_{n}", focus="look") for n in NAMES])
            else:
                plan = [{"specialist": n, "called": True, "reason": "r"} for n in NAMES]
                msg = ai(tool_call("OrchestratorOutput", plan=plan, summary="ok"))
        elif "Your assigned files are:" in system:
            if tools_seen == 0:
                with self.lock:
                    self.state["active"] = self.state.get("active", 0) + 1
                    self.state["max_active"] = max(
                        self.state.get("max_active", 0), self.state["active"]
                    )
                time.sleep(self.first_call_sleep)
                msg = ai(tool_call("list_changed_files"))
            else:
                with self.lock:
                    self.state["active"] -= 1
                files = self._assigned(system)
                found = self.sub_answer(files) if self.sub_answer else []
                msg = answer(found, summary=f"covered {', '.join(files)}")
        elif "Coordination with your sub-agents" in system:
            msg = answer(self.cross_findings, summary="cross-file pass")
        elif tools_seen == 0:
            msg = ai(tool_call("list_changed_files"))
        else:
            msg = answer(self.cross_findings)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_model(**kw) -> SpawnModel:
    return SpawnModel(lock=threading.Lock(), state={}, **kw)


def spawn_cfg(**spawn) -> ReviewConfig:
    return REAL.with_spawn(**spawn)


def run_spec(name, ctx, router, ft, cfg, focus=None, **kw):
    return run_specialist(
        name, ctx, pr_title=TITLE, pr_body=BODY, focus=focus, router=router, evaluation=True,
        clock=ft.clock, config=cfg, **kw,
    )  # fmt: skip


# --- Specs and prompts -----------------------------------------------------------------------


def test_the_base_prompts_are_unchanged():
    """Rule 1: no specialist prompt file, no orchestrator prompt and no single prompt changed."""
    for name in NAMES:
        assert specialist_spec(name).prompt_version() == PROMPT_VERSIONS[name]
        assert specialist_spec(name).appendix == ""
    assert orch.orchestrator_spec().prompt_version() == PROMPT_VERSIONS["orchestrator"]
    from reviewer.agents import single

    assert single.prompt_version() == PROMPT_VERSIONS["single"]


def test_a_subagent_is_its_parent_with_one_fragment_added():
    parent = specialist_spec(BUG)
    batch = Batch(index=2, files=("inventory/reports.py", "tests/test_integration.py"),
                  estimated_tokens=1)  # fmt: skip
    sub = subagent_spec(parent, batch, REAL.limits.subagent)
    assert sub.name == "bug_reviewer/sub2" and sub.run_name == "bug_reviewer/sub2-review"
    assert sub.role == "worker"
    assert sub.limits == REAL.limits.subagent
    assert (sub.limits.max_model_calls, sub.limits.max_tool_calls) == (15, 20)
    assert sub.limits.max_run_tests_calls == 1
    assert sub.tools == parent.tools  # the same tools as the specialist
    assert sub.prompt_path == parent.prompt_path and sub.focus_categories == parent.focus_categories
    prompt = sub.system_prompt()
    assert prompt.startswith(compose_prompt(parent.prompt_path, parent.focus_categories))
    tail = prompt[len(compose_prompt(parent.prompt_path, parent.focus_categories)) :]
    assert tail.startswith("\n\n# Your assignment")
    assert "sub-agents of the `bug_reviewer` reviewer" in tail
    assert "- `inventory/reports.py`\n- `tests/test_integration.py`" in tail
    assert "Call `read_diff` on every assigned file" in tail
    assert "Report findings only in your assigned files" in tail
    assert 'takes precedence over "read every changed file"' in tail
    assert sub.prompt_version() != parent.prompt_version()


def test_the_test_subagent_has_run_tests_limited_to_one_call():
    parent = specialist_spec(TST)
    assert "run_tests" in parent.tools and parent.limits.max_run_tests_calls == 3
    sub = subagent_spec(parent, Batch(1, ("a.py",), 1), REAL.limits.subagent)
    assert "run_tests" in sub.tools and sub.limits.max_run_tests_calls == 1


def test_the_coordination_pass_is_the_specialist_with_one_fragment_added():
    parent = specialist_spec(SEC)
    coord = coordination_spec(parent)
    base = compose_prompt(parent.prompt_path, parent.focus_categories)
    assert coord.system_prompt() == base + "\n\n" + fragment("coordination")
    assert coord.name == SEC and coord.limits == parent.limits and coord.tools == parent.tools
    assert coord.run_name == "security_reviewer-cross-file"
    assert coord.prompt_version() != parent.prompt_version()
    text = fragment("coordination")
    assert "defects that span files" in text and "Do not repeat the sub-agents' findings" in text
    assert 'takes precedence over "read every changed file"' in text


def test_the_fragments_are_new_files_and_the_prompt_files_are_untouched():
    root = Path(__file__).resolve().parents[1]
    for name in ("coordination", "subagent_assignment"):
        assert (root / "src/reviewer/prompts/common" / f"{name}.md").is_file()
    for name in (*NAMES, "orchestrator", "single"):
        data = (root / f"src/reviewer/prompts/{name}.md").read_bytes()
        assert b"{assigned_files}" not in data and b"Coordination with" not in data
    single = (root / "src/reviewer/prompts/single.md").read_bytes()
    assert hashlib.sha256(single).hexdigest()[:12] == PROMPT_VERSIONS["single"]


def test_a_prompt_with_no_appendix_is_composed_byte_for_byte_as_before():
    spec = specialist_spec(BUG)
    assert spec.system_prompt() == compose_prompt(spec.prompt_path, spec.focus_categories)
    assert spec.run_name == "bug_reviewer-review"


# --- Spawn off: identical to stage 3 (rule 2) ------------------------------------------------


def _capture_run_agent(monkeypatch, module):
    calls = []

    def fake(spec, ctx, message, router, **kw):
        calls.append((spec, message, kw))
        return ReviewResult(
            status="complete", agent=spec.name, prompt_version=spec.prompt_version(),
            limits=spec.limits, evaluation=True, model_calls=3,
        )  # fmt: skip

    monkeypatch.setattr(module, "run_agent", fake)
    return calls


@pytest.mark.parametrize("name", NAMES)
def test_spawn_off_calls_run_agent_exactly_as_stage_3_did(monkeypatch, ctx, router, ft, name):
    calls = _capture_run_agent(monkeypatch, specialists)
    sub_calls = _capture_run_agent(monkeypatch, subagents)
    cfg = spawn_cfg(enabled=False)
    assert should_spawn_for(name, ctx, cfg.spawn).spawn  # it would have spawned
    parent_config = {"tags": ["x"]}
    result = run_spec(name, ctx, router, ft, cfg, focus="look", parent_config=parent_config)
    assert sub_calls == [] and len(calls) == 1
    spec, message, kw = calls[0]
    assert spec == specialist_spec(name, cfg)  # the stage 3 spec: no appendix, run name -review
    assert spec.appendix == "" and spec.run_name == f"{name}-review"
    assert message == user_message(TITLE, BODY, "look")
    assert kw == {"item_id": None, "evaluation": True, "clock": ft.clock,
                  "parent_config": parent_config}  # fmt: skip
    # and the result is run_agent's own, untouched
    assert result.spawn is None and result.children == [] and result.model_calls == 3
    assert (result.own_model_calls, result.wait_s_wall) == (0, 0.0)


def test_spawn_off_and_a_small_diff_run_the_same_agent(monkeypatch, tmp_path, router, ft):
    """Spawning on but the share is small: the same call, plus the decision recorded."""
    calls = _capture_run_agent(monkeypatch, specialists)
    small = make_fixture_workspace("dev_tests_pr", tmp_path / "s")
    result = run_spec(TST, small, router, ft, spawn_cfg(enabled=True))
    assert len(calls) == 1 and calls[0][0] == specialist_spec(TST)
    assert result.spawn is not None and result.spawn.spawned is False and result.spawn.enabled
    assert result.spawn.py_files == 2 and result.children == []


def test_the_whole_review_is_identical_with_spawning_off(tmp_path, ft):
    """Two real runs on the same scripted model, one through ``run_specialist`` with spawning off
    and one straight through ``run_agent`` as stage 3 did, give the same result."""
    out = []
    for i in range(2):
        ctx = make_fixture_workspace(FIXTURE, tmp_path / f"r{i}")
        config = load_models_config()
        providers = {m: s.provider for m, s in config.models.items()}
        r = QuotaRouter(
            config, evaluation=True, tracker=QuotaTracker(tmp_path / f"q{i}.json", providers),
            limiters={}, clock=ft.clock, sleep=ft.sleep,
        )  # fmt: skip
        found = [finding(ctx, "inventory/pricing.py", 23, "wrong_comparison")]
        r.build = lambda model_id, f=found: make_model(cross_findings=f)
        if i == 0:
            out.append(run_spec(BUG, ctx, r, ft, spawn_cfg(enabled=False)))
        else:
            out.append(
                run_agent(specialist_spec(BUG), ctx, user_message(TITLE, BODY), r,
                          evaluation=True, clock=ft.clock)
            )  # fmt: skip
    drop = {"item_id", "langsmith_run_id"}
    a, b = (r.model_dump(mode="json", exclude=drop) for r in out)
    assert a == b
    assert out[0].status == "complete" and len(out[0].findings) == 1
    assert out[0].spawn is None and out[0].children == []


def test_the_orchestrator_with_spawning_off_has_no_children_below_the_specialists(ctx, router, ft):
    router.build = lambda model_id: make_model(cross_findings=[])
    cfg = spawn_cfg(enabled=False)
    res = orch.orchestrate(ctx, TITLE, BODY, router, clock=ft.clock, config=cfg)
    assert res.status == "complete", (res.error_class, res.error, res.notes)
    assert [c.agent for c in res.children] == list(NAMES)
    assert all(c.children == [] and c.spawn is None for c in res.children)
    assert all(c.prompt_version == PROMPT_VERSIONS[c.agent] for c in res.children)
    assert (res.own_model_calls, res.model_calls) == (3, 3 + 3 * 2)


# --- The fan-out and the specialist's flow ---------------------------------------------------


def two_batches(ctx):
    return plan_batches(list(should_spawn_for(BUG, ctx, REAL.spawn).area), ctx, REAL.spawn)


def sub_findings(ctx, per_file: dict[str, tuple[int, str]]):
    """A ``sub_answer``: for each assigned file it knows, one finding."""

    def answer_for(files):
        return [finding(ctx, f, *per_file[f]) for f in files if f in per_file]

    return answer_for


def test_a_large_share_spawns_one_subagent_per_batch(ctx, router, ft):
    plan = two_batches(ctx)
    assert [b.files for b in plan.batches] == [
        ("inventory/pricing.py", "tests/test_pricing.py"),
        ("inventory/reports.py", "inventory/stock.py", "tests/test_integration.py"),
    ]
    model = make_model(
        sub_answer=sub_findings(
            ctx, {"inventory/pricing.py": (23, "wrong_comparison"),
                  "inventory/stock.py": (25, "off_by_one")}),
        cross_findings=[finding(ctx, "inventory/reports.py", 17, "missing_none_check", 0.7)],
    )  # fmt: skip
    router.build = lambda model_id: model
    res = run_spec(BUG, ctx, router, ft, REAL)
    assert res.status == "complete", (res.error_class, res.error, res.notes)
    assert res.agent == BUG
    assert [c.agent for c in res.children] == ["bug_reviewer/sub1", "bug_reviewer/sub2"]
    assert all(c.status == "complete" for c in res.children)
    # every finding keeps who found it
    by_agent = {(f.agent, f.file) for f in res.findings}
    assert by_agent == {
        ("bug_reviewer/sub1", "inventory/pricing.py"),
        ("bug_reviewer/sub2", "inventory/stock.py"),
        ("bug_reviewer", "inventory/reports.py"),
    }
    # the spawn record
    sp = res.spawn
    assert sp is not None and sp.spawned and sp.triggers == ["files"]
    assert (sp.py_files, sp.min_py_files, sp.max_concurrency) == (5, 4, 3)
    assert [b.agent for b in sp.batches] == ["bug_reviewer/sub1", "bug_reviewer/sub2"]
    assert [t.agent for t in sp.fan_out.timings] == ["bug_reviewer/sub1", "bug_reviewer/sub2"]
    assert sp.cross_file is not None and sp.cross_file.status == "complete"
    # prompt_version is the hash of the composed prompt actually used (coordination applies)
    assert res.prompt_version == coordination_spec(specialist_spec(BUG)).prompt_version()
    assert res.prompt_version != PROMPT_VERSIONS[BUG]


def test_the_subagents_run_in_parallel_up_to_the_concurrency(ctx, router, ft):
    for concurrency, expected in ((3, 2), (2, 2), (1, 1)):
        model = make_model()
        router.build = lambda model_id, m=model: m
        res = run_spec(BUG, ctx, router, ft, spawn_cfg(max_concurrency=concurrency))
        assert res.status == "complete"
        assert model.state["max_active"] == expected, concurrency
        assert res.spawn.max_concurrency == concurrency


def test_the_specialists_cross_file_pass_sees_the_subagents_findings(ctx, router, ft):
    seen = {}

    class Recording(SpawnModel):
        def _generate(self, messages, *a, **kw):
            if "Coordination with your sub-agents" in messages[0].content:
                seen["user"] = messages[1].content
                seen["system"] = messages[0].content
            return super()._generate(messages, *a, **kw)

    model = Recording(
        lock=threading.Lock(), state={},
        sub_answer=sub_findings(ctx, {"inventory/pricing.py": (23, "wrong_comparison")}),
    )  # fmt: skip
    router.build = lambda model_id: model
    run_spec(BUG, ctx, router, ft, REAL)
    user = seen["user"]
    assert user.startswith(user_message(TITLE, BODY))  # the specialist's own message first
    assert "<subagent_reports>" in user and user.rstrip().endswith("</subagent_reports>")
    assert "specialist: bug_reviewer/sub1" in user and "specialist: bug_reviewer/sub2" in user
    assert "bug_reviewer/sub1#1 | inventory/pricing.py:23-23 | wrong_comparison" in user
    assert "summary: covered inventory/pricing.py, tests/test_pricing.py" in user
    assert seen["system"].rstrip().endswith(fragment("coordination"))


def test_a_finding_outside_the_assigned_files_is_dropped_as_not_assigned(ctx, router, ft):
    # Every sub-agent reports a defect in stock.py; only the one assigned that file may keep it.
    model = make_model(
        sub_answer=lambda files: [finding(ctx, "inventory/stock.py", 25, "off_by_one")]
    )
    router.build = lambda model_id: model
    res = run_spec(BUG, ctx, router, ft, REAL)
    sub1, sub2 = res.children
    assert sub1.findings == [] and [d.reason for d in sub1.dropped] == ["not_assigned"]
    assert "inventory/stock.py" in sub1.dropped[0].detail and "pricing.py" in sub1.dropped[0].detail
    assert [f.file for f in sub2.findings] == ["inventory/stock.py"]
    assert sub2.dropped == []
    assert [(f.agent, f.file) for f in res.findings] == [
        ("bug_reviewer/sub2", "inventory/stock.py")
    ]
    assert any(
        d.reason == "not_assigned" and "[bug_reviewer/sub1]" in d.detail for d in res.dropped
    )


# --- Merge, status, aborts (the real graph, a fake run_agent) --------------------------------


def rr(agent, status="complete", findings=(), **kw) -> ReviewResult:
    data = {
        "status": status, "agent": agent, "findings": list(findings),
        "prompt_version": "abc123def456", "limits": REAL.limits.subagent, "evaluation": True,
        "model_id": EVAL_MODEL, "model_calls": 2, "tool_calls": 1,
        "tool_calls_by_tool": {"list_changed_files": 1}, "input_tokens": 2000,
        "output_tokens": 100, "summary": f"{agent} summary",
    }  # fmt: skip
    data.update(kw)
    return ReviewResult(**data)


def fnd(
    ctx, agent, line=23, category="wrong_comparison", confidence=0.8, file="inventory/pricing.py"
):
    d = finding(ctx, file, line, category, confidence)
    return Finding(**{**d, "agent": agent, "model": EVAL_MODEL})


class FakeAgents:
    """A ``run_agent`` replacement: one result per agent name; records who ran, and when."""

    def __init__(self, results: dict[str, ReviewResult]):
        self.results = results
        self.ran: list[str] = []
        self.messages: dict[str, str] = {}
        self.lock = threading.Lock()

    def __call__(self, spec, ctx, message, router, **kw):
        with self.lock:
            self.ran.append(spec.name)
            self.messages[spec.name] = message
        return self.results[spec.name]


def spawn_with(monkeypatch, ctx, router, ft, results, cfg=None):
    fake = FakeAgents(results)
    monkeypatch.setattr(subagents, "run_agent", fake)
    res = run_spec(BUG, ctx, router, ft, cfg or REAL)
    return res, fake


def stock_fnd(ctx, agent, confidence=0.8):
    return fnd(ctx, agent, 25, "off_by_one", confidence, file="inventory/stock.py")


def test_equal_confidence_duplicates_keep_the_earlier_agent(monkeypatch, ctx, router, ft):
    """Sub-agents have disjoint files, so duplicates are between a sub-agent and the specialist's
    own cross-file pass: on a tie the sub-agent's finding is kept."""
    results = {
        "bug_reviewer/sub1": rr("bug_reviewer/sub1", findings=[fnd(ctx, "bug_reviewer/sub1")]),
        "bug_reviewer/sub2": rr(
            "bug_reviewer/sub2", findings=[stock_fnd(ctx, "bug_reviewer/sub2")]
        ),
        "bug_reviewer": rr(
            "bug_reviewer", findings=[fnd(ctx, "bug_reviewer"), stock_fnd(ctx, "bug_reviewer")]
        ),
    }
    res, _ = spawn_with(monkeypatch, ctx, router, ft, results)
    assert [f.agent for f in res.findings] == ["bug_reviewer/sub1", "bug_reviewer/sub2"]
    dropped = [d for d in res.dropped if d.reason == "duplicate"]
    assert [d.draft["finding_id"] for d in dropped] == ["bug_reviewer#1", "bug_reviewer#2"]
    assert [d.draft["duplicate_of"] for d in dropped] == [
        "bug_reviewer/sub1#1",
        "bug_reviewer/sub2#1",
    ]


def test_a_higher_confidence_beats_an_earlier_agent_and_the_specialist_comes_last(
    monkeypatch, ctx, router, ft
):
    results = {
        "bug_reviewer/sub1": rr(
            "bug_reviewer/sub1", findings=[fnd(ctx, "bug_reviewer/sub1", confidence=0.6)]
        ),
        "bug_reviewer/sub2": rr("bug_reviewer/sub2"),
        "bug_reviewer": rr("bug_reviewer", findings=[fnd(ctx, "bug_reviewer", confidence=0.9)]),
    }
    res, _ = spawn_with(monkeypatch, ctx, router, ft, results)
    assert [f.agent for f in res.findings] == ["bug_reviewer"]
    # distinct findings stay in candidate order: sub-agents first, the specialist's own last
    other = {
        **results,
        "bug_reviewer": rr("bug_reviewer", findings=[stock_fnd(ctx, "bug_reviewer")]),
    }
    res2, _ = spawn_with(monkeypatch, ctx, router, ft, other)
    assert [f.agent for f in res2.findings] == ["bug_reviewer/sub1", "bug_reviewer"]


def test_the_totals_count_each_agent_once_and_own_is_the_cross_file_pass(
    monkeypatch, ctx, router, ft
):
    results = {
        "bug_reviewer/sub1": rr("bug_reviewer/sub1"),
        "bug_reviewer/sub2": rr("bug_reviewer/sub2"),
        "bug_reviewer": rr("bug_reviewer", model_calls=1, tool_calls=0, tool_calls_by_tool={}),
    }
    res, _ = spawn_with(monkeypatch, ctx, router, ft, results)
    assert (res.model_calls, res.tool_calls) == (2 + 2 + 1, 1 + 1 + 0)
    assert res.tool_calls_by_tool == {"list_changed_files": 2}
    assert (res.own_model_calls, res.own_tool_calls, res.own_tool_calls_by_tool) == (1, 0, {})
    assert res.input_tokens == 6000 and res.output_tokens == 300
    assert res.own_input_tokens == 2000 and res.own_output_tokens == 100


def test_complete_only_if_the_specialist_and_every_subagent_are_complete(
    monkeypatch, ctx, router, ft
):
    base = {n: rr(n) for n in ("bug_reviewer/sub1", "bug_reviewer/sub2", "bug_reviewer")}
    res, fake = spawn_with(monkeypatch, ctx, router, ft, base)
    assert res.status == "complete" and res.notes == []
    assert fake.ran[-1] == "bug_reviewer"  # the cross-file pass runs after the fan-out

    for bad in ("limit_reached", "error", "partial"):
        kw = {"error_class": "NoStructuredResponse", "error": "x"} if bad == "error" else {}
        results = {**base, "bug_reviewer/sub2": rr("bug_reviewer/sub2", bad, **kw)}
        res, fake = spawn_with(monkeypatch, ctx, router, ft, results)
        assert res.status == "partial", bad
        assert f"bug_reviewer/sub2 ended {bad}; its findings are kept" in res.notes
        assert "bug_reviewer" in fake.ran  # the specialist goes on


def test_the_cross_file_pass_failing_decides_the_specialists_status(monkeypatch, ctx, router, ft):
    kept = fnd(ctx, "bug_reviewer/sub1")
    base = {"bug_reviewer/sub1": rr("bug_reviewer/sub1", findings=[kept]),
            "bug_reviewer/sub2": rr("bug_reviewer/sub2")}  # fmt: skip
    lim = {**base, "bug_reviewer": rr("bug_reviewer", "limit_reached", limit="max_model_calls")}
    res, _ = spawn_with(monkeypatch, ctx, router, ft, lim)
    assert res.status == "limit_reached" and res.limit == "max_model_calls"
    assert [f.agent for f in res.findings] == ["bug_reviewer/sub1"]  # the children's are kept
    assert any("cross-file pass ended limit_reached" in n for n in res.notes)
    err = {
        **base,
        "bug_reviewer": rr("bug_reviewer", "error", error_class="NoStructuredResponse", error="no"),
    }
    res, _ = spawn_with(monkeypatch, ctx, router, ft, err)
    assert (res.status, res.error_class) == ("error", "NoStructuredResponse")
    assert [f.agent for f in res.findings] == ["bug_reviewer/sub1"]
    part = {**base, "bug_reviewer": rr("bug_reviewer", "partial", notes=["sandbox unavailable: x"])}
    res, _ = spawn_with(monkeypatch, ctx, router, ft, part)
    assert res.status == "partial" and "sandbox unavailable: x" in res.notes


@pytest.mark.parametrize(
    ("kw", "status", "klass"),
    [
        (
            {"status": "quota_paused", "error_class": "QuotaPaused", "error": "daily"},
            "quota_paused",
            "QuotaPaused",
        ),
        (
            {"status": "error", "error_class": "RetriesExhausted", "error": "503"},
            "error",
            "RetriesExhausted",
        ),
        (
            {"status": "error", "error_class": "AllExhausted", "error": "all"},
            "error",
            "AllExhausted",
        ),
    ],
    ids=["QuotaPaused", "RetriesExhausted", "AllExhausted"],
)
def test_router_conditions_in_a_subagent_end_the_review(
    monkeypatch, ctx, router, ft, kw, status, klass
):
    results = {
        "bug_reviewer/sub1": rr("bug_reviewer/sub1", findings=[fnd(ctx, "bug_reviewer/sub1")]),
        "bug_reviewer/sub2": rr("bug_reviewer/sub2", **kw),
        "bug_reviewer": rr("bug_reviewer"),
    }
    res, fake = spawn_with(monkeypatch, ctx, router, ft, results, spawn_cfg(max_concurrency=1))
    assert (res.status, res.error_class) == (status, klass)
    assert "bug_reviewer" not in fake.ran  # no cross-file pass after an abort
    assert [c.agent for c in res.children] == ["bug_reviewer/sub1", "bug_reviewer/sub2"]
    assert any("bug_reviewer/sub2 ended the review" in n for n in res.notes)
    assert res.spawn.cross_file is None
    # the orchestrator's collector ends the whole review on it
    runner = lambda name, focus, cfg: res  # noqa: E731
    with pytest.raises(SpecialistAbort):
        SpecialistCollector(runner).call(BUG, "x", {})


def test_other_subagent_failures_do_not_stop_the_others(monkeypatch, ctx, router, ft):
    results = {
        "bug_reviewer/sub1": rr(
            "bug_reviewer/sub1", "error", error_class="ServerError", error="500"
        ),
        "bug_reviewer/sub2": rr(
            "bug_reviewer/sub2", findings=[stock_fnd(ctx, "bug_reviewer/sub2")]
        ),
        "bug_reviewer": rr("bug_reviewer"),
    }
    res, fake = spawn_with(monkeypatch, ctx, router, ft, results, spawn_cfg(max_concurrency=1))
    assert res.status == "partial" and sorted(fake.ran) == sorted(results)
    assert [f.agent for f in res.findings] == ["bug_reviewer/sub2"]


# --- Waits (D7) nest -------------------------------------------------------------------------


def test_fan_out_waits_nest_and_net_time_removes_the_wall_clock_wait(monkeypatch, ctx, router, ft):
    """Sequentially (concurrency 1) on the fake clock: each sub-agent waits 2 s and works 3 s; the
    cross-file pass waits 1 s and works 1 s."""

    def fake(spec, ctx_, message, router_, *, clock, **kw):
        wait, work = (1.0, 1.0) if spec.name == "bug_reviewer" else (2.0, 3.0)
        with review_waits() as log:
            ft.sleep(wait)
            log.record(EVAL_MODEL, "rate_limiter", wait, ft.clock())
            ft.sleep(work)
            total, own_wait = wait + work, log.total()
        return rr(spec.name).model_copy(
            update={"duration_s": total, "wait_s": own_wait, "net_duration_s": total - own_wait}
        )

    monkeypatch.setattr(subagents, "run_agent", fake)
    with review_waits() as outer:
        res = run_spec(BUG, ctx, router, ft, spawn_cfg(max_concurrency=1))
    fan = res.spawn.fan_out
    assert (fan.wall_s, fan.wait_s_sum, fan.wait_s_wall, fan.net_s) == (10.0, 4.0, 4.0, 6.0)
    assert [(t.start_s, t.end_s) for t in fan.timings] == [(0.0, 5.0), (5.0, 10.0)]
    cross = res.spawn.cross_file
    assert (cross.duration_s, cross.wait_s, cross.net_duration_s) == (2.0, 1.0, 1.0)
    assert res.duration_s == 12.0 and res.wait_s == res.wait_s_sum == 5.0
    assert res.wait_s_wall == 5.0 and res.net_duration_s == 7.0
    assert res.waits_by_reason == {"rate_limiter": 5.0}
    assert outer.total() == 5.0 and outer.wall() == 5.0  # joined the caller's log once


# --- The real callback flow: parallel Send, tracing off and on --------------------------------


def run_flow(tmp_path, ft, *, concurrency=3):
    ctx = make_fixture_workspace(FIXTURE, tmp_path)
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    router = QuotaRouter(
        config, evaluation=True, tracker=QuotaTracker(tmp_path / "q.json", providers),
        limiters={}, clock=ft.clock, sleep=ft.sleep,
    )  # fmt: skip
    model = make_model(
        sub_answer=sub_findings(ctx, {"inventory/pricing.py": (23, "wrong_comparison")})
    )
    router.build = lambda model_id: model
    cfg = spawn_cfg(max_concurrency=concurrency)
    res = orch.orchestrate(ctx, TITLE, BODY, router, clock=ft.clock, config=cfg)
    return res, model


@pytest.mark.parametrize("tracing", [False, True], ids=["tracing_off", "tracing_on"])
def test_parallel_subagents_below_parallel_specialists_are_counted_once(
    tmp_path, ft, request, tracing
):
    if tracing:
        request.getfixturevalue("fake_tracing")
    res, model = run_flow(tmp_path, ft)
    assert res.status == "complete", (res.error_class, res.error, res.notes)
    assert model.state["max_active"] >= 4  # sub-agents of several specialists ran at once
    # The orchestrator alone: list the files, call the specialists, answer.
    assert (res.own_model_calls, res.own_tool_calls) == (3, 4)
    assert res.own_input_tokens == 3 * USAGE["input_tokens"]
    assert res.own_output_tokens == 3 * USAGE["output_tokens"]
    assert [c.agent for c in res.children] == list(NAMES)
    for c in res.children:
        assert c.spawn.spawned and len(c.children) == 2
        # each sub-agent: list the files, answer; the cross-file pass: answer
        for sub in c.children:
            assert (sub.model_calls, sub.tool_calls) == (2, 1)
            assert sub.input_tokens == 2 * USAGE["input_tokens"]
            assert sub.tool_calls_by_tool == {"list_changed_files": 1}
        assert (c.own_model_calls, c.own_tool_calls) == (1, 0)
        assert (c.model_calls, c.tool_calls) == (2 + 2 + 1, 1 + 1 + 0)
        assert c.input_tokens == 5 * USAGE["input_tokens"]
        assert c.output_tokens == 5 * USAGE["output_tokens"]
        assert c.tool_calls_by_tool == {"list_changed_files": 2}
    # Totals are own plus all descendants, nothing twice.
    assert res.model_calls == 3 + 3 * 5
    assert res.tool_calls == 4 + 3 * 2
    assert res.input_tokens == 18 * USAGE["input_tokens"]
    assert res.output_tokens == 18 * USAGE["output_tokens"]


@pytest.mark.parametrize("tracing", [False, True], ids=["tracing_off", "tracing_on"])
def test_a_standalone_spawning_specialist_is_counted_once(tmp_path, ft, request, tracing):
    if tracing:
        request.getfixturevalue("fake_tracing")
    ctx = make_fixture_workspace(FIXTURE, tmp_path)
    config = load_models_config()
    providers = {m: s.provider for m, s in config.models.items()}
    router = QuotaRouter(
        config, evaluation=True, tracker=QuotaTracker(tmp_path / "q.json", providers),
        limiters={}, clock=ft.clock, sleep=ft.sleep,
    )  # fmt: skip
    model = make_model()
    router.build = lambda model_id: model
    res = run_spec(BUG, ctx, router, ft, REAL)
    assert (res.model_calls, res.tool_calls) == (5, 2) and model.state["max_active"] == 2
    assert (res.own_model_calls, res.own_tool_calls) == (1, 0)
    if tracing:  # every attempt has its own LangSmith run id, nested under the specialist's run
        ids = [c.langsmith_run_id for c in res.children] + [res.langsmith_run_id]
        assert all(ids) and len(set(ids)) == 3


def test_the_orchestrator_with_the_expected_defects_is_informational():
    """The planted defects of ``dev_multi_pr`` are listed (three logic ones, the cross-file one
    flagged, and the SQL surface added in stage 4B)."""
    defects = expected_defects(FIXTURE)
    assert [d["kind"] for d in defects] == [
        "wrong_comparison", "off_by_one", "missing_none_check", "sql_injection",
    ]  # fmt: skip
    assert [bool(d.get("cross_file")) for d in defects] == [False, False, True, False]
