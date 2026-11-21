"""``reviewer review --agent graph`` (stage 4B, decision 0013): argument rules, the report and the
exit code, with the graph replaced by a fake run. No model, network or Docker call."""

from __future__ import annotations

import json

import pytest

from reviewer import cli
from reviewer import graph as graph_mod
from tests.test_graph import DIFF, WORKSPACE, fake_runtime, initial
from tests.test_judging import CART, CATALOG, LOW

BASE = ["review", "--agent", "graph", "--workspace", WORKSPACE, "--diff", DIFF, "--eval-mode"]


def final_state(judge="verifier", **kw):
    rt = fake_runtime([CATALOG, CART, LOW], rejected=[LOW], **kw)
    return graph_mod.run_graph(initial(judge), rt)


def test_judge_and_graph_argument_rules(capsys):
    with pytest.raises(SystemExit):
        cli.main(["review", "--workspace", WORKSPACE, "--diff", DIFF, "--judge", "none"])
    assert "--judge is for --agent graph" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["review", "--agent", "graph", "--item", "item_001"])
    assert "not run on dataset items" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([*BASE, "--judge", "maybe"])
    with pytest.raises(SystemExit):
        cli.main([*BASE, "--focus", "x"])


def test_report_lists_the_payload_in_order_then_dropped_then_stage_totals():
    text = cli.format_graph_report(final_state())
    lines = text.splitlines()
    assert lines[0].endswith("judge: verifier")
    first = next(i for i, ln in enumerate(lines) if ln.startswith("  1. "))
    assert "[high] shop/cart.py:22-23 missing_none_check" in lines[first]
    assert "(test_reviewer/sub1)" in lines[first] and lines[first + 2].startswith("     verdict: ")
    assert (
        lines[first + 3].startswith("  2. ") and "[medium] shop/catalog.py:36" in lines[first + 3]
    )
    assert "dropped by the judge (1):" in text and "[verifier_rejected]" in text
    assert text.index("approval payload") < text.index("dropped by the judge")
    assert text.index("dropped by the judge") < text.index("stages:")
    stages = text.split("stages:\n")[1].splitlines()
    assert [s.split()[0] for s in stages] == ["review", "judge", "total"]
    assert "9 model / 7 tool calls" in stages[0] and "4 model / 2 tool calls" in stages[1]
    assert "13 model / 9 tool calls" in stages[2]


def test_judge_none_report_says_unchecked():
    text = cli.format_graph_report(final_state("none"))
    assert "verdict: unchecked" in text and "dropped by the judge (0):" in text


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [({}, 0), ({"step_status": "partial"}, 5), ({"step_status": "error"}, 1),
     ({"step_status": "quota_paused"}, 3), ({"review_status": "limit_reached"}, 4)],
)  # fmt: skip
def test_exit_code_follows_the_graph_status(monkeypatch, capsys, tmp_path, kwargs, code):
    state = final_state(**kwargs)
    seen = {}

    def fake_run(initial_state, runtime=None, **kw):
        seen["state"], seen["config"] = initial_state, runtime.config
        return state

    monkeypatch.setattr(graph_mod, "run_graph", fake_run)
    out = tmp_path / "state.json"
    assert cli.main([*BASE, "--judge", "orchestrator", "--no-spawn", "--out", str(out)]) == code
    s = seen["state"]
    assert s["judge"] == "orchestrator" and s["spawn_enabled"] is False and s["evaluation"] is True
    assert s["workspace"] == WORKSPACE and s["review_key"].startswith("local#adhoc@")
    assert json.loads(out.read_bytes())["status"] == state["status"]
    assert "approval payload" in capsys.readouterr().out


def test_default_judge_is_the_verifier(monkeypatch):
    seen = {}
    state = final_state()
    monkeypatch.setattr(graph_mod, "run_graph", lambda s, rt=None, **kw: seen.update(s) or state)
    assert cli.main(BASE) == 0 and seen["judge"] == "verifier"
