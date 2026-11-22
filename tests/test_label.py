"""eval/label.py: merging, hiding, resuming and the per-configuration output."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import label as lb
from eval.score import _load_labels  # the scorer's own reader, to prove the format

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def finding(**kw) -> dict:
    base = {
        "file": "pkg/util.py", "line_start": 2, "line_end": 2, "category": "wrong_comparison",
        "severity": "medium", "confidence": 0.8, "title": "T", "explanation": "because",
        "evidence": "return xs[len(xs)]", "agent": "bug_reviewer/sub2",
        "model": "gemini-secret-model",
    }  # fmt: skip
    base.update(kw)
    return base


@pytest.fixture
def world(tmp_path):
    dataset, results = tmp_path / "dataset", tmp_path / "results"
    shutil.copytree(FIXTURES / "dataset", dataset)
    for i in ("item_001", "item_002"):
        (dataset / i / "meta.json").write_text(
            json.dumps({"pr_title": f"PR {i}"}), encoding="utf-8"
        )
    return dataset, results


def save(results: Path, config: str, item: str, findings: list[dict]):
    d = results / config
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{item}.json").write_text(
        json.dumps({"item_id": item, "config": config, "findings": findings}), encoding="utf-8"
    )


def test_only_strict_unmatched_in_diff_findings_are_claims(world):
    dataset, results = world
    hit = finding(category="off_by_one")  # matches planted b2: not a claim
    stray = finding(category="wrong_comparison")  # strict-unmatched, in the diff: a claim
    outside = finding(file="pkg/util.py", line_start=50, line_end=50)  # not in the diff: discarded
    save(results, "single", "item_001", [hit, stray, outside])
    claims = lb.collect_claims(results, dataset)
    assert len(claims) == 1
    assert claims[0].members == [("single", "item_001#1")]


def test_identical_claims_across_configurations_are_merged_once(world):
    dataset, results = world
    a = finding(line_start=2, line_end=3)
    b = finding(line_start=3, line_end=3, title="other wording")  # overlaps a, same category
    c = finding(category="unhandled_exception")  # same place, different category: its own claim
    for cfg in ("single", "orchestrator", "subagents"):
        save(results, cfg, "item_001", [a, c])
    save(results, "verifier", "item_001", [b])
    claims = lb.collect_claims(results, dataset)
    assert len(claims) == 2
    big = max(claims, key=lambda x: len(x.members))
    assert sorted(big.members) == [
        ("orchestrator", "item_001#0"), ("single", "item_001#0"),
        ("subagents", "item_001#0"), ("verifier", "item_001#0"),
    ]  # fmt: skip


def test_overlap_groups_are_transitive_but_other_items_and_files_stay_apart(world):
    dataset, results = world
    f1 = finding(line_start=1, line_end=2)
    f2 = finding(line_start=2, line_end=4)
    f3 = finding(line_start=4, line_end=5)  # overlaps f2 only: still one claim
    other_file = finding(file="pkg/calc.py", line_start=11, line_end=11)
    save(results, "single", "item_001", [f1, f2, f3, other_file])
    claims = lb.collect_claims(results, dataset)
    sizes = sorted(len(c.members) for c in claims)
    assert sizes == [1, 3]


def test_screen_never_shows_configuration_agent_or_model(world):
    dataset, results = world
    save(results, "orchestrator", "item_001", [finding()])
    claim = lb.collect_claims(results, dataset)[0]
    code = lb.code_around(dataset.parent / "ws", "pkg/util.py", 2, 2)
    text = lb.render_claim(claim, "PR item_001", code, "[1/1]")
    for secret in ("orchestrator", "bug_reviewer", "sub2", "gemini-secret-model", "confidence"):
        assert secret not in text
    for shown in (
        "PR item_001",
        "pkg/util.py",
        "2-2",
        "wrong_comparison",
        "because",
        "xs[len(xs)]",
    ):
        assert shown in text


def test_code_around_marks_the_claim_lines_and_clips_at_file_ends(tmp_path):
    ws = tmp_path / "ws"
    (ws / "pkg").mkdir(parents=True)
    (ws / "pkg" / "m.py").write_text(
        "\n".join(f"line{n}" for n in range(1, 41)) + "\n", encoding="utf-8"
    )
    text = lb.code_around(ws, "pkg/m.py", 20, 21)
    lines = text.split("\n")
    assert lines[0].endswith("line5") and lines[-1].endswith("line36")
    marked = [ln for ln in lines if ln.startswith(">")]
    assert marked == [">    20 | line20", ">    21 | line21"]
    head = lb.code_around(ws, "pkg/m.py", 2, 2)
    assert head.split("\n")[0].endswith("line1")
    assert "not found" in lb.code_around(ws, "pkg/none.py", 1, 1)


def run(claims, answers, tmp_path, state):
    it = iter(answers)
    shown: list[str] = []
    out = lb.run_session(
        claims,
        title_of=lambda i: f"PR {i}",
        workspace_of=lambda i: tmp_path / "ws",
        save=lambda: lb.save_state(state, claims),
        input_fn=lambda prompt: next(it),
        print_fn=shown.append,
    )
    return out, shown


def make_claims(world):
    dataset, results = world
    save(
        results,
        "single",
        "item_001",
        [finding(line_start=2), finding(file="pkg/calc.py", line_start=11, line_end=11)],
    )
    save(results, "orchestrator", "item_001", [finding(line_start=2)])
    return lb.collect_claims(results, dataset)


def test_session_saves_every_answer_and_resumes(world, tmp_path):
    claims = make_claims(world)
    assert len(claims) == 2
    state = tmp_path / "labels" / "claims.json"
    out, _ = run(claims, ["r", "q"], tmp_path, state)
    assert out == "quit"
    assert sum(1 for c in claims if c.label == "real_bug") == 1
    # A fresh process rebuilds the claims and carries the label over: only one claim is left.
    fresh = lb.collect_claims(*reversed(world))
    lb.load_state(state, fresh)
    assert sorted(str(c.label) for c in fresh) == ["None", "real_bug"]
    out, shown = run(fresh, ["f"], tmp_path, state)
    assert out == "done"
    assert len([s for s in shown if s.startswith("=")]) == 1  # only the unlabelled one was shown
    assert sorted(c.label for c in fresh) == ["false_alarm", "real_bug"]


def test_skip_comes_back_last_and_invalid_input_is_asked_again(world, tmp_path):
    claims = make_claims(world)
    state = tmp_path / "claims.json"
    out, shown = run(claims, ["s", "x", "r", "f"], tmp_path, state)
    # First claim skipped, second answered after an invalid "x", then the skipped one comes back.
    assert out == "done"
    assert any("Please answer" in s for s in shown)
    assert sorted(c.label for c in claims) == ["false_alarm", "real_bug"]


def test_eof_counts_as_quit(world, tmp_path):
    claims = make_claims(world)
    state = tmp_path / "claims.json"

    def boom(prompt):
        raise EOFError

    out = lb.run_session(
        claims, title_of=lambda i: "t", workspace_of=lambda i: tmp_path,
        save=lambda: lb.save_state(state, claims), input_fn=boom, print_fn=lambda s: None,
    )  # fmt: skip
    assert out == "quit" and state.is_file()


def test_manual_labels_are_written_per_configuration_in_the_scorers_format(world, tmp_path):
    dataset, results = world
    claims = make_claims(world)
    by_title = {c.file: c for c in claims}
    by_title["pkg/util.py"].label = "real_bug"
    by_title["pkg/calc.py"].label = "skipped"  # skipped: left out
    save(results, "verifier", "item_001", [])  # a configuration with nothing to label
    written = lb.write_manual_labels(claims, results)
    assert written == ["single", "orchestrator", "verifier"] or sorted(written) == sorted(
        ["single", "orchestrator", "verifier"]
    )
    single = json.loads((results / "single" / "manual_labels.json").read_bytes())
    assert single == {"item_001#0": "real_bug"}  # the util.py finding of `single`; calc.py skipped
    orch = json.loads((results / "orchestrator" / "manual_labels.json").read_bytes())
    assert orch == {"item_001#0": "real_bug"}  # the same claim, applied to the other configuration
    assert json.loads((results / "verifier" / "manual_labels.json").read_bytes()) == {}
    # The scorer accepts the files.
    assert _load_labels(results / "single", {"item_001#0", "item_001#1"}) == single


def test_resume_matches_labels_by_member_not_by_position(world, tmp_path):
    dataset, results = world
    claims = make_claims(world)
    claims[0].label = "false_alarm"
    state = tmp_path / "claims.json"
    lb.save_state(state, claims)
    again = lb.collect_claims(results, dataset)
    lb.load_state(state, again)
    got = {tuple(c.members): c.label for c in again}
    want = {tuple(c.members): c.label for c in claims}
    assert got == want


def test_shuffle_is_fixed(world):
    dataset, results = world
    cats = ["wrong_comparison", "unhandled_exception", "sql_injection", "other"]
    save(
        results, "single", "item_001", [finding(category=c, line_start=1, line_end=1) for c in cats]
    )
    a = [c.category for c in lb.collect_claims(results, dataset)]
    b = [c.category for c in lb.collect_claims(results, dataset)]
    assert a == b and sorted(a) == sorted(cats)


def test_status_and_main_write(world, tmp_path, capsys):
    dataset, results = world
    claims = make_claims(world)
    for c in claims:
        c.label = "false_alarm"
    labels = tmp_path / "claims.json"
    lb.save_state(labels, claims)
    rc = lb.main(
        ["--write", "--results", str(results), "--dataset", str(dataset), "--labels", str(labels)]
    )
    assert rc == 0
    assert (results / "single" / "manual_labels.json").is_file()
    assert "2 claims: 0 real, 2 false alarms" in capsys.readouterr().out
