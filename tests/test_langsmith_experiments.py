"""eval/langsmith_experiments.py with a fake client and a fake ``evaluate`` (no network)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import langsmith_experiments as lx

FIXTURES = Path(__file__).resolve().parent / "fixtures"

HIT = {
    "file": "pkg/util.py", "line_start": 2, "line_end": 2, "category": "off_by_one",
    "severity": "high", "confidence": 0.9, "title": "t", "explanation": "e",
    "evidence": "return xs[len(xs)]", "agent": "single", "model": "m",
}  # fmt: skip
LOC_ONLY = {
    **HIT,
    "file": "pkg/calc.py",
    "line_start": 11,
    "line_end": 11,
}  # wrong kind: b1 wants none-check


class FakeClient:
    def __init__(self):
        self.datasets: dict[str, list[SimpleNamespace]] = {}
        self.created = 0
        self.updated = 0

    def has_dataset(self, dataset_name):
        return dataset_name in self.datasets

    def create_dataset(self, name, description=None):
        self.datasets[name] = []

    def list_examples(self, dataset_name):
        return list(self.datasets[dataset_name])

    def create_example(self, inputs, outputs, metadata, dataset_name):
        self.created += 1
        self.datasets[dataset_name].append(
            SimpleNamespace(
                id=f"ex{self.created}", inputs=inputs, outputs=outputs, metadata=metadata
            )
        )

    def update_example(self, example_id, inputs, outputs, metadata):
        self.updated += 1
        for e in self.datasets[next(iter(self.datasets))]:
            if e.id == example_id:
                e.inputs, e.outputs, e.metadata = inputs, outputs, metadata


@pytest.fixture
def world(tmp_path):
    dataset, results = tmp_path / "dataset", tmp_path / "results"
    shutil.copytree(FIXTURES / "dataset", dataset)
    for i in ("item_001", "item_002"):
        (dataset / i / "meta.json").write_text(
            json.dumps({"pr_title": f"title {i}"}), encoding="utf-8"
        )
    return dataset, results


def save(results: Path, config: str, item: str, findings: list[dict], **cost):
    base = {"model_calls": 5, "tool_calls": 4, "input_tokens": 1000, "output_tokens": 100,
            "net_duration_s": 4.0, "wait_s": 1.5, "wait_s_wall": 0.5}  # fmt: skip
    base.update(cost)
    (results / config / "runs").mkdir(parents=True, exist_ok=True)
    (results / config / f"{item}.json").write_text(
        json.dumps({"item_id": item, "config": config, "findings": findings}), encoding="utf-8"
    )
    (results / config / "runs" / f"{item}.json").write_text(
        json.dumps({"result": base, "scoring": {"decision": "scored", "cause": "complete"}}),
        encoding="utf-8",
    )


def test_examples_have_inputs_and_reference_outputs(world):
    dataset, _ = world
    exs = lx.build_examples(dataset)
    assert [e["inputs"]["item_id"] for e in exs] == ["item_001", "item_002"]
    assert exs[0]["inputs"] == {
        "item_id": "item_001",
        "pr_title": "title item_001",
        "bucket": "small",
    }
    assert exs[0]["outputs"]["ground_truth"]["bugs"][0]["bug_id"] == "b1"


def test_sync_dataset_is_idempotent(world):
    dataset, _ = world
    client = FakeClient()
    lx.sync_dataset(client, lx.DATASET_NAME, lx.build_examples(dataset))
    lx.sync_dataset(client, lx.DATASET_NAME, lx.build_examples(dataset))
    assert client.created == 2  # created once, then only updated
    assert client.updated == 2
    assert len(client.datasets[lx.DATASET_NAME]) == 2


def test_target_returns_saved_findings_without_a_model(world):
    dataset, results = world
    save(results, "single", "item_001", [HIT])
    target = lx.make_target(results, "single")
    assert target({"item_id": "item_001"}) == {"findings": [HIT]}


def test_evaluator_scores_strict_location_cost_and_time(world):
    dataset, results = world
    save(results, "single", "item_001", [HIT, LOC_ONLY])
    save(results, "single", "item_002", [])
    ev = lx.make_evaluator(results, dataset, "single")
    ex = SimpleNamespace(inputs={"item_id": "item_001"})
    got = {
        r["key"]: r["score"]
        for r in ev(SimpleNamespace(outputs={"findings": [HIT, LOC_ONLY]}), ex)["results"]
    }
    assert (got["strict_tp"], got["strict_fp"], got["strict_fn"]) == (1, 1, 1)
    assert (got["location_tp"], got["location_fp"], got["location_fn"]) == (2, 0, 0)
    assert got["findings"] == 2
    assert got["model_calls"] == 5 and got["net_time_s"] == 4.0 and got["wait_s"] == 1.5
    assert got["est_cost_usd"] == pytest.approx((1000 * 0.25 + 100 * 1.5) / 1e6)


def test_orchestrated_runs_report_the_wall_clock_wait(world):
    dataset, results = world
    save(results, "orchestrator", "item_001", [])
    ev = lx.make_evaluator(results, dataset, "orchestrator")
    ex = SimpleNamespace(inputs={"item_id": "item_001"})
    got = {
        r["key"]: r["score"] for r in ev(SimpleNamespace(outputs={"findings": []}), ex)["results"]
    }
    assert got["wait_s"] == 0.5  # wait_s_wall, not the summed wait


def test_only_complete_configurations_get_an_experiment(world):
    dataset, results = world
    save(results, "single", "item_001", [HIT])
    save(results, "single", "item_002", [])
    save(results, "orchestrator", "item_001", [])  # item_002 missing: incomplete
    client, sink, msgs = FakeClient(), [], []

    def fake_evaluate(target, data, evaluators, experiment_prefix, **kw):
        scores = {}
        for ex in client.datasets[data]:
            run = SimpleNamespace(outputs=target(ex.inputs))
            scores[ex.inputs["item_id"]] = {
                r["key"]: r["score"] for r in evaluators[0](run, ex)["results"]
            }
        sink.append({"prefix": experiment_prefix, "kw": kw, "scores": scores})

    created = lx.run_experiments(
        client, results, dataset, "176125b", evaluate_fn=fake_evaluate, log=msgs.append
    )
    assert created == ["single@176125b"]
    assert sink[0]["prefix"] == "single@176125b"
    assert sink[0]["kw"]["metadata"] == {"config": "single", "protocol_commit": "176125b"}
    assert sink[0]["scores"]["item_001"]["strict_tp"] == 1
    assert sink[0]["scores"]["item_002"]["strict_fp"] == 0
    assert any("orchestrator: results incomplete" in m for m in msgs)


def test_dry_run_makes_no_client(world, capsys):
    dataset, results = world
    save(results, "single", "item_001", [])
    save(results, "single", "item_002", [])
    rc = lx.main(
        [
            "--dry-run",
            "--protocol-commit",
            "abc1234",
            "--results",
            str(results),
            "--dataset",
            str(dataset),
        ]
    )
    out = capsys.readouterr().out
    assert (
        rc == 0 and "single@abc1234: would run" in out and "orchestrator@abc1234: incomplete" in out
    )
