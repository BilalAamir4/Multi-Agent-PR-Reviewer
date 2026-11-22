"""LangSmith experiments from saved results (stage 5Z, decision 0015). Makes no model calls.

    uv run python -m eval.langsmith_experiments [--config NAME ...] [--dataset-name mapr-eval-v1]
        [--protocol-commit HASH] [--dry-run]

Creates or updates the LangSmith dataset ``mapr-eval-v1`` (one example per dataset item: input
``item_id``, PR title and bucket; reference output: the ground truth) and, for each configuration
whose results are complete, one experiment through ``langsmith.evaluate``. The target function only
returns the findings saved in ``eval/results/<config>/<item>.json``; the evaluators compute the
item's strict and location-only tp, fp and fn (with ``eval.score.score_item``, the frozen rule) and
the review's cost and net time from the runs files. Experiments are named
``<config>@<protocol commit short hash>`` (LangSmith appends its own random suffix).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.ground_truth import load_item
from eval.results_data import (
    CONFIG_ORDER,
    dataset_dir,
    dataset_ids,
    estimated_cost,
    item_bucket,
    item_cost,
    item_status,
    load_run,
    pr_title,
    results_root,
)
from eval.score import MODES, score_item
from reviewer.config import repo_root
from reviewer.schemas import Finding

DATASET_NAME = "mapr-eval-v1"
PROTOCOL_FILE = "docs/decisions/0015-evaluation-protocol.md"


def protocol_commit() -> str:
    """Short hash of the commit that added the protocol record (decision 0015)."""
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%h", "-n", "1", "--", PROTOCOL_FILE],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not out:
        raise RuntimeError(f"no commit adds {PROTOCOL_FILE}")
    return out


def experiment_name(config: str, commit: str) -> str:
    return f"{config}@{commit}"


def complete_configs(results: Path, ids: list[str]) -> list[str]:
    """Configurations (in D6 order) where every item is scored or reused."""
    return [
        c
        for c in CONFIG_ORDER
        if all(item_status(results, c, i)[0] in ("scored", "reused") for i in ids)
    ]


def build_examples(dataset: Path) -> list[dict[str, Any]]:
    """One example per item: inputs ``item_id``, ``pr_title``, ``bucket``; outputs the ground
    truth file."""
    examples = []
    for i in dataset_ids(dataset):
        truth = json.loads((dataset / i / "ground_truth.json").read_bytes())
        examples.append(
            {
                "inputs": {
                    "item_id": i,
                    "pr_title": pr_title(dataset, i),
                    "bucket": item_bucket(dataset, i),
                },
                "outputs": {"ground_truth": truth},
                "metadata": {"item_id": i},
            }
        )
    return examples


def sync_dataset(client: Any, name: str, examples: list[dict[str, Any]]) -> None:
    """Create the dataset if needed, then add the missing examples and update the existing ones
    (matched by ``metadata.item_id``). Running it twice changes nothing."""
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description="Multi-Agent-PR-Reviewer evaluation items (20 PRs)")
    existing = {
        (e.metadata or {}).get("item_id"): e for e in client.list_examples(dataset_name=name)
    }
    for ex in examples:
        found = existing.get(ex["metadata"]["item_id"])
        if found is None:
            client.create_example(
                inputs=ex["inputs"],
                outputs=ex["outputs"],
                metadata=ex["metadata"],
                dataset_name=name,
            )
        else:
            client.update_example(
                found.id, inputs=ex["inputs"], outputs=ex["outputs"], metadata=ex["metadata"]
            )


def make_target(results: Path, config: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The experiment's target: returns the saved findings of the item. No model is called."""

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        path = results / config / f"{inputs['item_id']}.json"
        return {"findings": json.loads(path.read_bytes())["findings"]}

    return target


def make_evaluator(
    results: Path, dataset: Path, config: str
) -> Callable[[Any, Any], dict[str, Any]]:
    """One run evaluator that returns every per-item metric: strict and location-only tp, fp and
    fn, and the review's cost and time."""

    def per_item(run: Any, example: Any) -> dict[str, Any]:
        item_id = example.inputs["item_id"]
        gt, diff_text = load_item(dataset / item_id)
        findings = [Finding.model_validate(f) for f in (run.outputs or {}).get("findings", [])]
        scored = score_item(gt, diff_text, findings)
        out: list[dict[str, Any]] = []
        for mode in MODES:
            m = scored["modes"][mode]
            prefix = "strict" if mode == "strict" else "location"
            out += [
                {"key": f"{prefix}_tp", "score": m["tp"]},
                {"key": f"{prefix}_fp", "score": m["fp"]},
                {"key": f"{prefix}_fn", "score": m["fn"]},
            ]
        out.append({"key": "findings", "score": len(findings)})
        saved = load_run(results, config, item_id)
        if saved is not None:
            cost = item_cost(saved, config)
            usd = estimated_cost(cost["input_tokens"], cost["output_tokens"])
            out += [
                {"key": "model_calls", "score": cost["model_calls"]},
                {"key": "input_tokens", "score": cost["input_tokens"]},
                {"key": "output_tokens", "score": cost["output_tokens"]},
                {"key": "net_time_s", "score": cost["net_s"]},
                {"key": "wait_s", "score": cost["wait_s"]},
            ]
            if usd is not None:
                out.append({"key": "est_cost_usd", "score": usd})
        return {"results": out}

    return per_item


def run_experiments(
    client: Any,
    results: Path,
    dataset: Path,
    commit: str,
    *,
    configs: Sequence[str] | None = None,
    dataset_name: str = DATASET_NAME,
    evaluate_fn: Callable[..., Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Sync the dataset, then run one experiment per complete configuration. Returns the
    experiment prefixes created. ``evaluate_fn`` defaults to ``langsmith.evaluate``."""
    if evaluate_fn is None:
        from langsmith import evaluate as evaluate_fn
    ids = dataset_ids(dataset)
    wanted = list(configs) if configs else list(CONFIG_ORDER)
    done = complete_configs(results, ids)
    sync_dataset(client, dataset_name, build_examples(dataset))
    created = []
    for config in wanted:
        if config not in done:
            log(f"{config}: results incomplete, no experiment")
            continue
        prefix = experiment_name(config, commit)
        evaluate_fn(
            make_target(results, config),
            data=dataset_name,
            evaluators=[make_evaluator(results, dataset, config)],
            experiment_prefix=prefix,
            description=f"Saved findings of `{config}`; no model calls (decision 0015).",
            metadata={"config": config, "protocol_commit": commit},
            client=client,
            max_concurrency=1,
        )
        log(f"{config}: experiment {prefix} created")
        created.append(prefix)
    return created


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.langsmith_experiments", description=__doc__
    )
    parser.add_argument("--config", nargs="+", choices=CONFIG_ORDER)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--protocol-commit", default=None)
    parser.add_argument("--dry-run", action="store_true", help="show what would be created")
    parser.add_argument("--results", type=Path, default=results_root())
    parser.add_argument("--dataset", type=Path, default=dataset_dir())
    args = parser.parse_args(argv)
    load_dotenv(repo_root() / ".env")
    commit = args.protocol_commit or protocol_commit()
    ids = dataset_ids(args.dataset)
    done = complete_configs(args.results, ids)
    wanted = args.config or list(CONFIG_ORDER)
    if args.dry_run:
        print(f"dataset {args.dataset_name}: {len(ids)} examples")
        for c in wanted:
            print(f"{experiment_name(c, commit)}: " + ("would run" if c in done else "incomplete"))
        return 0
    from langsmith import Client

    run_experiments(
        Client(), args.results, args.dataset, commit, configs=wanted, dataset_name=args.dataset_name
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
