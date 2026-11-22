"""Read-only helpers over ``eval/results/`` shared by the stage 5 tooling (decision 0015).

Nothing here calls a model or changes a result: it reads the scored files (``item_*.json``), the run
files (``runs/item_*.json``) and the dataset, and estimates the model calls an item will need.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from eval.ground_truth import size_bucket
from reviewer.config import repo_root
from reviewer.models import estimate_tokens

CONFIG_ORDER = ("single", "orchestrator", "subagents", "verifier", "judge")
OPTIONAL = frozenset({"judge"})  # row 4b: only if quota allows
PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini's daily quota resets at midnight Pacific

# Decision 0015: defaults used until 3 items of the configuration and bucket have been observed.
DEFAULT_CALLS = {
    "single": {"small": 10, "medium": 10, "large": 10},
    "orchestrator": {"small": 10, "medium": 30, "large": 70},
    "subagents": {"small": 10, "medium": 30, "large": 70},
}
CALLS_PER_FINDING = 3  # verifier
JUDGE_CALLS = 8  # limits.judge.max_model_calls: one run for all findings
MIN_OBSERVED = 3
# Estimate only (decision 0012, stage 4C): fewer .py files than this and a small diff do not spawn.
SPAWN_MIN_PY_FILES = 4
SPAWN_MAX_TOKENS = 30000


def dataset_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "eval" / "dataset"


def results_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "eval" / "results"


def dataset_ids(dataset: Path | None = None) -> list[str]:
    d = dataset or dataset_dir()
    return sorted(p.name for p in d.iterdir() if (p / "ground_truth.json").is_file())


def diff_text(dataset: Path, item_id: str) -> str:
    return (dataset / item_id / "pr.diff").read_bytes().decode("utf-8")


def item_bucket(dataset: Path, item_id: str) -> str:
    return size_bucket(diff_text(dataset, item_id))


def pr_title(dataset: Path, item_id: str) -> str:
    """The PR title from ``meta.json`` (reporting and labelling only; never a prompt)."""
    return str(json.loads((dataset / item_id / "meta.json").read_bytes())["pr_title"])


def load_run(results: Path, config: str, item_id: str) -> dict[str, Any] | None:
    path = results / config / "runs" / f"{item_id}.json"
    return json.loads(path.read_bytes()) if path.is_file() else None


def load_scored(results: Path, config: str, item_id: str) -> dict[str, Any] | None:
    path = results / config / f"{item_id}.json"
    return json.loads(path.read_bytes()) if path.is_file() else None


def item_status(results: Path, config: str, item_id: str) -> tuple[str, str | None]:
    """``(status, detail)``: ``scored``, ``reused`` (detail = source configuration),
    ``rerun_required`` (detail = cause) or ``missing``.

    The latest scoring decision in the run file wins over an older scored file; stage 1 run files
    have no decision and count as scored when the scored file exists."""
    run = load_run(results, config, item_id)
    scored = load_scored(results, config, item_id) is not None
    if run is not None:
        scoring = run.get("scoring") or {}
        if scoring.get("decision") == "rerun_required":
            return "rerun_required", scoring.get("cause")
        if scored and run.get("reused_from"):
            return "reused", str(run["reused_from"])
    if scored:
        return "scored", None
    return "missing", None


def run_calls(run: dict[str, Any]) -> int:
    return int(run["result"].get("model_calls") or 0)


def would_spawn_estimate(dataset: Path, item_id: str) -> bool:
    """A cheap estimate of D12 from the diff alone (no workspace): 4 or more changed ``.py`` files,
    or a raw diff over the token limit. ``eval.run`` decides for real."""
    from eval.ground_truth import changed_py_file_count

    text = diff_text(dataset, item_id)
    return changed_py_file_count(text) >= SPAWN_MIN_PY_FILES or (
        estimate_tokens(text) > SPAWN_MAX_TOKENS
    )


def findings_count(results: Path, item_id: str) -> int:
    """Findings a judge would see: the saved ``subagents`` ones, else ``orchestrator``'s."""
    for config in ("subagents", "orchestrator"):
        scored = load_scored(results, config, item_id)
        if scored is not None:
            return len(scored["findings"])
    return 0


def observed_mean(
    results: Path, dataset: Path, config: str, bucket: str, ids: list[str]
) -> float | None:
    """Mean model calls of this configuration's own (not reused) runs in the bucket, when at
    least ``MIN_OBSERVED`` exist."""
    calls = []
    for i in ids:
        if item_bucket(dataset, i) != bucket:
            continue
        status, _ = item_status(results, config, i)
        run = load_run(results, config, i)
        if status == "scored" and run is not None:
            calls.append(run_calls(run))
    return statistics.mean(calls) if len(calls) >= MIN_OBSERVED else None


def estimate_calls(
    results: Path, dataset: Path, config: str, item_id: str, ids: list[str] | None = None
) -> float:
    """Estimated model calls of one item for one configuration (decision 0015)."""
    ids = ids if ids is not None else dataset_ids(dataset)
    if config == "verifier":
        return float(CALLS_PER_FINDING * findings_count(results, item_id))
    if config == "judge":
        return float(JUDGE_CALLS if findings_count(results, item_id) else 0)
    if config == "subagents" and not would_spawn_estimate(dataset, item_id):
        return 0.0  # D6: reused from the orchestrator results
    bucket = item_bucket(dataset, item_id)
    seen = observed_mean(results, dataset, config, bucket, ids)
    return float(seen if seen is not None else DEFAULT_CALLS[config][bucket])


def pacific_day(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(PACIFIC).date().isoformat()


def calls_on_pacific_day(results: Path, day: str | None = None) -> int:
    """Model calls recorded in every ``run_log`` entry of every configuration that started on
    ``day`` (Pacific time, the quota's day; default today). Retries are not counted."""
    day = day or datetime.now(UTC).astimezone(PACIFIC).date().isoformat()
    total = 0
    for path in sorted(results.glob("*/runs/item_*.json")):
        for entry in json.loads(path.read_bytes()).get("run_log", []):
            if entry.get("reused_from") or "started_at" not in entry:
                continue
            if pacific_day(entry["started_at"]) == day:
                total += int(entry.get("model_calls") or 0)
    return total


# --- Cost, time and per-agent figures (report and LangSmith experiments) ---------------------

ORCHESTRATED = ("orchestrator", "subagents")  # their wait is the wall-clock union (decision 0009)


def item_cost(run: dict[str, Any], config: str) -> dict[str, float]:
    """One review's cost as the results table reports it: calls, tokens, net time (D7) and wait.

    ``verifier`` and ``judge`` are cumulative (the saved ``subagents`` review plus the step,
    decision 0013). Orchestrated runs remove the wall-clock union of waits (``wait_s_wall``), the
    single agent its summed wait. A reused ``subagents`` item carries the review it was copied
    from."""
    if "cumulative" in run:
        c = run["cumulative"]
        return {
            "model_calls": float(c["model_calls"]),
            "tool_calls": float(c["tool_calls"]),
            "input_tokens": float(c["input_tokens"]),
            "output_tokens": float(c["output_tokens"]),
            "net_s": float(c["net_duration_s"]),
            "wait_s": float(c["wait_s_wall"]),
        }
    r = run["result"]
    wait = r.get("wait_s_wall") if config in ORCHESTRATED else r.get("wait_s")
    return {
        "model_calls": float(r.get("model_calls") or 0),
        "tool_calls": float(r.get("tool_calls") or 0),
        "input_tokens": float(r.get("input_tokens") or 0),
        "output_tokens": float(r.get("output_tokens") or 0),
        "net_s": float(r.get("net_duration_s") or 0),
        "wait_s": float(wait or 0),
    }


def estimated_cost(input_tokens: float, output_tokens: float) -> float | None:
    """USD at the D5 prices in ``config/models.yaml`` (free tier: nothing is billed), or None."""
    from reviewer.config import load_models_config

    models = load_models_config()
    spec = models.models[models.evaluation.model]
    if spec.price_input_per_mtok is None or spec.price_output_per_mtok is None:
        return None
    return (
        input_tokens * spec.price_input_per_mtok + output_tokens * spec.price_output_per_mtok
    ) / 1e6


def agent_breakdown(run: dict[str, Any], config: str) -> list[tuple[str, int, int, int]]:
    """``(agent, model_calls, input_tokens, output_tokens)`` for the agents of one run.

    Orchestrated runs: the orchestrator's own calls, each specialist's own pass (its ``own_*``
    counters when it spawned, else its totals), and its sub-agents together as
    ``<specialist>/sub``. ``single``, ``verifier`` and ``judge``: the one agent (for the judges,
    the judge step alone; the review's cost is in ``item_cost``)."""
    r = run["result"]
    if config not in ORCHESTRATED:
        name = {"single": "single", "verifier": "verifier", "judge": "judge"}[config]
        return [(name, int(r.get("model_calls") or 0), int(r.get("input_tokens") or 0),
                 int(r.get("output_tokens") or 0))]  # fmt: skip
    rows = [(
        "orchestrator",
        int(r.get("own_model_calls") or 0),
        int(r.get("own_input_tokens") or 0),
        int(r.get("own_output_tokens") or 0),
    )]  # fmt: skip
    for child in r.get("children", []):
        subs = child.get("children") or []
        if subs:
            rows.append((
                child["agent"],
                int(child.get("own_model_calls") or 0),
                int(child.get("own_input_tokens") or 0),
                int(child.get("own_output_tokens") or 0),
            ))  # fmt: skip
            rows.append((
                f"{child['agent']}/sub",
                sum(int(s.get("model_calls") or 0) for s in subs),
                sum(int(s.get("input_tokens") or 0) for s in subs),
                sum(int(s.get("output_tokens") or 0) for s in subs),
            ))  # fmt: skip
        else:
            rows.append((
                child["agent"],
                int(child.get("model_calls") or 0),
                int(child.get("input_tokens") or 0),
                int(child.get("output_tokens") or 0),
            ))  # fmt: skip
    return rows
