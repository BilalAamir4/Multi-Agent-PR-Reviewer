"""Run one configuration over the dataset and score it (stage 1B; plan § Evaluation, D4, D7).

    uv run python -m eval.run --config single [--items item_001 item_003 ...] [--force]
        [--reason TEXT]

Always uses ``QuotaRouter(evaluation=True)``: one pinned model, no fallback (D4). For each item
it writes

- ``eval/results/<config>/<item_id>.json``: the scorer's input, only when the review is
  ``complete`` or ``limit_reached`` (a ``limit_reached`` item is scored with no findings);
- ``eval/results/<config>/runs/<item_id>.json``: the full ``ReviewResult`` and a ``run_log``
  of every attempt with its reason. The scorer reads only ``item_*.json`` at the top level.

Items with a scored result are skipped unless ``--force`` is given. On ``quota_paused`` the run
stops with exit code 3. Once every dataset item is scored, ``eval.score`` runs and
``summary.md`` is written. Only ``pr_title`` and ``pr_body`` are read from ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.ground_truth import DatasetError
from eval.score import BUCKETS, MODES, score_config
from reviewer.agents.single import ReviewResult, review
from reviewer.cli import format_reset, item_inputs
from reviewer.config import load_models_config, load_review_config, repo_root
from reviewer.models import QuotaRouter

CONFIGS = ("single",)
SCORED = ("complete", "limit_reached")


def results_dir(config: str, root: Path | None = None) -> Path:
    return (root or repo_root() / "eval" / "results") / config


def dataset_items(dataset: Path) -> list[str]:
    return sorted(p.name for p in dataset.iterdir() if (p / "ground_truth.json").is_file())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _load_run_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(json.loads(path.read_bytes()).get("run_log", []))


def run_item(
    item_id: str, config: str, router: QuotaRouter, out_dir: Path, reason: str
) -> ReviewResult:
    """Review one item and write its run file (always) and scored file (complete or
    limit_reached)."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    ctx, title, body = item_inputs(item_id)
    result = review(ctx, title, body, router, item_id=item_id)
    run_path = out_dir / "runs" / f"{item_id}.json"
    run_log = _load_run_log(run_path)
    run_log.append(
        {
            "started_at": started_at,
            "reason": reason,
            "status": result.status,
            "error": result.error,
            "model_id": result.model_id,
            "prompt_version": result.prompt_version,
            "model_calls": result.model_calls,
            "tool_calls": result.tool_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "duration_s": result.duration_s,
            "wait_s": result.wait_s,
            "attempts": [a.model_dump(mode="json") for a in result.attempts],
        }
    )
    _write_json(run_path, {"result": result.model_dump(mode="json"), "run_log": run_log})
    if result.status in SCORED:
        findings = [f.model_dump(mode="json") for f in result.findings]
        _write_json(
            out_dir / f"{item_id}.json",
            {"item_id": item_id, "config": config, "findings": findings},
        )
    return result


# --- Summary ---------------------------------------------------------------------------------


def _fmt(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def summary_markdown(
    report: dict[str, Any], runs: dict[str, ReviewResult], config: str, today: str
) -> str:
    models = load_models_config()
    spec = models.models[models.evaluation.model]
    limits = load_review_config().limits.single
    agg = report["aggregates"]
    results = list(runs.values())
    model_ids = sorted({r.model_id for r in results if r.model_id})
    versions = sorted({r.prompt_version for r in results})

    out = [f"# Results: `{config}`", ""]
    out += [
        f"- Date: {today} (UTC)",
        f"- Model: {', '.join(f'`{m}`' for m in model_ids) or '-'} (evaluation mode, D4)",
        f"- Prompt version: {', '.join(f'`{v}`' for v in versions)} (`src/reviewer/prompts/"
        f"{config}.md`, SHA-256, first 12 hex)",
        f"- Limits: `max_model_calls` {limits.max_model_calls}, `max_tool_calls` "
        f"{limits.max_tool_calls} (`config/review.yaml` `limits.{config}`)",
        f"- Items: {agg['strict']['all']['items']} ({agg['strict']['all']['clean_items']} clean); "
        f"line tolerance {report['line_tolerance']}",
        "",
        "## Recall and precision",
        "",
        "| Mode | Bucket | Items | Recall | Precision | tp | fp | fn | Discarded |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in MODES:
        for b in BUCKETS:
            a = agg[mode][b]
            out.append(
                f"| {mode} | {b} | {a['items']} | {_fmt(a['recall'])} | {_fmt(a['precision'])} "
                f"| {a['tp']} | {a['fp']} | {a['fn']} | {a['discarded']} |"
            )
    fa_strict = agg["strict"]["all"]["false_alarms_per_clean_pr"]
    fa_loc = agg["location_only"]["all"]["false_alarms_per_clean_pr"]
    out += [
        "",
        f"- False alarms per clean PR: strict {_fmt(fa_strict)}, location-only {_fmt(fa_loc)}",
        "- Adjusted precision (D9b): not computed; no manual labels yet.",
        "",
        "## Per item",
        "",
        "| Item | Bucket | Status | Findings | tp | fp | fn | Loc tp | Discarded | Dropped "
        "| Model calls | Tool calls | Tokens in / out | Net s | Wait s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    ]
    for it in report["items"]:
        r = runs.get(it["item_id"])
        s, loc = it["modes"]["strict"], it["modes"]["location_only"]
        if r is None:
            extra = "| - | - | - | - | - | - | - |"
        else:
            dropped = Counter(d.reason for d in r.dropped)
            dtext = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) or "0"
            extra = (
                f"| {r.status} | {it['findings']} | {s['tp']} | {s['fp']} | {s['fn']} "
                f"| {loc['tp']} | {it['discarded']} | {dtext} | {r.model_calls} "
                f"| {r.tool_calls} | {r.input_tokens} / {r.output_tokens} "
                f"| {r.net_duration_s:.1f} | {r.wait_s:.1f} |"
            )
        out.append(f"| {it['item_id']} | {it['size_bucket']} {extra}")

    net = [r.net_duration_s for r in results]
    wait = [r.wait_s for r in results]
    tin = [r.input_tokens for r in results]
    tout = [r.output_tokens for r in results]
    mean_in = statistics.mean(tin) if tin else None
    mean_out = statistics.mean(tout) if tout else None
    cost = None
    if (
        mean_in is not None
        and mean_out is not None
        and spec.price_input_per_mtok is not None
        and spec.price_output_per_mtok is not None
    ):
        cost = (mean_in * spec.price_input_per_mtok + mean_out * spec.price_output_per_mtok) / 1e6
    dropped_all = Counter(d.reason for r in results for d in r.dropped)
    out += [
        "",
        "## Cost and time",
        "",
        f"- Median review time with rate-limit waiting removed (D7): {_fmt(_median(net), 1)} s; "
        f"median wait beside it: {_fmt(_median(wait), 1)} s",
        f"- Mean tokens per review: {_fmt(mean_in, 0)} input, {_fmt(mean_out, 0)} output",
        f"- Estimated cost per review: ${_fmt(cost, 5)} at ${_fmt(spec.price_input_per_mtok)} / "
        f"${_fmt(spec.price_output_per_mtok)} per million input / output tokens (D5 estimate; the "
        "free tier was used, so nothing was billed)",
        "",
        "## Dropped findings (post-processing, decision 0007)",
        "",
        "| Reason | Count |",
        "| --- | --- |",
    ]
    for reason in ("schema", "not_in_diff", "evidence", "duplicate"):
        out.append(f"| {reason} | {dropped_all.get(reason, 0)} |")
    return "\n".join(out) + "\n"


def load_runs(out_dir: Path) -> dict[str, ReviewResult]:
    runs: dict[str, ReviewResult] = {}
    for path in sorted((out_dir / "runs").glob("item_*.json")):
        data = json.loads(path.read_bytes())
        runs[path.stem] = ReviewResult.model_validate(data["result"])
    return runs


def score_and_summarize(config: str, out_dir: Path, dataset: Path) -> str:
    report = score_config(dataset, out_dir)
    _write_json(out_dir / "score.json", report)
    today = datetime.now(UTC).date().isoformat()
    text = summary_markdown(report, load_runs(out_dir), config, today)
    (out_dir / "summary.md").write_bytes(text.encode("utf-8"))
    return text


# --- CLI -------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__)
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--items", nargs="+", help="item ids (default: every dataset item)")
    parser.add_argument("--force", action="store_true", help="re-run items already scored")
    parser.add_argument("--reason", default=None, help="why this run happens (recorded in run_log)")
    parser.add_argument("--dataset", type=Path, default=root / "eval" / "dataset")
    parser.add_argument("--results", type=Path, default=root / "eval" / "results")
    args = parser.parse_args(argv)
    load_dotenv(root / ".env")

    out_dir = args.results / args.config
    all_items = dataset_items(args.dataset)
    items = args.items or all_items
    unknown = sorted(set(items) - set(all_items))
    if unknown:
        print(f"error: not dataset items: {', '.join(unknown)}", file=sys.stderr)
        return 2

    router = QuotaRouter(evaluation=True)
    failed: list[str] = []
    for item_id in items:
        scored = out_dir / f"{item_id}.json"
        if scored.is_file() and not args.force:
            print(f"{item_id}: already scored, skipped (use --force to re-run)")
            continue
        reason = args.reason or ("re-run (--force)" if scored.is_file() else "first run")
        try:
            result = run_item(item_id, args.config, router, out_dir, reason)
        except Exception as e:  # materialize or I/O failure: record nothing, report it
            print(f"{item_id}: error: {type(e).__name__}: {e}".splitlines()[0], file=sys.stderr)
            failed.append(item_id)
            continue
        print(
            f"{item_id}: {result.status}, {len(result.findings)} finding(s), "
            f"{result.model_calls} model / {result.tool_calls} tool calls, "
            f"{result.net_duration_s:.1f}s net + {result.wait_s:.1f}s wait"
        )
        if result.status == "quota_paused":
            when = format_reset(result.reset_at) if result.reset_at else "unknown"
            print(f"quota paused: {result.model_id} resets at {when}; re-run after that")
            return 3
        if result.status == "error":
            print(f"{item_id}: {result.error_class}: {result.error}".splitlines()[0])
            failed.append(item_id)

    missing = [i for i in all_items if not (out_dir / f"{i}.json").is_file()]
    if missing:
        print(f"not scored yet: {', '.join(missing)}; summary not written")
        return 1 if failed else 0
    try:
        text = score_and_summarize(args.config, out_dir, args.dataset)
    except DatasetError as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2
    print(text)
    print(f"wrote {out_dir / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
