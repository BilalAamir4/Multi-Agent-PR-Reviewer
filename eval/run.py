"""Run one configuration over the dataset and score it (stage 1B; plan § Evaluation, D4, D7).

    uv run python -m eval.run --config single [--items item_001 item_003 ...] [--force]
        [--reason TEXT]

``--config`` is ``single`` (stage 1) or ``orchestrator`` (stage 2B: the orchestrator calling the
specialists; its first run on the dataset is the stage 5 evaluation). The orchestrator's run
file also holds its plan and its children's full results, and its ``wait_s_wall``.

Always uses ``QuotaRouter(evaluation=True)``: one pinned model, no fallback (D4). For each item
it writes

- ``eval/results/<config>/<item_id>.json``: the scorer's input, only when ``decide_scoring``
  says ``scored`` (decision 0010, the same for every configuration): the findings that exist are
  written, an agent that failed contributes nothing;
- ``eval/results/<config>/runs/<item_id>.json``: the full ``ReviewResult``, the scoring decision
  and its cause, and a ``run_log`` of every attempt with its reason. The scorer reads only
  ``item_*.json`` at the top level.

A review that is ``rerun_required`` (quota paused, retries or models exhausted, sandbox
unavailable) is recorded with its cause, not scored, and makes the run exit non-zero so the item
is retried later. Items with a scored result are skipped unless ``--force`` is given. On
``quota_paused`` the run stops with exit code 3. Once every dataset item is scored,
``eval.score`` runs and
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
from reviewer.agents.orchestrator import OrchestratorResult, orchestrate
from reviewer.agents.single import ReviewResult, review
from reviewer.cli import format_reset, item_inputs
from reviewer.config import load_models_config, load_review_config, repo_root
from reviewer.models import QuotaRouter

CONFIGS = ("single", "orchestrator")

# Scoring policy for every configuration (decision 0010). Infrastructure causes say nothing about
# the architecture, so the item is re-run; every other end state is scored with the findings that
# exist (an agent that failed contributes nothing).
# RemoteUnavailable and RemoteError: the A2A service could not be reached or its call failed
# (decision 0011); that says nothing about the architecture either.
INFRA_ERRORS = frozenset({"RetriesExhausted", "AllExhausted", "RemoteUnavailable", "RemoteError"})
SANDBOX_NOTE = "sandbox unavailable"


def _children(r: ReviewResult) -> list[ReviewResult]:
    return list(r.children) if isinstance(r, OrchestratorResult) else []


def _rerun_cause(r: ReviewResult) -> str | None:
    """The infrastructure cause in ``r`` or any child, or None."""
    if r.status == "quota_paused":
        return "quota_paused"
    if r.error_class in INFRA_ERRORS:
        return str(r.error_class)
    if r.status == "partial" and any(SANDBOX_NOTE in n for n in r.notes):
        return "sandbox unavailable"
    for c in _children(r):
        cause = _rerun_cause(c)
        if cause:
            return f"{cause} ({c.agent})"
    return None


def decide_scoring(result: ReviewResult) -> tuple[str, str]:
    """``("scored" | "rerun_required", cause)`` for one review, the same for every configuration.

    Scored: ``complete``; ``limit_reached``; ``error`` for a non-infrastructure reason (for
    example ``NoStructuredResponse`` or a fatal model error); ``partial`` where every
    non-complete child ended ``limit_reached``, a non-infrastructure ``error`` or such a
    ``partial``. Rerun required: any ``quota_paused``; ``RetriesExhausted`` or ``AllExhausted``
    anywhere in the review; a ``partial`` caused by ``sandbox unavailable`` (D2b); the A2A
    security service not running or its call failing (``RemoteUnavailable``, ``RemoteError``,
    decision 0011); a ``partial`` with no recorded cause."""
    cause = _rerun_cause(result)
    if cause is not None:
        return "rerun_required", cause
    if result.status in ("complete", "limit_reached"):
        return "scored", result.status
    if result.status == "error":
        return "scored", f"error: {result.error_class or 'unknown'}"
    if result.status == "partial":
        bad = [c for c in _children(result) if c.status != "complete"]
        if not bad:
            return "rerun_required", "partial: no recorded cause"
        parts = []
        for c in bad:
            if c.status == "partial":  # a partial child with no sandbox note: cause unknown
                return "rerun_required", f"partial: {c.agent} partial with no recorded cause"
            parts.append(f"{c.agent} {c.status}")
        return "scored", "partial: " + ", ".join(parts)
    return "rerun_required", f"unexpected status {result.status}"


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
    """Review one item and write its run file (always) and its scored file (when
    ``decide_scoring`` says ``scored``); the decision and its cause go in the run file."""
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    ctx, title, body = item_inputs(item_id)
    review_config = load_review_config()
    security_mode = review_config.a2a.security_reviewer.mode if config == "orchestrator" else None
    if config == "orchestrator":
        result = orchestrate(ctx, title, body, router, item_id=item_id)
    else:
        result = review(ctx, title, body, router, item_id=item_id)
    run_path = out_dir / "runs" / f"{item_id}.json"
    run_log = _load_run_log(run_path)
    extra: dict[str, Any] = {}
    if isinstance(result, OrchestratorResult):
        security = next((c for c in result.children if c.agent == "security_reviewer"), None)
        extra = {
            "plan": [e.model_dump() for e in result.plan],
            "children": {c.agent: c.status for c in result.children},
            "wait_s_sum": result.wait_s_sum,
            "wait_s_wall": result.wait_s_wall,
            "own_model_calls": result.own_model_calls,
            "own_input_tokens": result.own_input_tokens,
            "own_output_tokens": result.own_output_tokens,
            # Stage 3 (decision 0011): how the security specialist was reached. The configured
            # mode comes from config/review.yaml only; the transport is what the child recorded.
            "security_mode": security_mode,
            "security_transport": security.transport if security else None,
            "security_transport_detail": (
                security.transport_detail.model_dump()
                if security and security.transport_detail
                else None
            ),
        }
    decision, cause = decide_scoring(result)
    run_log.append(
        {
            **extra,
            "started_at": started_at,
            "reason": reason,
            "status": result.status,
            "scoring": {"decision": decision, "cause": cause},
            "retries_by_reason": result.retries_by_reason,
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
    _write_json(
        run_path,
        {
            "result": result.model_dump(mode="json"),
            "security_mode": security_mode,
            "scoring": {"decision": decision, "cause": cause},
            "run_log": run_log,
        },
    )
    if decision == "scored":
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


def load_scoring(out_dir: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    """The latest scoring decision of each item, and how many recorded runs there were in total
    and how many of them were ``rerun_required``."""
    latest: dict[str, dict[str, str]] = {}
    total = reruns = 0
    for path in sorted((out_dir / "runs").glob("item_*.json")):
        data = json.loads(path.read_bytes())
        if "scoring" in data:
            latest[path.stem] = data["scoring"]
        for entry in data.get("run_log", []):
            total += 1
            reruns += (entry.get("scoring") or {}).get("decision") == "rerun_required"
    return latest, total, reruns


def summary_markdown(
    report: dict[str, Any],
    runs: dict[str, ReviewResult],
    config: str,
    today: str,
    scoring: tuple[dict[str, dict[str, str]], int, int] | None = None,
) -> str:
    models = load_models_config()
    spec = models.models[models.evaluation.model]
    limits = load_review_config().limits.for_agent(config)
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
    wall = [r.wait_s_wall for r in results if isinstance(r, OrchestratorResult)]
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
        f"median wait beside it: {_fmt(_median(wait), 1)} s"
        + (
            "; for the orchestrator the removed wait is the wall-clock union of all waits "
            f"(wait_s_wall, decision 0009), median {_fmt(_median(wall), 1)} s"
            if wall
            else ""
        ),
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
    latest, total_runs, rerun_runs = scoring or ({}, 0, 0)
    if not latest:  # no decisions stored: work them out from the results
        latest = {i: dict(zip(("decision", "cause"), decide_scoring(r), strict=True))
                  for i, r in runs.items()}  # fmt: skip
    by_decision = Counter(v["decision"] for v in latest.values())
    by_cause = Counter((v["decision"], v["cause"]) for v in latest.values())
    out += [
        "",
        "## Scoring decisions (decision 0010)",
        "",
        f"- Items scored: {by_decision.get('scored', 0)}; "
        f"items needing a re-run: {by_decision.get('rerun_required', 0)}",
        f"- Recorded runs: {total_runs}, of which `rerun_required` (not scored, re-run): "
        f"{rerun_runs}",
        "",
        "| Decision | Cause | Items |",
        "| --- | --- | --- |",
    ]
    for (decision, cause), n in sorted(by_cause.items()):
        out.append(f"| {decision} | {cause} | {n} |")
    return "\n".join(out) + "\n"


def load_runs(out_dir: Path, config: str = "single") -> dict[str, ReviewResult]:
    model = OrchestratorResult if config == "orchestrator" else ReviewResult
    runs: dict[str, ReviewResult] = {}
    for path in sorted((out_dir / "runs").glob("item_*.json")):
        data = json.loads(path.read_bytes())
        runs[path.stem] = model.model_validate(data["result"])
    return runs


def score_and_summarize(config: str, out_dir: Path, dataset: Path) -> str:
    report = score_config(dataset, out_dir)
    _write_json(out_dir / "score.json", report)
    today = datetime.now(UTC).date().isoformat()
    text = summary_markdown(
        report, load_runs(out_dir, config), config, today, load_scoring(out_dir)
    )
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
        decision, cause = decide_scoring(result)
        print(f"{item_id}: scoring decision: {decision} ({cause})")
        if result.status == "error":
            print(f"{item_id}: {result.error_class}: {result.error}".splitlines()[0])
        if decision == "rerun_required":
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
