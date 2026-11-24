"""The stage 5 report (stage 5Z, decision 0015): ``eval/results/REPORT.md`` and the README table.

    uv run python -m eval.report [--no-readme]

Reads ``eval/results/<config>/`` (scored files, run files, ``manual_labels.json``) and the dataset
and scores with the frozen ``eval.score``; every number traces back to those files. Only
configurations whose results are complete (all 20 items scored or reused) get a row. No model calls.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.dataset_stats import load_notes
from eval.ground_truth import load_item
from eval.results_data import (
    CONFIG_ORDER,
    agent_breakdown,
    dataset_dir,
    dataset_ids,
    estimated_cost,
    item_bucket,
    item_cost,
    item_status,
    load_run,
    load_scored,
    results_root,
)
from eval.score import BUCKETS, score_config, score_item
from reviewer.config import repo_root
from reviewer.schemas import Finding

TITLES = {
    "single": "One agent",
    "orchestrator": "Orchestrator calling specialists",
    "subagents": "Plus spawned sub-agents on large diffs",
    "verifier": "Plus verifier",
    "judge": "Plus orchestrator as judge (optional)",
}
KINDS = (
    "off_by_one", "missing_none_check", "wrong_comparison", "unhandled_exception",
    "sql_injection", "hardcoded_secret", "ineffective_test",
)  # fmt: skip
IN_PROGRESS = (
    "in progress: paused by a Gemini API outage on Nov 24 (503 high demand); "
    "same model, will resume"
)
STAGE1_SINGLE = ("item_001", "item_002", "item_003", "item_004", "item_005")  # decision 0015
SUB_AGENT = re.compile(r"/sub\d+$")  # Finding.agent of a sub-agent, e.g. bug_reviewer/sub2


@dataclass
class ConfigData:
    name: str
    score: dict[str, Any]  # eval.score.score_config output
    runs: dict[str, dict[str, Any]]
    findings: dict[str, list[Finding]]
    costs: dict[str, dict[str, float]]


def load_config(results: Path, dataset: Path, config: str) -> ConfigData | None:
    """The configuration's data, or None when any item is not scored or reused."""
    ids = dataset_ids(dataset)
    if any(item_status(results, config, i)[0] not in ("scored", "reused") for i in ids):
        return None
    runs = {i: r for i in ids if (r := load_run(results, config, i)) is not None}
    findings = {
        i: [Finding.model_validate(f) for f in (load_scored(results, config, i) or {})["findings"]]
        for i in ids
    }
    return ConfigData(
        config,
        score_config(dataset, results / config),
        runs,
        findings,
        {i: item_cost(r, config) for i, r in runs.items()},
    )


# --- Formatting helpers ----------------------------------------------------------------------


def fmt(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def usd(x: float | None) -> str:
    return "-" if x is None else f"${x:.4f}"


def kilo(x: float | None) -> str:
    return "-" if x is None else f"{x / 1000:.0f}K"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


# --- Metrics ---------------------------------------------------------------------------------


def cost_summary(cd: ConfigData, dataset: Path, bucket: str) -> dict[str, float | None]:
    sel = [c for i, c in cd.costs.items() if bucket == "all" or item_bucket(dataset, i) == bucket]
    tin, tout = mean([c["input_tokens"] for c in sel]), mean([c["output_tokens"] for c in sel])
    return {
        "in": tin,
        "out": tout,
        "usd": estimated_cost(tin, tout) if tin is not None and tout is not None else None,
        "net": median([c["net_s"] for c in sel]),
        "wait": median([c["wait_s"] for c in sel]),
    }


def metric_row(cd: ConfigData, dataset: Path, bucket: str) -> list[str]:
    s = cd.score["aggregates"]["strict"][bucket]
    loc = cd.score["aggregates"]["location_only"][bucket]
    c = cost_summary(cd, dataset, bucket)
    adj = s.get("adjusted_precision")
    return [
        TITLES[cd.name],
        str(s["items"]),
        fmt(s["recall"]),
        f"{fmt(s['precision'])} / {fmt(adj)}",
        fmt(s["false_alarms_per_clean_pr"]),
        f"{fmt(loc['recall'])} / {fmt(loc['precision'])}",
        f"{kilo(c['in'])} in, {fmt(c['out'], 0)} out",
        usd(c["usd"]),
        f"{fmt(c['net'], 1)} s",
        f"{fmt(c['wait'], 1)} s",
    ]


MAIN_HEADER = [
    "Configuration", "Items", "Recall (strict)", "Precision (strict / adjusted)",
    "False alarms per clean PR", "Location-only recall / precision", "Mean tokens per review",
    "Est. cost per review", "Median net time", "Median wait",
]  # fmt: skip


def per_kind(cd: ConfigData, truth: dict[str, dict[str, str]]) -> dict[str, tuple[int, int]]:
    """``kind -> (strict hits, planted)`` over the dataset. ``truth[item][bug_id] = kind``."""
    hits = dict.fromkeys(KINDS, 0)
    total = dict.fromkeys(KINDS, 0)
    for item in cd.score["items"]:
        kinds = truth[item["item_id"]]
        for kind in kinds.values():
            total[kind] += 1
        for bug_id, _fid in item["modes"]["strict"]["matched"]:
            hits[kinds[bug_id]] += 1
    return {k: (hits[k], total[k]) for k in KINDS}


def hit_map(cd: ConfigData) -> dict[tuple[str, str], Finding]:
    """``(item, bug) -> the finding that matched it`` in strict mode."""
    out: dict[tuple[str, str], Finding] = {}
    for item in cd.score["items"]:
        for bug_id, fid in item["modes"]["strict"]["matched"]:
            out[(item["item_id"], bug_id)] = cd.findings[item["item_id"]][int(fid.split("#")[1])]
    return out


def subagent_only_hits(
    sub: ConfigData, base: ConfigData | None
) -> list[tuple[str, str, str, bool]]:
    """Strict hits of ``sub`` whose finding came from a sub-agent: ``(item, bug, agent,
    missed_by_base)``. ``missed_by_base`` is True when ``base`` (the orchestrator row) did not hit
    that bug."""
    base_hits = set(hit_map(base)) if base is not None else set()
    return [
        (item, bug, f.agent, (item, bug) not in base_hits)
        for (item, bug), f in sorted(hit_map(sub).items())
        if SUB_AGENT.search(f.agent)
    ]


def agent_table(cd: ConfigData) -> list[list[str]]:
    totals: dict[str, list[int]] = {}
    for run in cd.runs.values():
        for name, calls, tin, tout in agent_breakdown(run, cd.name):
            acc = totals.setdefault(name, [0, 0, 0])
            acc[0] += calls
            acc[1] += tin
            acc[2] += tout
    n = max(len(cd.runs), 1)
    return [[name, str(c), f"{c / n:.1f}", str(i), str(o)] for name, (c, i, o) in totals.items()]


def judge_drops(
    results: Path, dataset: Path, judge: ConfigData, saved: ConfigData
) -> dict[str, int]:
    """What a judge dropped from the saved ``subagents`` findings: how many, how many of those
    were strict hits (recall lost), strict false alarms (precision gained) and the rest (outside
    the changed lines or duplicates of a hit)."""
    out = {"input": 0, "kept": 0, "dropped": 0, "hits": 0, "false_alarms": 0, "other": 0,
           "bugs_lost": 0}  # fmt: skip
    for item_id, before in saved.findings.items():
        gt, diff_text = load_item(dataset / item_id)
        scored = score_item(gt, diff_text, before)["modes"]["strict"]
        hit_idx = {int(fid.split("#")[1]) for _bug, fid in scored["matched"]}
        fp_idx = {int(fid.split("#")[1]) for fid in scored["unmatched_findings"]}
        remaining = [f.model_dump() for f in judge.findings[item_id]]
        out["input"] += len(before)
        out["kept"] += len(judge.findings[item_id])
        for idx, f in enumerate(before):
            dump = f.model_dump()
            if dump in remaining:
                remaining.remove(dump)
                continue
            out["dropped"] += 1
            if idx in hit_idx:
                out["hits"] += 1
            elif idx in fp_idx:
                out["false_alarms"] += 1
            else:
                out["other"] += 1
    s_before = saved.score["aggregates"]["strict"]["all"]["tp"]
    s_after = judge.score["aggregates"]["strict"]["all"]["tp"]
    out["bugs_lost"] = s_before - s_after
    return out


def rerun_lines(results: Path, ids: list[str], config: str) -> list[str]:
    """Items with more than one recorded run or a ``rerun_required`` run, with the reasons."""
    lines = []
    for i in ids:
        run = load_run(results, config, i)
        if run is None:
            continue
        log = run.get("run_log", [])
        bad = [e for e in log if (e.get("scoring") or {}).get("decision") == "rerun_required"]
        if len(log) > 1 or bad:
            why = "; ".join(
                f"{e.get('reason', '?')}: {(e.get('scoring') or {}).get('decision', 'scored')}"
                f" ({(e.get('scoring') or {}).get('cause', '-')})"
                for e in log
            )
            lines.append(f"- `{config}` {i}: {len(log)} recorded run(s): {why}")
    return lines


# --- The report ------------------------------------------------------------------------------


def build_report(
    results: Path, dataset: Path, notes: dict[str, str] | None = None
) -> tuple[str, list[ConfigData]]:
    ids = dataset_ids(dataset)
    loaded = {c: load_config(results, dataset, c) for c in CONFIG_ORDER}
    configs = [cd for cd in loaded.values() if cd is not None]
    missing = [c for c, cd in loaded.items() if cd is None]
    truth = {
        i: {
            b["bug_id"]: b["kind"]
            for b in json.loads((dataset / i / "ground_truth.json").read_bytes())["bugs"]
        }
        for i in ids
    }
    notes = load_notes() if notes is None else notes
    out = ["# Evaluation report", ""]
    out += [
        "Generated by `eval/report.py` from `eval/results/` (decision 0015). Every number comes "
        "from the scored files, run files and manual labels there. Model: `gemini-3.1-flash-lite` "
        "for every role (D4), free tier; costs are the D5 price estimate, nothing was billed.",
        "",
        f"Configurations with complete results: {', '.join(c.name for c in configs) or 'none'}. "
        + (f"Not complete (no row): {', '.join(missing)}." if missing else ""),
        "",
        *([f"Rows `{'`, `'.join(missing)}`: {IN_PROGRESS}.", ""] if missing else []),
        "## Main table (all 20 items)",
        "",
        *table(MAIN_HEADER, [metric_row(cd, dataset, "all") for cd in configs]),
        "",
        "Net time removes rate-limit waiting (D7); for orchestrated rows the removed wait is the "
        "wall-clock union `wait_s_wall`. `verifier` and `judge` cost and time are cumulative "
        "(the `subagents` review plus the step). Reused `subagents` items carry the cost of the "
        "orchestrator review they were copied from. Adjusted precision is `-` until "
        "`manual_labels.json` exists for the configuration (blind labelling, D9b).",
        "",
    ]
    unl = {
        cd.name: cd.score["aggregates"]["strict"]["all"].get("unlabelled_unmatched", 0)
        for cd in configs
    }
    out += [
        "Unmatched findings without a label: "
        + ", ".join(f"{k} {v}" for k, v in unl.items())
        + ".",
        "",
    ]
    for bucket in BUCKETS[1:]:
        out += [f"## By size: {bucket}", ""]
        out += table(MAIN_HEADER, [metric_row(cd, dataset, bucket) for cd in configs])
        out.append("")

    out += ["## Strict and location-only counts", ""]
    rows = []
    for cd in configs:
        for mode in ("strict", "location_only"):
            a = cd.score["aggregates"][mode]["all"]
            rows.append([TITLES[cd.name], mode, fmt(a["recall"]), fmt(a["precision"]),
                         str(a["tp"]), str(a["fp"]), str(a["fn"]),
                         fmt(a["false_alarms_per_clean_pr"])])  # fmt: skip
    out += table(
        ["Configuration", "Mode", "Recall", "Precision", "tp", "fp", "fn", "FA / clean PR"], rows
    )

    out += ["", "## Recall by planted kind (strict hits / planted)", ""]
    kinds = {cd.name: per_kind(cd, truth) for cd in configs}
    rows = []
    for k in KINDS:
        row = [k]
        for cd in configs:
            h, t = kinds[cd.name][k]
            row.append(f"{h}/{t}" + (f" ({h / t:.2f})" if t else ""))
        rows.append(row)
    out += table(["Kind", *[TITLES[cd.name] for cd in configs]], rows)

    cross = {k: v for k, v in notes.items()}
    out += ["", f"## Recall on the {len(cross)} cross-file bugs (strict)", ""]
    rows = []
    for key, other in sorted(cross.items()):
        item, bug = key.split("/")
        row = [key, other]
        for cd in configs:
            row.append("hit" if (item, bug) in hit_map(cd) else "miss")
        rows.append(row)
    out += table(["Bug", "Other file", *[TITLES[cd.name] for cd in configs]], rows)
    summary = ", ".join(
        f"{cd.name} {sum(1 for k in cross if tuple(k.split('/')) in hit_map(cd))}/{len(cross)}"
        for cd in configs
    )
    out += ["", f"Cross-file recall: {summary or '-'}.", ""]

    out += ["## Hits found by sub-agents", ""]
    sub, base = loaded.get("subagents"), loaded.get("orchestrator")
    if sub is None:
        out += ["`subagents` is not complete."]
    else:
        hits = subagent_only_hits(sub, base)
        out += [
            f"Strict hits of `subagents` whose finding's `agent` is a sub-agent (`*/sub<k>`): "
            f"{len(hits)}; of these the `orchestrator` row missed: {sum(1 for h in hits if h[3])}.",
            "",
        ]
        if hits:
            out += table(
                ["Item / bug", "Agent", "Missed by `orchestrator`"],
                [[f"{i}/{b}", a, "yes" if m else "no"] for i, b, a, m in hits],
            )

    out += ["", "## Calls and tokens by agent (totals over the 20 reviews)", ""]
    for cd in configs:
        out += [
            f"### {TITLES[cd.name]}"
            + (
                " (the judge step alone; cumulative cost is in the main table)"
                if cd.name in ("verifier", "judge")
                else ""
            ),
            "",
            *table(
                ["Agent", "Model calls", "Per review", "Input tokens", "Output tokens"],
                agent_table(cd),
            ),
            "",
        ]

    out += ["## Judges: what they dropped (decision 0013)", ""]
    saved = loaded.get("subagents")
    judged = [cd for cd in configs if cd.name in ("verifier", "judge")]
    if saved is None or not judged:
        out += ["No judge row is complete."]
    else:
        rows = []
        for cd in judged:
            d = judge_drops(results, dataset, cd, saved)
            rows.append([TITLES[cd.name], str(d["input"]), str(d["kept"]), str(d["dropped"]),
                         str(d["hits"]), str(d["false_alarms"]), str(d["other"]),
                         str(d["bugs_lost"])])  # fmt: skip
        out += table(
            ["Judge", "Findings in", "Kept", "Dropped", "Dropped hits (recall lost)",
             "Dropped false alarms (precision gained)", "Dropped other", "Bugs lost (tp)"],
            rows,
        )  # fmt: skip

    out += ["", "## Reused and re-run items", ""]
    if sub is not None:
        reused = sorted(i for i in ids if item_status(results, "subagents", i)[0] == "reused")
        out.append(
            f"- `subagents` reused from `orchestrator` ({len(reused)}): {', '.join(reused) or '-'}"
        )
    out.append(
        f"- `single` from stage 1 (task-level retries, no retries occurred): "
        f"{', '.join(STAGE1_SINGLE)}; the rest ran under per-call retries."
    )
    reruns = [line for c in CONFIG_ORDER for line in rerun_lines(results, ids, c)]
    out += reruns or ["- No item needed a re-run or has more than one recorded run."]
    out += [
        "",
        "## Sources",
        "",
        "`eval/results/<config>/item_*.json` (findings), `runs/item_*.json` (cost, time, calls, "
        "run log), `manual_labels.json` (D9b labels), `eval/dataset/` (ground truth), "
        "`eval/dataset_notes.yaml` (cross-file bugs), `config/models.yaml` (prices).",
    ]
    return "\n".join(out) + "\n", configs


# --- README ----------------------------------------------------------------------------------


def readme_rows(configs: list[ConfigData], dataset: Path) -> list[str]:
    by_name = {cd.name: cd for cd in configs}
    rows = []
    for name in CONFIG_ORDER:
        cd = by_name.get(name)
        if cd is None:
            rows.append(f"| {TITLES[name]} | {IN_PROGRESS} | | | | |")
            continue
        s = cd.score["aggregates"]["strict"]["all"]
        loc = cd.score["aggregates"]["location_only"]["all"]
        c = cost_summary(cd, dataset, "all")
        rows.append(
            f"| {TITLES[name]} | {fmt(s['recall'])} (location-only {fmt(loc['recall'])}) "
            f"| {fmt(s['precision'])} / {fmt(s.get('adjusted_precision'))} "
            f"| {fmt(s['false_alarms_per_clean_pr'])} "
            f"| {usd(c['usd'])} ({kilo(c['in'])} input, {fmt(c['out'], 0)} output tokens) "
            f"| {fmt(c['net'], 1)} s (+{fmt(c['wait'], 1)} s waiting) |"
        )
    return rows


def update_readme(text: str, rows: list[str]) -> str:
    """Replace the results table of the README (header line to the last table row)."""
    header = "| Configuration | Recall | Precision (strict / adjusted) |"
    start = text.index(header)
    lines = text[start:].split("\n")
    end = 0
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    new = [lines[0], lines[1], *rows]
    return text[:start] + "\n".join(new + lines[end:])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__)
    parser.add_argument("--results", type=Path, default=results_root())
    parser.add_argument("--dataset", type=Path, default=dataset_dir())
    parser.add_argument("--readme", type=Path, default=repo_root() / "README.md")
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args(argv)
    text, configs = build_report(args.results, args.dataset)
    (args.results / "REPORT.md").write_bytes(text.encode("utf-8"))
    print(f"wrote {args.results / 'REPORT.md'} ({len(configs)} configuration(s))")
    if not args.no_readme:
        readme = args.readme.read_bytes().decode("utf-8")
        args.readme.write_bytes(
            update_readme(readme, readme_rows(configs, args.dataset)).encode("utf-8")
        )
        print(f"updated the results table in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
