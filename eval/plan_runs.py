"""Stage 5 run planner (decision 0015): what is done, what is left, what to run next.

    uv run python -m eval.plan_runs [--budget 430] [--results eval/results] [--dataset eval/dataset]

Prints a status matrix (items x configurations: ``scored``, ``reused``, ``rerun_required`` with its
cause, ``missing``), the model calls estimated to remain per configuration, and the next command in
the D6 order. No model calls; read-only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.results_data import (
    CONFIG_ORDER,
    OPTIONAL,
    calls_on_pacific_day,
    dataset_dir,
    dataset_ids,
    estimate_calls,
    item_status,
    results_root,
)

SHORT = {"scored": "scored", "reused": "reused", "rerun_required": "RERUN", "missing": "-"}


def status_matrix(results: Path, ids: list[str]) -> dict[str, dict[str, tuple[str, str | None]]]:
    return {c: {i: item_status(results, c, i) for i in ids} for c in CONFIG_ORDER}


def remaining_estimates(
    results: Path,
    dataset: Path,
    ids: list[str],
    matrix: dict[str, dict[str, tuple[str, str | None]]],
) -> dict[str, float]:
    """Estimated model calls still needed per configuration (items that are not scored or
    reused)."""
    return {
        c: sum(
            estimate_calls(results, dataset, c, i, ids)
            for i in ids
            if matrix[c][i][0] in ("missing", "rerun_required")
        )
        for c in CONFIG_ORDER
    }


def next_command(
    matrix: dict[str, dict[str, tuple[str, str | None]]], max_calls: int | None
) -> str | None:
    """The command for the first configuration (D6 order) that still has items to do, or None."""
    for c in CONFIG_ORDER:
        todo = [i for i, (s, _) in matrix[c].items() if s in ("missing", "rerun_required")]
        if not todo:
            continue
        cmd = f"uv run python -m eval.run --config {c}"
        if max_calls is not None:
            cmd += f" --max-calls {max_calls}"
        if c in OPTIONAL:
            cmd += "   # optional row 4b: only if quota allows"
        return cmd
    return None


def format_plan(
    results: Path, dataset: Path, budget: int | None = None, day: str | None = None
) -> str:
    ids = dataset_ids(dataset)
    matrix = status_matrix(results, ids)
    est = remaining_estimates(results, dataset, ids, matrix)
    used = calls_on_pacific_day(results, day)
    left = None if budget is None else max(budget - used, 0)

    lines = ["Status (scored / reused / RERUN = rerun_required / - = missing)", ""]
    lines.append("| Item | " + " | ".join(CONFIG_ORDER) + " |")
    lines.append("| --- |" + " --- |" * len(CONFIG_ORDER))
    for i in ids:
        cells = []
        for c in CONFIG_ORDER:
            s, detail = matrix[c][i]
            cells.append(SHORT[s] + (f" ({detail})" if s == "rerun_required" and detail else ""))
        lines.append(f"| {i} | " + " | ".join(cells) + " |")

    lines += ["", "Counts and estimated model calls remaining", ""]
    lines.append("| Configuration | scored | reused | rerun_required | missing | est. calls left |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for c in CONFIG_ORDER:
        n = {k: sum(1 for s, _ in matrix[c].values() if s == k) for k in SHORT}
        lines.append(
            f"| {c}{' (optional)' if c in OPTIONAL else ''} | {n['scored']} | {n['reused']} "
            f"| {n['rerun_required']} | {n['missing']} | {est[c]:.0f} |"
        )
    required = sum(v for c, v in est.items() if c not in OPTIONAL)
    lines += [
        "",
        f"Estimated calls left (without the optional judge row): {required:.0f}",
        f"Model calls recorded today (Pacific day): {used}"
        + (f"; budget {budget}, left today {left}" if budget is not None else ""),
    ]
    cmd = next_command(matrix, left)
    lines += ["", "Next: " + (cmd or "nothing left to run (run eval.score, then the report)")]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.plan_runs", description=__doc__)
    parser.add_argument(
        "--budget", type=int, default=430, help="calls to spend per day (default 430)"
    )
    parser.add_argument("--results", type=Path, default=results_root())
    parser.add_argument("--dataset", type=Path, default=dataset_dir())
    args = parser.parse_args(argv)
    print(format_plan(args.results, args.dataset, args.budget))
    return 0


if __name__ == "__main__":
    sys.exit(main())
