"""The fixed scoring rule (plan D8 to D9b; details in docs/decisions/0002-scoring-details.md).

Do not change this file after the first evaluation run without a new decision record.

Usage:
    uv run python -m eval.score --config <config_name> [--dataset eval/dataset]
        [--results eval/results] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from eval.ground_truth import (
    DatasetError,
    GroundTruth,
    PlantedBug,
    describe_validation_error,
    load_item,
    parse_diff,
    size_bucket,
)
from reviewer.config import repo_root
from reviewer.schemas import Finding

LINE_TOLERANCE = 3

type Mode = Literal["strict", "location_only"]
MODES: tuple[Mode, ...] = ("strict", "location_only")
BUCKETS: tuple[str, ...] = ("all", "small", "medium", "large")
LABELS_FILENAME = "manual_labels.json"
FINDINGS_GLOB = "item_*.json"

type Label = Literal["real_bug", "false_alarm"]
_LABELS_ADAPTER: TypeAdapter[dict[str, Label]] = TypeAdapter(dict[str, Label])


class FindingsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    config: str
    findings: list[Finding]


def finding_id(item_id: str, index: int) -> str:
    return f"{item_id}#{index}"


# --- Step 1: changed lines (D9a) -------------------------------------------------------------


def changed_lines(diff_text: str) -> dict[str, set[int]]:
    """Changed line numbers on the target (new-file) side, per non-deleted file.

    Added lines count at their target line. A removed line counts at the target line of the
    next line in the same hunk that has one; if none follows, at the hunk's last target line.
    """
    result: dict[str, set[int]] = {}
    for pf in parse_diff(diff_text):
        if pf.is_removed_file:
            continue
        lines = result.setdefault(pf.path, set())
        for hunk in pf:
            hunk_lines = list(hunk)
            last_target = max(hunk.target_start, hunk.target_start + hunk.target_length - 1)
            for i, line in enumerate(hunk_lines):
                if line.is_added:
                    lines.add(line.target_line_no)
                elif line.is_removed:
                    nxt = next(
                        (
                            later.target_line_no
                            for later in hunk_lines[i + 1 :]
                            if later.target_line_no is not None
                        ),
                        last_target,
                    )
                    lines.add(nxt)
    return result


# --- Step 2: discard -------------------------------------------------------------------------


def is_in_diff(finding: Finding, changed: Mapping[str, set[int]]) -> bool:
    lines = changed.get(finding.file)
    if not lines:
        return False
    lo = finding.line_start - LINE_TOLERANCE
    hi = finding.line_end + LINE_TOLERANCE
    return any(lo <= c <= hi for c in lines)


# --- Step 3: match ---------------------------------------------------------------------------


def can_match(bug: PlantedBug, finding: Finding, mode: Mode) -> bool:
    if finding.file != bug.file:
        return False
    if not (
        finding.line_start <= bug.line_end + LINE_TOLERANCE
        and finding.line_end >= bug.line_start - LINE_TOLERANCE
    ):
        return False
    if mode == "strict":
        return finding.category != "other" and finding.category in bug.accepted_kinds
    return True


def _indexed(findings: Sequence[Finding] | Mapping[int, Finding]) -> list[tuple[int, Finding]]:
    if isinstance(findings, Mapping):
        return sorted(findings.items())
    return list(enumerate(findings))


def match(
    bugs: Sequence[PlantedBug],
    findings: Sequence[Finding] | Mapping[int, Finding],
    mode: Mode,
) -> list[tuple[str, int]]:
    """One-to-one maximum bipartite matching (augmenting paths).

    ``findings`` is a sequence (index = position) or a mapping of finding index to finding.
    Bugs are tried in bug_id order and candidate findings in index order, so the result is
    deterministic. Returns (bug_id, finding_index) pairs sorted by bug_id.
    """
    ordered_bugs = sorted(bugs, key=lambda b: b.bug_id)
    indexed = _indexed(findings)
    candidates = {
        b.bug_id: [idx for idx, f in indexed if can_match(b, f, mode)] for b in ordered_bugs
    }
    owner: dict[int, str] = {}  # finding index -> bug_id

    def augment(bug_id: str, seen: set[int]) -> bool:
        for idx in candidates[bug_id]:
            if idx in seen:
                continue
            seen.add(idx)
            if idx not in owner or augment(owner[idx], seen):
                owner[idx] = bug_id
                return True
        return False

    for b in ordered_bugs:
        augment(b.bug_id, set())
    return sorted(((bid, idx) for idx, bid in owner.items()), key=lambda p: p[0])


# --- Step 4: per item ------------------------------------------------------------------------


def score_item(gt: GroundTruth, diff_text: str, findings: Sequence[Finding]) -> dict[str, Any]:
    """Per-item numbers for both modes."""
    changed = changed_lines(diff_text)
    in_diff = {i: f for i, f in enumerate(findings) if is_in_diff(f, changed)}
    discarded = [finding_id(gt.item_id, i) for i in range(len(findings)) if i not in in_diff]
    bugs_by_id = {b.bug_id: b for b in gt.bugs}

    modes: dict[str, Any] = {}
    for mode in MODES:
        pairs = match(gt.bugs, in_diff, mode)
        matched_bugs = {bid for bid, _ in pairs}
        matched_findings = {idx for _, idx in pairs}
        unmatched = [i for i in in_diff if i not in matched_findings]
        duplicates = [
            i
            for i in unmatched
            if any(can_match(bugs_by_id[bid], in_diff[i], mode) for bid in matched_bugs)
        ]
        modes[mode] = {
            "tp": len(pairs),
            "fn": len(gt.bugs) - len(pairs),
            "fp": len(unmatched),
            "duplicates": len(duplicates),
            "matched": [[bid, finding_id(gt.item_id, idx)] for bid, idx in pairs],
            "unmatched_findings": [finding_id(gt.item_id, i) for i in unmatched],
            "duplicate_findings": [finding_id(gt.item_id, i) for i in duplicates],
        }

    return {
        "item_id": gt.item_id,
        "size_bucket": size_bucket(diff_text),
        "is_clean": gt.is_clean,
        "bugs": len(gt.bugs),
        "findings": len(findings),
        "in_diff": len(in_diff),
        "discarded": len(discarded),
        "discarded_findings": discarded,
        "modes": modes,
    }


# --- Step 5: aggregates ----------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def aggregate(
    items: Sequence[Mapping[str, Any]], labels: Mapping[str, str] | None
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        out[mode] = {}
        for bucket in BUCKETS:
            sel = [it for it in items if bucket == "all" or it["size_bucket"] == bucket]
            tp = sum(it["modes"][mode]["tp"] for it in sel)
            fp = sum(it["modes"][mode]["fp"] for it in sel)
            fn = sum(it["modes"][mode]["fn"] for it in sel)
            clean = [it for it in sel if it["is_clean"]]
            row: dict[str, Any] = {
                "items": len(sel),
                "clean_items": len(clean),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "duplicates": sum(it["modes"][mode]["duplicates"] for it in sel),
                "discarded": sum(it["discarded"] for it in sel),
                "recall": _ratio(tp, tp + fn),
                "precision": _ratio(tp, tp + fp),
                "false_alarms_per_clean_pr": (
                    sum(it["modes"][mode]["fp"] for it in clean) / len(clean) if clean else None
                ),
            }
            if mode == "strict":
                unmatched = [fid for it in sel for fid in it["modes"][mode]["unmatched_findings"]]
                if labels is None:
                    real = 0
                    unlabelled = len(unmatched)
                    adjusted = None
                else:
                    real = sum(1 for fid in unmatched if labels.get(fid) == "real_bug")
                    unlabelled = sum(1 for fid in unmatched if fid not in labels)
                    adjusted = _ratio(tp + real, tp + fp)
                row["real_bug_labels"] = real
                row["unlabelled_unmatched"] = unlabelled
                row["adjusted_precision"] = adjusted
            out[mode][bucket] = row
    return out


# --- Loading ---------------------------------------------------------------------------------


def _dataset_items(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.is_dir():
        raise DatasetError(f"dataset folder not found: {dataset_dir}")
    items = sorted(p for p in dataset_dir.iterdir() if (p / "ground_truth.json").is_file())
    if not items:
        raise DatasetError(f"no dataset items (folders with ground_truth.json) in {dataset_dir}")
    return items


def _load_findings(findings_dir: Path) -> dict[str, FindingsFile]:
    if not findings_dir.is_dir():
        raise DatasetError(f"findings folder not found: {findings_dir}")
    out: dict[str, FindingsFile] = {}
    for path in sorted(findings_dir.glob(FINDINGS_GLOB)):
        try:
            ff = FindingsFile.model_validate_json(path.read_bytes())
        except ValidationError as e:
            raise DatasetError(
                f"invalid findings file {path}: {describe_validation_error(e)}"
            ) from e
        if ff.item_id != path.stem:
            raise DatasetError(f"{path}: item_id {ff.item_id!r} does not match the file name")
        if ff.config != findings_dir.name:
            raise DatasetError(
                f"{path}: config {ff.config!r} does not match folder {findings_dir.name!r}"
            )
        out[ff.item_id] = ff
    return out


def _load_labels(findings_dir: Path, known_ids: set[str]) -> dict[str, str] | None:
    path = findings_dir / LABELS_FILENAME
    if not path.is_file():
        return None
    try:
        labels = _LABELS_ADAPTER.validate_json(path.read_bytes())
    except ValidationError as e:
        raise DatasetError(f"invalid {path}: {describe_validation_error(e)}") from e
    unknown = sorted(set(labels) - known_ids)
    if unknown:
        raise DatasetError(f"{path}: labels for unknown finding IDs: {', '.join(unknown)}")
    return labels


def score_config(dataset_dir: Path, findings_dir: Path) -> dict[str, Any]:
    """Score one configuration's findings against the whole dataset."""
    item_dirs = _dataset_items(dataset_dir)
    findings = _load_findings(findings_dir)
    dataset_ids = {d.name for d in item_dirs}

    missing = sorted(dataset_ids - set(findings))
    if missing:
        raise DatasetError(f"no findings file for item(s): {', '.join(missing)}")
    extra = sorted(set(findings) - dataset_ids)
    if extra:
        raise DatasetError(f"findings file(s) for item(s) not in the dataset: {', '.join(extra)}")

    items = []
    known_ids: set[str] = set()
    for d in item_dirs:
        gt, diff_text = load_item(d)
        ff = findings[gt.item_id]
        items.append(score_item(gt, diff_text, ff.findings))
        known_ids.update(finding_id(gt.item_id, i) for i in range(len(ff.findings)))

    labels = _load_labels(findings_dir, known_ids)
    return {
        "config": findings_dir.name,
        "line_tolerance": LINE_TOLERANCE,
        "manual_labels": labels is not None,
        "aggregates": aggregate(items, labels),
        "items": items,
    }


# --- CLI -------------------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def format_table(report: Mapping[str, Any]) -> str:
    agg = report["aggregates"]
    head = ["mode"]
    for b in BUCKETS:
        head += [f"{b} R", f"{b} P"]
    head += ["FA/clean", "adj P"]
    rows = [head]
    for mode in MODES:
        row = [mode]
        for b in BUCKETS:
            row += [_fmt(agg[mode][b]["recall"]), _fmt(agg[mode][b]["precision"])]
        row.append(_fmt(agg[mode]["all"]["false_alarms_per_clean_pr"]))
        row.append(_fmt(agg[mode]["all"].get("adjusted_precision")) if mode == "strict" else "")
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
    a = agg["strict"]["all"]
    lines.append(
        f"items={a['items']} clean={a['clean_items']} discarded={a['discarded']} "
        f"strict tp={a['tp']} fp={a['fp']} fn={a['fn']} duplicates={a['duplicates']}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(prog="python -m eval.score", description=__doc__)
    parser.add_argument("--config", required=True, help="configuration name (results subfolder)")
    parser.add_argument("--dataset", type=Path, default=root / "eval" / "dataset")
    parser.add_argument("--results", type=Path, default=root / "eval" / "results")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    findings_dir = args.results / args.config
    out = args.out or findings_dir / "score.json"
    try:
        report = score_config(args.dataset, findings_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    except (DatasetError, OSError) as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2
    print(format_table(report))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
