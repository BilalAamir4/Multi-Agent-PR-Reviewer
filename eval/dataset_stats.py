"""Dataset statistics and target check (decision 0014).

Usage:
    uv run python -m eval.dataset_stats [--no-spawn]

Prints, for the items built so far: the bucket by type counts, the bug kind counts (overall and
per session), the items per repository, the multi-line and cross-file counts and which items
would trigger D12 spawning (``should_spawn_for``; code only, no model calls). Then it compares
every built item and every session total with the targets of the dataset expansion (D5 and the
E1/E2 briefs) and exits 1 on a mismatch.

Cross-file bugs are listed in ``eval/dataset_notes.yaml`` (the recipes do not store that fact);
multi-line means ``line_end > line_start`` in ``ground_truth.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from reviewer.config import repo_root

SESSIONS: dict[str, tuple[str, ...]] = {
    "0D": tuple(f"item_{n:03d}" for n in range(1, 6)),
    "E1": tuple(f"item_{n:03d}" for n in range(6, 14)),
    "E2": tuple(f"item_{n:03d}" for n in range(14, 21)),
}

# item -> (bucket, type, number of bugs). item_001 to item_005 are from session 0D.
TARGET_ITEMS: dict[str, tuple[str, str, int]] = {
    "item_001": ("small", "planted", 1),
    "item_002": ("small", "clean", 0),
    "item_003": ("medium", "planted", 2),
    "item_004": ("large", "planted", 3),
    "item_005": ("large", "clean", 0),
    "item_006": ("large", "planted", 2),
    "item_007": ("large", "planted", 3),
    "item_008": ("large", "planted", 2),
    "item_009": ("large", "clean", 0),
    "item_010": ("medium", "planted", 2),
    "item_011": ("small", "planted", 1),
    "item_012": ("small", "clean", 0),
    "item_013": ("large", "planted", 2),
    "item_014": ("large", "planted", 3),
    "item_015": ("large", "planted", 2),
    "item_016": ("large", "clean", 0),
    "item_017": ("medium", "planted", 2),
    "item_018": ("medium", "clean", 0),
    "item_019": ("small", "planted", 2),
    "item_020": ("small", "planted", 1),
}

# Bug kinds the new items must add per session (E1: 12 bugs, E2: 10 bugs).
TARGET_KINDS: dict[str, dict[str, int]] = {
    "E1": {
        "off_by_one": 2,
        "hardcoded_secret": 2,
        "sql_injection": 2,
        "missing_none_check": 2,
        "wrong_comparison": 2,
        "unhandled_exception": 1,
        "ineffective_test": 1,
    },
    "E2": {
        "off_by_one": 2,
        "hardcoded_secret": 1,
        "sql_injection": 1,
        "missing_none_check": 1,
        "wrong_comparison": 1,
        "unhandled_exception": 2,
        "ineffective_test": 2,
    },
}
MAX_ITEMS_PER_REPO = 6
MIN_MULTI_LINE_NEW = 3  # across the 22 new bugs
MIN_CROSS_FILE_PER_SESSION = 1
SPECIALISTS = ("bug_reviewer", "security_reviewer", "test_reviewer")


@dataclass(frozen=True)
class BugInfo:
    bug_id: str
    kind: str
    file: str
    line_start: int
    line_end: int
    cross_file: str | None = None  # the other changed file the defect shows through

    @property
    def multi_line(self) -> bool:
        return self.line_end > self.line_start


@dataclass(frozen=True)
class ItemInfo:
    item_id: str
    repo: str
    bucket: str
    py_files: int
    lines: int
    bugs: tuple[BugInfo, ...] = field(default_factory=tuple)

    @property
    def type(self) -> str:
        return "planted" if self.bugs else "clean"


# --- Pure counting functions (unit-tested) ------------------------------------------------------


def session_of(item_id: str) -> str | None:
    for name, ids in SESSIONS.items():
        if item_id in ids:
            return name
    return None


def bucket_by_type(items: Iterable[ItemInfo]) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter((i.bucket, i.type) for i in items)
    return dict(sorted(counts.items()))


def kind_counts(items: Iterable[ItemInfo], session: str | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        if session is not None and session_of(item.item_id) != session:
            continue
        counts.update(b.kind for b in item.bugs)
    return dict(sorted(counts.items()))


def items_per_repo(items: Iterable[ItemInfo]) -> dict[str, int]:
    return dict(sorted(Counter(i.repo for i in items).items()))


def multi_line_bugs(items: Iterable[ItemInfo]) -> list[tuple[str, str]]:
    return [(i.item_id, b.bug_id) for i in items for b in i.bugs if b.multi_line]


def cross_file_bugs(items: Iterable[ItemInfo]) -> list[tuple[str, str]]:
    return [(i.item_id, b.bug_id) for i in items for b in i.bugs if b.cross_file]


def load_notes(path: Path | None = None) -> dict[str, str]:
    """``eval/dataset_notes.yaml``: ``cross_file: {"item_007/b2": "<other changed file>"}``."""
    path = path or repo_root() / "eval" / "dataset_notes.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_bytes().decode("utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("cross_file") or {}).items()}


def load_items(dataset: Path | None = None, notes: dict[str, str] | None = None) -> list[ItemInfo]:
    dataset = dataset or repo_root() / "eval" / "dataset"
    notes = load_notes() if notes is None else notes
    items = []
    for item_dir in sorted(p for p in dataset.iterdir() if (p / "ground_truth.json").is_file()):
        gt = json.loads((item_dir / "ground_truth.json").read_bytes())
        meta = json.loads((item_dir / "meta.json").read_bytes())
        bugs = tuple(
            BugInfo(
                b["bug_id"],
                b["kind"],
                b["file"],
                b["line_start"],
                b["line_end"],
                notes.get(f"{gt['item_id']}/{b['bug_id']}"),
            )
            for b in gt["bugs"]
        )
        items.append(
            ItemInfo(
                gt["item_id"], meta["repo"], meta["bucket"], meta["py_files_changed"],
                meta["lines_changed"], bugs,
            )
        )  # fmt: skip
    return items


def check_targets(items: Sequence[ItemInfo]) -> list[str]:
    """Problems found when comparing the built items with the targets (empty when all match)."""
    problems = []
    built = {i.item_id for i in items}
    for item in items:
        want = TARGET_ITEMS.get(item.item_id)
        if want is None:
            problems.append(f"{item.item_id}: not in the target table")
            continue
        got = (item.bucket, item.type, len(item.bugs))
        if got != want:
            problems.append(f"{item.item_id}: {got} != target {want}")
        if item.bucket == "medium" and not 4 <= item.py_files <= 7:
            problems.append(f"{item.item_id}: medium item with {item.py_files} .py files")
        files = {b.file for b in item.bugs}
        if len(item.bugs) >= 2 and len(files) < 2:
            problems.append(f"{item.item_id}: {len(item.bugs)} bugs but only {len(files)} file")
    for session, target in TARGET_KINDS.items():
        ids = SESSIONS[session]
        if not all(i in built for i in ids):
            continue  # a session is compared only when all its items are built
        got_kinds = kind_counts(items, session)
        if got_kinds != dict(sorted(target.items())):
            problems.append(f"{session} kinds {got_kinds} != target {dict(sorted(target.items()))}")
    for repo, n in items_per_repo(items).items():
        if n > MAX_ITEMS_PER_REPO:
            problems.append(f"{repo} has {n} items (max {MAX_ITEMS_PER_REPO})")
    for session in ("E1", "E2"):
        ids = SESSIONS[session]
        done = [i for i in items if i.item_id in ids]
        if not done:
            continue
        if len({i.repo for i in done}) < 3:
            problems.append(f"{session} uses fewer than 3 repositories")
        if len(done) == len(ids) and len(cross_file_bugs(done)) < MIN_CROSS_FILE_PER_SESSION:
            problems.append(f"{session} has no cross-file bug")
    new = [i for i in items if session_of(i.item_id) in ("E1", "E2")]
    if all(i in built for s in ("E1", "E2") for i in SESSIONS[s]) and (
        len(multi_line_bugs(new)) < MIN_MULTI_LINE_NEW
    ):
        problems.append(f"only {len(multi_line_bugs(new))} multi-line new bugs")
    return problems


# --- Spawn rows (code only) ------------------------------------------------------------------


def spawn_rows(items: Sequence[ItemInfo]) -> list[tuple[str, list[str], int, int]]:
    """(item_id, specialists that would spawn, .py files, estimated tokens of the bug reviewer's
    area) per item, via ``should_spawn_for`` on the item's materialized workspace."""
    from eval.repos import materialize
    from reviewer.agents.subagents import should_spawn_for
    from reviewer.config import load_review_config
    from reviewer.tools.context import ReviewContext

    cfg = load_review_config().spawn
    rows = []
    for item in items:
        workspace = materialize(item.item_id)
        ctx = ReviewContext.from_paths(
            workspace, repo_root() / "eval" / "dataset" / item.item_id / "pr.diff"
        )
        decisions = {n: should_spawn_for(n, ctx, cfg) for n in SPECIALISTS}
        spawn = [n for n, d in decisions.items() if d.spawn]
        first = decisions[SPECIALISTS[0]]
        rows.append((item.item_id, spawn, first.py_files, first.estimated_tokens))
    return rows


# --- Report ----------------------------------------------------------------------------------


def format_report(items: Sequence[ItemInfo], spawn: list | None) -> str:
    out: list[str] = []
    planted = sum(1 for i in items if i.type == "planted")
    out.append(f"items: {len(items)} ({planted} planted, {len(items) - planted} clean)")
    out.append("")
    out.append("bucket by type:")
    for (bucket, typ), n in bucket_by_type(items).items():
        out.append(f"  {bucket:<7} {typ:<8} {n}")
    out.append("")
    out.append("bug kinds (overall / 0D / E1 / E2):")
    overall = kind_counts(items)
    per = {s: kind_counts(items, s) for s in SESSIONS}
    for kind in sorted({*overall}):
        cells = " / ".join(str(per[s].get(kind, 0)) for s in SESSIONS)
        out.append(f"  {kind:<20} {overall[kind]:>2}   ({cells})")
    out.append(f"  total bugs           {sum(overall.values()):>2}")
    out.append("")
    out.append("items per repository:")
    for repo, n in items_per_repo(items).items():
        out.append(f"  {repo:<14} {n}")
    new = [i for i in items if session_of(i.item_id) in ("E1", "E2")]
    ml = multi_line_bugs(new)
    cf = cross_file_bugs(new)
    out.append("")
    out.append(f"multi-line new bugs: {len(ml)}  {', '.join(f'{i}/{b}' for i, b in ml)}")
    out.append(f"cross-file new bugs: {len(cf)}  {', '.join(f'{i}/{b}' for i, b in cf)}")
    if spawn is not None:
        out.append("")
        out.append("D12 spawning (specialists that would spawn):")
        for item_id, names, n, tokens in spawn:
            out.append(
                f"  {item_id}  py_files={n:<2} tokens~{tokens:<6} "
                + (", ".join(names) if names else "none")
            )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.dataset_stats", description=__doc__)
    parser.add_argument("--no-spawn", action="store_true", help="skip the D12 spawn rows")
    args = parser.parse_args(argv)
    items = load_items()
    for item in items:  # the bucket is recomputed from pr.diff in check_dataset; here from meta
        assert item.bucket in ("small", "medium", "large")
    spawn = None if args.no_spawn else spawn_rows(items)
    print(format_report(items, spawn))
    problems = check_targets(items)
    print()
    if problems:
        print("TARGET MISMATCH")
        for p in problems:
            print(f"  {p}")
        return 1
    print("targets: all built items and completed sessions match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
