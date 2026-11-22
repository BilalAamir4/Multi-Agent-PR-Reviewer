"""Blind labelling of unmatched findings (plan D9b; decision 0015). Run it yourself, in stage 5Z.

    uv run python -m eval.label            # label, resume where you stopped
    uv run python -m eval.label --write    # write manual_labels.json from the labels so far
    uv run python -m eval.label --status   # counts only

Collects every strict-unmatched, in-diff finding of every configuration, merges identical claims
(same item, file and category, overlapping lines, grouped transitively) so each claim is labelled
once, shuffles them with a fixed seed and shows them **without** the configuration, agent, model,
confidence or how many configurations made the claim. For each claim you see the PR title and the
finding's file, lines, category, title, explanation and evidence, and the code around it
(+-15 lines of the PR's target state). Answers: ``r`` real bug, ``f`` false alarm, ``s`` skip for
now, ``q`` save and quit. Progress is saved in ``eval/labels/claims.json`` after every answer.
When every claim has a label (or with ``--write``) each configuration's
``eval/results/<config>/manual_labels.json`` is written in the scorer's format.

``claims.json`` also records which findings belong to each claim (``members``) so labels can be
written back; the labelling screen never prints them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from eval.ground_truth import load_item
from eval.results_data import CONFIG_ORDER, dataset_dir, dataset_ids, pr_title, results_root
from eval.score import LABELS_FILENAME, score_item
from reviewer.config import repo_root
from reviewer.schemas import Finding

SEED = 20261122
RADIUS = 15
ANSWERS = {"r": "real_bug", "f": "false_alarm", "s": "skipped"}
FINAL = ("real_bug", "false_alarm")


def labels_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "eval" / "labels" / "claims.json"


@dataclass
class Claim:
    item_id: str
    file: str
    line_start: int
    line_end: int
    category: str
    title: str
    explanation: str
    evidence: str
    members: list[tuple[str, str]] = field(default_factory=list)  # (config, finding id)
    label: str | None = None  # None, "real_bug", "false_alarm" or "skipped"

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "file": self.file, "line_start": self.line_start,
            "line_end": self.line_end, "category": self.category, "title": self.title,
            "explanation": self.explanation, "evidence": self.evidence,
            "label": self.label, "members": [list(m) for m in self.members],
        }  # fmt: skip


# --- Collecting and merging claims -----------------------------------------------------------


def strict_unmatched(dataset: Path, item_id: str, findings: list[Finding]) -> list[int]:
    """Indexes of the in-diff findings the frozen scorer leaves unmatched in strict mode."""
    gt, diff_text = load_item(dataset / item_id)
    scored = score_item(gt, diff_text, findings)
    prefix = f"{item_id}#"
    return [
        int(fid.removeprefix(prefix)) for fid in scored["modes"]["strict"]["unmatched_findings"]
    ]


def _overlap(a: Finding, b: Finding) -> bool:
    return a.line_start <= b.line_end and a.line_end >= b.line_start


def merge_claims(entries: list[tuple[str, str, Finding]]) -> list[Claim]:
    """``entries`` are ``(config, finding_id, finding)``. Two findings are the same claim when item,
    file and category agree and their line ranges overlap (no tolerance); claims are the
    transitive groups. Deterministic: the displayed text is that of the member with the highest
    confidence (ties: configuration order, then finding id)."""
    order = {c: n for n, c in enumerate(CONFIG_ORDER)}
    entries = sorted(entries, key=lambda e: (order.get(e[0], 99), e[1]))
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def item_of(fid: str) -> str:
        return fid.split("#", 1)[0]

    for i, (_, fid_i, f_i) in enumerate(entries):
        for j in range(i):
            _, fid_j, f_j = entries[j]
            if (
                item_of(fid_i) == item_of(fid_j)
                and f_i.file == f_j.file
                and f_i.category == f_j.category
                and _overlap(f_i, f_j)
            ):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(entries)):
        groups.setdefault(find(i), []).append(i)
    claims = []
    for idxs in groups.values():
        best = min(idxs, key=lambda k: (-entries[k][2].confidence, k))
        _, fid, f = entries[best]
        claims.append(
            Claim(
                item_id=item_of(fid),
                file=f.file,
                line_start=f.line_start,
                line_end=f.line_end,
                category=f.category,
                title=f.title,
                explanation=f.explanation,
                evidence=f.evidence,
                members=[(entries[k][0], entries[k][1]) for k in idxs],
            )  # fmt: skip
        )
    return claims


def collect_claims(
    results: Path, dataset: Path, configs: Sequence[str] | None = None
) -> list[Claim]:
    """Every strict-unmatched, in-diff finding of every configuration with saved findings,
    merged into claims, in a fixed-seed shuffled order."""
    entries: list[tuple[str, str, Finding]] = []
    for config in configs or CONFIG_ORDER:
        if not (results / config).is_dir():
            continue
        for item_id in dataset_ids(dataset):
            path = results / config / f"{item_id}.json"
            if not path.is_file():
                continue
            findings = [
                Finding.model_validate(f) for f in json.loads(path.read_bytes())["findings"]
            ]
            for idx in strict_unmatched(dataset, item_id, findings):
                entries.append((config, f"{item_id}#{idx}", findings[idx]))
    claims = merge_claims(entries)
    claims.sort(key=lambda c: (c.item_id, c.file, c.line_start, c.category, c.members[0]))
    random.Random(SEED).shuffle(claims)
    return claims


def apply_saved(claims: list[Claim], saved: dict[str, Any] | None) -> None:
    """Carry labels over from a saved ``claims.json``: a claim takes the label of any saved claim
    that shares a member with it (a final label beats ``skipped``)."""
    if not saved:
        return
    by_member: dict[tuple[str, str], str] = {}
    for sc in saved.get("claims", []):
        if sc.get("label"):
            for m in sc["members"]:
                by_member[(m[0], m[1])] = sc["label"]
    for c in claims:
        found = [by_member[m] for m in c.members if m in by_member]
        finals = [x for x in found if x in FINAL]
        c.label = finals[0] if finals else ("skipped" if found else None)


def load_state(path: Path, claims: list[Claim]) -> None:
    if path.is_file():
        apply_saved(claims, json.loads(path.read_bytes()))


def save_state(path: Path, claims: list[Claim]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"seed": SEED, "claims": [c.to_json() for c in claims]}
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    tmp.replace(path)


# --- Showing a claim -------------------------------------------------------------------------


def code_around(workspace: Path, file: str, start: int, end: int, radius: int = RADIUS) -> str:
    """The file's lines from ``start - radius`` to ``end + radius`` with line numbers; the
    claim's own lines are marked with ``>``."""
    path = workspace / PurePosixPath(file)
    if not path.is_file():
        return f"(file not found in the workspace: {file})"
    lines = path.read_bytes().decode("utf-8", errors="replace").split("\n")
    lo, hi = max(1, start - radius), min(len(lines), end + radius)
    out = []
    for n in range(lo, hi + 1):
        mark = ">" if start <= n <= end else " "
        out.append(f"{mark} {n:5d} | {lines[n - 1].rstrip(chr(13))}")
    return "\n".join(out)


def render_claim(claim: Claim, pr_title_text: str, code: str, position: str) -> str:
    """The screen for one claim. It shows no configuration, agent, model, confidence or count."""
    return "\n".join(
        [
            "=" * 78,
            f"{position}   PR: {pr_title_text}",
            f"File: {claim.file}   Lines: {claim.line_start}-{claim.line_end}   "
            f"Category: {claim.category}",
            f"Title: {claim.title}",
            "",
            "Explanation:",
            claim.explanation,
            "",
            "Evidence:",
            claim.evidence,
            "",
            "Code (PR target state):",
            code,
            "",
        ]
    )


# --- The session -----------------------------------------------------------------------------


def run_session(
    claims: list[Claim],
    *,
    title_of: Callable[[str], str],
    workspace_of: Callable[[str], Path],
    save: Callable[[], None],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> str:
    """Show the claims that have no final label (never shown first, then the skipped ones), one at
    a time. Returns ``"done"`` (every claim has a final label), ``"quit"`` or ``"pending"`` (some
    were skipped)."""
    total = len(claims)
    # Pass 1: claims never shown. Pass 2: the ones skipped (now or in an earlier session), once.
    for wanted in (None, "skipped"):
        for c in [c for c in claims if c.label == wanted]:
            done = sum(1 for x in claims if x.label in FINAL)
            screen = render_claim(
                c,
                title_of(c.item_id),
                code_around(workspace_of(c.item_id), c.file, c.line_start, c.line_end),
                f"[{done + 1}/{total}]",
            )
            print_fn(screen)
            while True:
                try:
                    answer = input_fn(
                        "Is this a real bug?  r = real, f = false alarm, s = skip, q = quit > "
                    )
                except EOFError:
                    answer = "q"
                answer = answer.strip().lower()
                if answer in ANSWERS or answer == "q":
                    break
                print_fn("Please answer r, f, s or q.")
            if answer == "q":
                save()
                return "quit"
            c.label = ANSWERS[answer]
            save()
    return "done" if all(c.label in FINAL for c in claims) else "pending"


def write_manual_labels(claims: list[Claim], results: Path) -> list[str]:
    """Write ``manual_labels.json`` for every configuration that has claim members: finding id ->
    ``real_bug`` | ``false_alarm`` for the labelled ones (skipped and unlabelled claims are left
    out, so the scorer reports them as unlabelled). Returns the configurations written."""
    # Every configuration with findings gets a file (an empty one when it has nothing to label),
    # so the scorer computes its adjusted precision instead of leaving it out.
    per_config: dict[str, dict[str, str]] = {
        config: {} for config in CONFIG_ORDER if any((results / config).glob("item_*.json"))
    }
    for c in claims:
        for config, fid in c.members:
            per_config.setdefault(config, {})
            if c.label in FINAL:
                per_config[config][fid] = c.label
    for config, labels in per_config.items():
        path = results / config / LABELS_FILENAME
        path.write_bytes(
            (json.dumps(dict(sorted(labels.items())), indent=2) + "\n").encode("utf-8")
        )
    return sorted(per_config)


def status_line(claims: list[Claim]) -> str:
    n = lambda lab: sum(1 for c in claims if c.label == lab)  # noqa: E731
    unl = sum(1 for c in claims if c.label is None)
    return (
        f"{len(claims)} claims: {n('real_bug')} real, {n('false_alarm')} false alarms, "
        f"{n('skipped')} skipped, {unl} not yet shown"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.label", description=__doc__)
    parser.add_argument("--write", action="store_true", help="write manual_labels.json and exit")
    parser.add_argument("--status", action="store_true", help="show counts and exit")
    parser.add_argument("--results", type=Path, default=results_root())
    parser.add_argument("--dataset", type=Path, default=dataset_dir())
    parser.add_argument("--labels", type=Path, default=labels_path())
    args = parser.parse_args(argv)

    claims = collect_claims(args.results, args.dataset)
    load_state(args.labels, claims)
    save_state(args.labels, claims)
    print(status_line(claims))
    if args.status:
        return 0
    if not args.write:
        from eval.repos import materialize

        cache: dict[str, Path] = {}

        def workspace_of(item_id: str) -> Path:
            if item_id not in cache:
                cache[item_id] = materialize(item_id)
            return cache[item_id]

        outcome = run_session(
            claims,
            title_of=lambda i: pr_title(args.dataset, i),
            workspace_of=workspace_of,
            save=lambda: save_state(args.labels, claims),
        )
        print(status_line(claims))
        if outcome != "done":
            print("Labelling is not finished; run again to continue (--write writes what exists).")
            return 0
    written = write_manual_labels(claims, args.results)
    print(f"wrote {LABELS_FILENAME} for: {', '.join(written) or '(no findings to label)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
