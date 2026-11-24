"""Turn approved findings into one GitHub review (decision 0016): which findings can be inline
comments (their lines are in the PR diff), the comment text, and the summary body with the marker
that makes posting idempotent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from eval.ground_truth import parse_diff
from eval.score import changed_lines


def marker_for(review_key: str) -> str:
    return f"<!-- mapr-review:{review_key} -->"


def _fence(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def comment_body(f: dict[str, Any]) -> str:
    fence = _fence(f["evidence"])
    footer = f"found by {f['agent']} · verified: {f['verdict']}"
    if f.get("edited"):
        footer += " · edited by reviewer"
    return (
        f"**{f['title']}** · severity: {f['severity']}\n\n"
        f"{f['explanation']}\n\n"
        f"{fence}\n{f['evidence'].rstrip(chr(10))}\n{fence}\n\n"
        f"<sub>{footer}</sub>"
    )


def _hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for pf in parse_diff(diff_text):
        if pf.is_removed_file:
            continue
        out[str(PurePosixPath(pf.path))] = [
            (h.target_start, h.target_start + max(h.target_length, 1) - 1) for h in pf
        ]
    return out


@dataclass
class ReviewPlan:
    body: str
    comments: list[dict[str, Any]] = field(default_factory=list)
    inline_ids: list[str] = field(default_factory=list)
    summary_ids: list[str] = field(default_factory=list)  # approved, lines not in the diff


def build_review(approved: list[dict[str, Any]], diff_text: str, marker: str) -> ReviewPlan:
    """Inline comments for findings whose ``line_end`` (and ``line_start`` for a range) are changed
    lines of the diff; the others go into the summary body. A range whose ends are in different
    hunks becomes a single-line comment on ``line_end`` (GitHub needs both ends in one hunk)."""
    changed = changed_lines(diff_text)
    hunks = _hunks(diff_text)
    plan = ReviewPlan(body="")
    off_diff: list[dict[str, Any]] = []
    for f in approved:
        lines = changed.get(f["file"], set())
        start, end = f["line_start"], f["line_end"]
        ok = end in lines and (start == end or start in lines)
        if not ok:
            off_diff.append(f)
            plan.summary_ids.append(f["id"])
            continue
        comment: dict[str, Any] = {
            "path": f["file"],
            "line": end,
            "side": "RIGHT",
            "body": comment_body(f),
        }
        if start != end and any(a <= start and end <= b for a, b in hunks.get(f["file"], [])):
            comment["start_line"] = start
            comment["start_side"] = "RIGHT"
        plan.comments.append(comment)
        plan.inline_ids.append(f["id"])
    lines_out = [
        marker,
        f"Automated review: {len(approved)} finding(s) approved by a human reviewer, "
        f"{len(plan.comments)} as inline comment(s).",
    ]
    if off_diff:
        lines_out += ["", "Findings outside the changed lines of this diff:"]
        for f in off_diff:
            span = (
                str(f["line_start"])
                if f["line_start"] == f["line_end"]
                else f"{f['line_start']}-{f['line_end']}"
            )
            edited = " · edited by reviewer" if f.get("edited") else ""
            lines_out.append(
                f"- `{f['file']}:{span}` **{f['title']}** (severity: {f['severity']}): "
                f"{f['explanation']} (found by {f['agent']} · verified: {f['verdict']}{edited})"
            )
    plan.body = "\n".join(lines_out)
    return plan
