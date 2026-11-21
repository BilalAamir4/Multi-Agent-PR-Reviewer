"""Plain-text and merge helpers shared by the orchestrator and by specialists that spawn
sub-agents (stage 4A, decision 0012). Moved here from ``orchestrator.py`` unchanged."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from reviewer.agents.output import Dropped, _is_duplicate
from reviewer.schemas import Finding

if TYPE_CHECKING:
    from reviewer.agents.base import ReviewResult


def finding_id(agent: str, index: int) -> str:
    """``<agent>#<index>``, 1-based over the agent's findings."""
    return f"{agent}#{index}"


def one_line(text: str) -> str:
    return " ".join(text.split())


def format_specialist_output(result: ReviewResult) -> str:
    """The compact text the orchestrator reads: name, status, notes, one line per finding (no
    explanations) and the dropped counts."""
    lines = [f"specialist: {result.agent}", f"status: {result.status}"]
    if result.status in ("error", "limit_reached"):
        why = result.limit or f"{result.error_class}: {result.error}"
        lines.append(f"reason: {one_line(why)}")
    for note in result.notes:
        lines.append(f"note: {one_line(note)}")
    lines.append(
        f"findings ({len(result.findings)}): id | file:lines | category | severity | "
        "confidence | title"
    )
    for i, f in enumerate(result.findings, 1):
        span = f"{f.line_start}-{f.line_end}"
        lines.append(
            f"{finding_id(result.agent, i)} | {f.file}:{span} | {f.category} | {f.severity} | "
            f"{f.confidence:.2f} | {one_line(f.title)}"
        )
    if not result.findings:
        lines.append("(none)")
    if result.dropped:
        counts = Counter(d.reason for d in result.dropped)
        lines.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def merge_candidates(cands: list[tuple[str, Finding]]) -> tuple[list[Finding], list[Dropped]]:
    """The deterministic duplicate rule of decision 0009 over ``cands`` (``(id, finding)`` in
    tie-break order).

    Candidates are visited from the highest confidence down, a tie going to the earlier
    candidate; each is kept unless it duplicates one already kept (same file, same category,
    overlapping lines). A dropped one is recorded with the reason ``duplicate`` and the id of the
    finding it duplicated. Kept findings stay in candidate order and keep their own ``agent`` and
    ``model``."""
    order = sorted(range(len(cands)), key=lambda i: (-cands[i][1].confidence, i))
    kept: list[int] = []
    dropped: list[Dropped] = []
    for i in order:
        fid, f = cands[i]
        winner = next((j for j in kept if _is_duplicate(cands[j][1], f)), None)
        if winner is None:
            kept.append(i)
            continue
        wid, w = cands[winner]
        draft = {**f.model_dump(mode="json"), "finding_id": fid, "duplicate_of": wid}
        detail = (
            f"{fid} duplicates {wid} ({w.file}:{w.line_start}-{w.line_end}, {w.category}, "
            f"confidence {w.confidence}), which is kept"
        )
        dropped.append(Dropped(draft=draft, reason="duplicate", detail=detail))
    return [cands[i][1] for i in sorted(kept)], dropped
