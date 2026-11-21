"""What a reviewer agent returns, and the deterministic post-processing every agent's findings go
through (decision 0007, D11).

``FindingDraft`` is ``Finding`` without ``agent`` and ``model`` (code fills those in) and with only
simple constraints, so the structured-output schema stays plain for Gemini. ``to_findings`` turns
drafts into ``Finding`` objects in four steps, always in this order:

1. ``schema``: build the ``Finding``; a ``ValidationError`` drops the draft.
2. ``not_in_diff``: the file must be a changed, non-deleted file of the diff. The D9a +-3 line
   filter is not applied here; the scorer applies it.
3. ``evidence`` (D11): the stripped evidence lines must equal a contiguous run of the file's
   stripped lines after the PR, at least one of them inside ``[line_start, line_end]`` and all of
   them inside ``[line_start - 3, line_end + 3]``.
4. ``duplicate``: same file, same category and overlapping line ranges: the higher confidence
   wins (the earlier finding on a tie).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reviewer.schemas import Category, Finding
from reviewer.tools.context import ReviewContext, ToolError
from reviewer.tools.files import _read_text, resolve_repo_path, split_file_lines

EVIDENCE_TOLERANCE = 3  # lines around [line_start, line_end] that evidence may also cover

# ``not_assigned`` (stage 4A, decision 0012): a sub-agent's finding outside its assigned files.
type DropReason = Literal["schema", "not_in_diff", "evidence", "duplicate", "not_assigned"]

CategoryLiteral = Category.__value__  # the plain Literal, so the JSON schema has no $defs


class FindingDraft(BaseModel):
    """One defect in the pull request."""

    # ``Finding`` minus ``agent`` and ``model``. The docstring above is sent to the model as
    # the schema description, so it stays model-facing.

    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="Repository-relative POSIX path of a changed file")
    line_start: int = Field(ge=1, description="First defective line, new-file numbering")
    line_end: int = Field(ge=1, description="Last defective line, new-file numbering")
    category: CategoryLiteral  # type: ignore[valid-type]
    severity: Literal["high", "medium", "low"]
    confidence: float = Field(ge=0.0, le=1.0, description="Probability this is a real defect")
    title: str = Field(description="One-line summary of the defect")
    explanation: str = Field(description="Why this is a defect and what goes wrong")
    evidence: str = Field(
        description=(
            "1 to 5 lines of code copied exactly from the file, without the line numbers, "
            "'NN | ' prefixes or diff markers that the tools add"
        )
    )


class ReviewOutput(BaseModel):
    """The reviewer's final answer."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="2 to 4 sentences on what the change does")
    findings: list[FindingDraft] = Field(
        description="One entry per distinct defect; empty when the change has no defects"
    )


class Dropped(BaseModel):
    """A draft that post-processing removed, with the step that removed it."""

    model_config = ConfigDict(extra="forbid")

    draft: dict[str, Any]
    reason: DropReason
    detail: str


def evidence_lines(evidence: str) -> list[str]:
    """Evidence split on newlines, leading and trailing blank lines dropped, each line stripped."""
    lines = [ln.strip() for ln in evidence.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def evidence_problem(finding: Finding, file_lines: list[str]) -> str | None:
    """Why ``finding.evidence`` fails D11 against ``file_lines`` (the file after the PR), or None.

    Accepted when the stripped evidence equals a contiguous run of stripped file lines
    ``s..e`` (1-based) with ``s <= line_end``, ``e >= line_start`` (one line inside the range)
    and ``line_start - 3 <= s``, ``e <= line_end + 3`` (every line near the range).
    """
    want = evidence_lines(finding.evidence)
    if not want:
        return "evidence is blank"
    have = [ln.strip() for ln in file_lines]
    k = len(want)
    runs = [s + 1 for s in range(len(have) - k + 1) if have[s : s + k] == want]
    if not runs:
        return "evidence does not match any contiguous run of lines in the file"
    lo = finding.line_start - EVIDENCE_TOLERANCE
    hi = finding.line_end + EVIDENCE_TOLERANCE
    for s in runs:
        e = s + k - 1
        if s <= finding.line_end and e >= finding.line_start and lo <= s and e <= hi:
            return None
    where = ", ".join(f"{s}-{s + k - 1}" for s in runs[:5])
    return (
        f"evidence matches lines {where}, not lines {finding.line_start}-{finding.line_end} "
        f"(within {EVIDENCE_TOLERANCE})"
    )


def _overlap(a: Finding, b: Finding) -> bool:
    return a.line_start <= b.line_end and a.line_end >= b.line_start


def _is_duplicate(a: Finding, b: Finding) -> bool:
    return a.file == b.file and a.category == b.category and _overlap(a, b)


def _draft_dict(draft: FindingDraft | dict[str, Any]) -> dict[str, Any]:
    return draft.model_dump() if isinstance(draft, BaseModel) else dict(draft)


def to_findings(
    drafts: list[FindingDraft] | list[dict[str, Any]],
    agent: str,
    model: str,
    ctx: ReviewContext,
) -> tuple[list[Finding], list[Dropped]]:
    """Post-process one agent's drafts (module docstring). Returns the kept findings in their
    original order and every dropped draft with its reason."""
    dropped: list[Dropped] = []
    changed = {f.path for f in ctx.files if f.status != "deleted"}
    file_cache: dict[str, list[str] | str] = {}

    # Steps 1 to 3, one draft at a time.
    passed: list[tuple[dict[str, Any], Finding]] = []
    for draft in drafts:
        data = _draft_dict(draft)
        try:
            finding = Finding(**{**data, "agent": agent, "model": model})
        except (ValidationError, TypeError) as e:
            dropped.append(Dropped(draft=data, reason="schema", detail=str(e)))
            continue
        if finding.file not in changed:
            detail = f"{finding.file!r} is not a changed (non-deleted) file of the diff"
            dropped.append(Dropped(draft=data, reason="not_in_diff", detail=detail))
            continue
        if finding.file not in file_cache:
            try:
                clean, target = resolve_repo_path(ctx, finding.file)
                file_cache[finding.file] = split_file_lines(_read_text(target, clean))
            except ToolError as e:
                file_cache[finding.file] = f"cannot read the file: {e}"
        lines = file_cache[finding.file]
        problem = lines if isinstance(lines, str) else evidence_problem(finding, lines)
        if problem is not None:
            dropped.append(Dropped(draft=data, reason="evidence", detail=problem))
            continue
        passed.append((data, finding))

    # Step 4: highest confidence first (earlier index on ties); keep a finding unless it
    # duplicates one already kept.
    order = sorted(range(len(passed)), key=lambda i: (-passed[i][1].confidence, i))
    kept_idx: list[int] = []
    for i in order:
        f = passed[i][1]
        winner = next((j for j in kept_idx if _is_duplicate(passed[j][1], f)), None)
        if winner is None:
            kept_idx.append(i)
            continue
        w = passed[winner][1]
        detail = (
            f"duplicates {w.file}:{w.line_start}-{w.line_end} ({w.category}, "
            f"confidence {w.confidence}), which has the higher confidence"
        )
        dropped.append(Dropped(draft=passed[i][0], reason="duplicate", detail=detail))
    kept = [passed[i][1] for i in sorted(kept_idx)]
    return kept, dropped
