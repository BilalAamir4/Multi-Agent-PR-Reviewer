"""The ``Finding`` schema (plan § "The Finding schema", D8, D10, D11)."""

from __future__ import annotations

import re
from typing import Any, Literal, Self, TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type Category = Literal[
    "off_by_one",
    "missing_none_check",
    "wrong_comparison",
    "unhandled_exception",
    "sql_injection",
    "hardcoded_secret",
    "ineffective_test",
    "other",
]
type Severity = Literal["high", "medium", "low"]

ALL_CATEGORIES: tuple[str, ...] = get_args(Category.__value__)
PLANTED_CATEGORIES: tuple[str, ...] = tuple(c for c in ALL_CATEGORIES if c != "other")

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def validate_posix_path(value: str) -> str:
    """Validate a repository-relative POSIX path (D3). Strips one leading ``./``.

    Rejects empty strings, backslashes, absolute paths (leading ``/`` or a drive letter)
    and any ``..`` segment. ``a/`` and ``b/`` prefixes are not stripped.
    """
    if value.startswith("./"):
        value = value[2:]
    if not value:
        raise ValueError("path must not be empty")
    if "\\" in value:
        raise ValueError("path must use forward slashes, not backslashes")
    if value.startswith("/") or _DRIVE_RE.match(value):
        raise ValueError("path must be relative, not absolute")
    if ".." in value.split("/"):
        raise ValueError("path must not contain '..' segments")
    return value


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace")
    return value


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str  # POSIX path from the diff
    line_start: int = Field(ge=1)
    line_end: int
    category: Category
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    title: str
    explanation: str
    evidence: str  # exact code quoted from the file; stored unstripped (D11)
    agent: str  # role that produced it
    model: str  # model that produced it (routing doc S7)

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str) -> str:
        return validate_posix_path(v)

    @field_validator("title", "explanation", "evidence", "agent", "model")
    @classmethod
    def _check_non_blank(cls, v: str) -> str:
        return _non_blank(v)

    @model_validator(mode="after")
    def _check_lines(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


# --- The review graph's state (stage 4B, decision 0013) -----------------------------------------

type JudgeKind = Literal["verifier", "orchestrator", "none"]
type ReviewStatus = Literal["complete", "partial", "limit_reached", "quota_paused", "error"]


class ReviewState(TypedDict, total=False):
    """State of the review graph ``START -> review -> judge -> present -> END``.

    Everything is JSON-serializable (findings and results are stored as plain dicts), so a
    checkpointer can save it after any node (stage 6). The first block is the input; every node
    returns only its own keys: ``review`` writes ``review_result``, ``findings``, ``status`` and
    ``notes``; ``judge`` writes ``judge_result``, ``final_findings``, ``status`` and ``notes``;
    ``present`` writes ``final_findings`` (sorted) and ``approval_payload``. ``Finding`` itself is
    unchanged: a finding in the state is ``Finding.model_dump(mode="json")``."""

    # input
    review_key: str  # "<repo or local>#<pr or item id>@<head sha or diff sha256[:12]>"; thread_id
    pr_title: str
    pr_body: str
    workspace: str  # checkout of the repository after the PR
    diff_path: str
    sandbox_repo: str | None  # eval/repos.yaml entry whose image run_tests uses
    security_mode: Literal["local", "remote"] | None  # None: config/review.yaml
    spawn_enabled: bool
    judge: JudgeKind
    evaluation: bool
    # written by the nodes
    review_result: dict[str, Any]  # the orchestrator's OrchestratorResult, model_dump(mode="json")
    findings: list[dict[str, Any]]  # the review's merged findings (after D11 and the merge)
    judge_result: dict[str, Any] | None  # the judge step's JudgeStepResult, None for "none"
    final_findings: list[dict[str, Any]]
    approval_payload: list[dict[str, Any]]
    status: ReviewStatus
    notes: list[str]
