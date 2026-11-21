"""The wire format between the orchestrator and the security specialist service (stage 3,
decision 0011).

The request travels as one JSON text part of an A2A ``message/send`` message; the response comes
back as the JSON text of the graph's last AI message (an A2A task artifact). Both are validated
here, by the client before it sends and by the service on arrival, so a malformed request or
response is a clean error and never an exception in the middle of a review.
"""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reviewer.agents.base import ReviewResult, SpawnInfo, Status
from reviewer.agents.output import Dropped
from reviewer.schemas import Finding

# "2" (stage 4A, decision 0012): the request carries ``spawn_enabled`` and the response the
# specialist's sub-agents. The service and the client change together; "1" is rejected.
PROTOCOL_VERSION = "2"
SERVICE_NAME = "security_reviewer"  # the graph name in a2a_security/langgraph.json

# Archive safety limits (decision 0011). Checked before decoding and while unpacking.
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024  # compressed
MAX_UNPACKED_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 20_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Keys of the trace headers ``trace_parent`` may carry (``RunTree.to_headers()``).
TRACE_HEADER_KEYS = frozenset({"langsmith-trace", "baggage"})


class SecurityReviewRequest(BaseModel):
    """What the orchestrator sends: the pull request and a snapshot of its workspace. Strict: no
    type coercion (``"yes"`` is not ``True``)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol_version: Literal["2"]
    pr_title: str
    pr_body: str
    focus: str | None
    diff_text: str
    workspace_archive: str = Field(description="base64 of the tar.gz snapshot of the workspace")
    workspace_sha256: str = Field(description="hex SHA-256 of the decoded tar.gz bytes")
    evaluation: bool
    spawn_enabled: bool = Field(
        description="the caller allows sub-agents; the service also applies its own config"
    )
    trace_parent: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not _SHA256.match(self.workspace_sha256):
            raise ValueError("workspace_sha256 must be 64 lowercase hex characters")
        if len(self.workspace_archive) > (MAX_ARCHIVE_BYTES * 4) // 3 + 8:
            raise ValueError(f"workspace_archive is larger than {MAX_ARCHIVE_BYTES} bytes")
        if self.trace_parent is not None:
            bad = sorted(set(self.trace_parent) - TRACE_HEADER_KEYS)
            if bad:
                raise ValueError(f"trace_parent may only hold {sorted(TRACE_HEADER_KEYS)}: {bad}")
        return self

    def archive_bytes(self) -> bytes:
        """The decoded archive; ``ValueError`` when it is not valid base64 or too large."""
        try:
            data = base64.b64decode(self.workspace_archive, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"workspace_archive is not valid base64: {e}") from e
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"workspace_archive is larger than {MAX_ARCHIVE_BYTES} bytes")
        return data


class WaitInterval(BaseModel):
    """One wait of the service's review, as Unix timestamps (decision 0011, D7 across
    processes). ``reason`` is the wait's reason, so the caller's wait log stays by reason."""

    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    reason: Literal["rate_limiter", "retry_minute", "retry_server"]


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pid: int
    version: str  # the security prompt's version (12 hex characters)


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_calls: int
    tool_calls: int
    tool_calls_by_tool: dict[str, int]
    input_tokens: int
    output_tokens: int
    duration_s: float
    wait_s: float
    net_duration_s: float


class OwnCounts(BaseModel):
    """The specialist's own pass alone, and its nested wait totals (stage 4A)."""

    model_config = ConfigDict(extra="forbid")

    own_model_calls: int = 0
    own_tool_calls: int = 0
    own_tool_calls_by_tool: dict[str, int] = Field(default_factory=dict)
    own_input_tokens: int = 0
    own_output_tokens: int = 0
    wait_s_sum: float = 0.0
    wait_s_wall: float = 0.0


class SecurityReviewResponse(BaseModel):
    """``run_security_review``'s response, plus the service block the A2A service adds. An
    invalid request answers with this same shape: ``status`` ``error``, ``error_class``
    ``InvalidRequest`` or ``ArchiveError`` and the reason in ``error``."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    status: Status
    findings: list[Finding]
    dropped: list[Dropped]
    notes: list[str]
    summary: str | None
    error_class: str | None
    error: str | None
    limit: Literal["max_model_calls", "max_tool_calls"] | None
    reset_at: datetime | None
    model_id: str | None
    prompt_version: str
    evaluation: bool
    counts: Counts
    retries_by_reason: dict[str, int]
    waits_by_reason: dict[str, float]
    wait_intervals: list[WaitInterval]  # every wait of the review, sub-agents' included, once
    service: ServiceInfo | None = None
    # Stage 4A: the sub-agents' full results, the spawn record and the specialist's own
    # counters (all empty when nothing spawned).
    children: list[ReviewResult] = Field(default_factory=list)
    spawn: SpawnInfo | None = None
    own: OwnCounts = Field(default_factory=OwnCounts)


def validate_response(data: Any) -> SecurityReviewResponse:
    """The response dict as a ``SecurityReviewResponse`` (``pydantic.ValidationError`` if not)."""
    return SecurityReviewResponse.model_validate(data)
