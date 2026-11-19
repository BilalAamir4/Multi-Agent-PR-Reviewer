"""Stage 0 done criterion (a): one traced model call returns a valid ``Finding``.

Makes exactly one call to the evaluation model (config/models.yaml) with a synthetic file that
contains one planted off-by-one bug, then fetches the LangSmith root run and prints its URL.

    uv run python scripts/check_finding.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
PROJECT = "multi-agent-pr-reviewer"
SYNTHETIC_PATH = "src/stats.py"

# Synthetic code written for this check only (never real repository content).
SYNTHETIC_FILE = '''\
"""Small statistics helpers."""


def mean(xs):
    """Return the arithmetic mean of xs."""
    return sum(xs) / len(xs)


def total_positive(xs):
    """Return the sum of the positive numbers in xs."""
    total = 0
    for i in range(len(xs) + 1):
        if xs[i] > 0:
            total += xs[i]
    return total
'''
PLANTED_LINES = (12, 13)  # the range() line and the line that indexes xs[i]

SYSTEM = (
    "You are a code reviewer. Report exactly one finding for the most serious bug in the file. "
    "Quote `evidence` exactly from the code, without the line-number prefixes. "
    'Set `file` to "{file}", `agent` to "env_check" and `model` to "{model}". '
    "Line numbers are 1-based, as shown."
)


@dataclass
class FindingCheck:
    ok: bool = False
    error: str | None = None
    finding: object | None = None
    category: str | None = None
    lines_hit: bool | None = None
    evidence_verbatim: bool | None = None
    trace_url: str | None = None
    trace_error: str | None = None
    key_in_trace: bool | None = None
    notes: list[str] = field(default_factory=list)


def numbered(text: str) -> str:
    return "\n".join(f"{i:>3}: {line}" for i, line in enumerate(text.splitlines(), start=1))


def missing_requirements() -> list[str]:
    load_dotenv(REPO / ".env")
    return [k for k in ("GOOGLE_API_KEY", "LANGSMITH_API_KEY") if not os.environ.get(k, "").strip()]


def _secret_prefixes() -> list[str]:
    keys = ("GOOGLE_API_KEY", "GROQ_API_KEY", "LANGSMITH_API_KEY")
    return [os.environ[k].strip()[:6] for k in keys if os.environ.get(k, "").strip()]


def _overlaps(start: int, end: int, planted: tuple[int, int], slack: int = 3) -> bool:
    return start <= planted[1] + slack and end >= planted[0] - slack


def run_check() -> FindingCheck:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tracers.context import collect_runs
    from langchain_core.tracers.langchain import wait_for_all_tracers

    from reviewer.config import load_models_config
    from reviewer.schemas import Finding

    res = FindingCheck()
    load_dotenv(REPO / ".env")
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        res.error = "LANGSMITH_TRACING is not 'true' in .env"
        return res
    if os.environ.get("LANGSMITH_PROJECT") != PROJECT:
        res.error = f"LANGSMITH_PROJECT is not {PROJECT!r}"
        return res

    model_id = load_models_config().evaluation.model
    # max_retries=0: the SDK would otherwise retry silently and spend the daily request quota.
    llm = init_chat_model(model_id, model_provider="google_genai", temperature=0, max_retries=0)
    structured = llm.with_structured_output(Finding)
    messages = [
        SystemMessage(SYSTEM.format(file=SYNTHETIC_PATH, model=model_id)),
        HumanMessage(f"File `{SYNTHETIC_PATH}`:\n\n{numbered(SYNTHETIC_FILE)}"),
    ]

    run_id = None
    try:
        with collect_runs() as cb:
            result = structured.invoke(messages, config={"run_name": "env_check_finding"})
            if cb.traced_runs:
                run_id = cb.traced_runs[0].id
    except Exception as e:  # report the exact error; do not edit Finding
        res.error = f"{type(e).__module__}.{type(e).__qualname__}: {e}"
        result = None

    if result is not None:
        res.finding = result
        res.ok = isinstance(result, Finding)
        if res.ok:
            res.category = result.category
            res.lines_hit = _overlaps(result.line_start, result.line_end, PLANTED_LINES)
            res.evidence_verbatim = result.evidence.strip() in SYNTHETIC_FILE
        else:
            res.error = f"result is {type(result).__name__}, not Finding"

    wait_for_all_tracers()
    _fetch_trace(res, run_id)
    return res


def _fetch_trace(res: FindingCheck, run_id) -> None:
    import langsmith
    from langsmith.utils import LangSmithAuthError, LangSmithError

    client = langsmith.Client()
    try:
        run = None
        for _ in range(10):  # LangSmith ingests asynchronously; read-only polling
            if run_id is not None:
                try:
                    run = client.read_run(run_id)
                except LangSmithError:
                    run = None
            if run is None:
                latest = list(client.list_runs(project_name=PROJECT, is_root=True, limit=1))
                run = latest[0] if latest else None
                if run is not None and run_id is not None and run.id != run_id:
                    run = None
            if run is not None:
                break
            time.sleep(2)
        if run is None:
            res.trace_error = "no root run appeared in LangSmith within 20 s"
            return
        res.trace_url = client.get_run_url(run=run, project_name=PROJECT)
        dumped = json.dumps(
            {"inputs": run.inputs, "outputs": run.outputs, "extra": run.extra}, default=str
        )
        res.key_in_trace = any(p and p in dumped for p in _secret_prefixes())
    except LangSmithAuthError as e:
        res.trace_error = (
            f"LangSmith auth error (401/403). It may be an EU-region account; ask whether "
            f"LANGSMITH_ENDPOINT should be https://eu.api.smith.langchain.com. {e}"
        )


def main() -> int:
    missing = missing_requirements()
    if missing:
        print(f"Missing from .env: {', '.join(missing)}")
        return 2
    res = run_check()
    print(f"valid Finding returned: {'PASS' if res.ok else 'FAIL'}")
    if res.error:
        print(f"error: {res.error}")
    if res.ok:
        f = res.finding
        print(f"category: {res.category} (planted: off_by_one)")
        print(f"lines: {f.line_start}-{f.line_end}, planted {PLANTED_LINES[0]}-{PLANTED_LINES[1]}")
        print(f"lines overlap within 3 (info): {res.lines_hit}")
        print(f"evidence verbatim in file (info): {res.evidence_verbatim}")
        print(f"evidence: {f.evidence!r}")
        print(f"title: {f.title}")
    if res.trace_url:
        print(f"LangSmith trace: {res.trace_url}")
        print(f"key material in trace: {res.key_in_trace}")
    if res.trace_error:
        print(f"trace: {res.trace_error}")
    return 0 if res.ok and res.trace_url and not res.key_in_trace else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.exit(main())
