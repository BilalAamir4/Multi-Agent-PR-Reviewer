"""Stage 4B dev check: run the verifier and the orchestrator-as-judge once each on the hand-built
mix of findings ``tests/fixtures/dev_pr/judge_mix.json`` (two real defects of the ``dev_pr``
fixture, one plausible but wrong finding, one wrong finding about a test) and print each verdict.

    uv run python scripts/check_judges.py [--judge verifier|orchestrator|both] [--eval-mode]

Informational only: nothing asserts the verdicts. It spends model calls (the verifier: up to 6 per
finding; the judge: up to 8 in all), so run it once, not in a loop. Uses the ``dev_pr`` fixture
only, never a dataset item. ``--eval-mode`` is the default (the pinned evaluation model, D4).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewer.agents.judge import run_judge  # noqa: E402
from reviewer.agents.judging import JudgeStepResult, assign_refs  # noqa: E402
from reviewer.agents.output import evidence_problem  # noqa: E402
from reviewer.agents.verifier import run_verifier  # noqa: E402
from reviewer.config import load_review_config  # noqa: E402
from reviewer.models import QuotaRouter  # noqa: E402
from reviewer.schemas import Finding  # noqa: E402
from reviewer.tools import ReviewContext  # noqa: E402
from reviewer.tools.files import _read_text, resolve_repo_path, split_file_lines  # noqa: E402
from tests.dev_pr import BODY, DEV_PR, TITLE  # noqa: E402

MIX = DEV_PR / "judge_mix.json"


def load_mix(ctx: ReviewContext, model: str) -> tuple[list[dict], list[Finding]]:
    """The mix's entries and its findings; every finding's evidence must pass D11 on the
    workspace (a typo in the mix would make a judge reject it for the wrong reason)."""
    data = json.loads(MIX.read_bytes())
    findings = [Finding(**{**e["finding"], "model": model}) for e in data["entries"]]
    for f in findings:
        clean, target = resolve_repo_path(ctx, f.file)
        problem = evidence_problem(f, split_file_lines(_read_text(target, clean)))
        if problem:
            raise SystemExit(f"judge_mix.json: {f.file}:{f.line_start}: {problem}")
    return data, findings


def show(step: JudgeStepResult, data: dict, findings: list[Finding]) -> None:
    refs = assign_refs(findings)
    by = {v.finding_ref: v for v in step.verifications}
    print(f"\n== {step.judge} ({step.status}) ==")
    for entry, ref, f in zip(data["entries"], refs, findings, strict=True):
        v = by.get(ref)
        verdict = f"{v.verdict}: {v.reason}" if v else "(no verdict)"
        print(f"- [{entry['label']}] {f.file}:{f.line_start}-{f.line_end} {f.title}")
        print(f"    {verdict}")
    kept, dropped = len(step.findings), len(step.dropped)
    print(f"kept {kept}, dropped {dropped}; notes: {step.notes or 'none'}")
    print(
        f"calls: {step.model_calls} model, {step.tool_calls} tool; "
        f"tokens: {step.input_tokens} in / {step.output_tokens} out; "
        f"{step.net_duration_s:.1f}s net + {step.wait_s_wall:.1f}s waiting; "
        f"prompt {step.prompt_version}"
    )
    for c in step.children:
        if c.langsmith_run_id:
            print(f"trace: LangSmith run {c.langsmith_run_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=("verifier", "orchestrator", "both"), default="both")
    parser.add_argument("--normal-mode", action="store_true", help="use the role chains, not D4")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    router = QuotaRouter(evaluation=not args.normal_mode)
    config = load_review_config()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        shutil.copytree(DEV_PR / "workspace", workspace)
        ctx = ReviewContext.from_paths(workspace, DEV_PR / "pr.diff")
        model = router.config.evaluation.model
        data, findings = load_mix(ctx, model)
        if args.judge in ("verifier", "both"):
            step = run_verifier(ctx, TITLE, findings, router, config=config)
            show(step, data, findings)
        if args.judge in ("orchestrator", "both"):
            step = run_judge(ctx, data["plan"], data["summary"], findings, router, config=config)
            show(step, data, findings)
    print(f"\n(PR title for the verifier: {TITLE!r}; body not used: {BODY[:30]!r}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
