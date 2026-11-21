"""``reviewer`` command line (stage 1B; specialists since stage 2A).

    uv run reviewer review [--agent NAME] --workspace PATH --diff PATH [--title TEXT]
        [--body-file PATH] [--focus TEXT] [--sandbox-repo NAME] [--out PATH] [--eval-mode]
        [--debug]
    uv run reviewer review --item item_001 [--out PATH] [--eval-mode] [--debug]

``--agent`` is ``single`` (the default, the stage 1 baseline) or one of the specialists
``bug_reviewer``, ``security_reviewer``, ``test_reviewer``. ``--focus`` is for specialists only.
``--sandbox-repo`` names an ``eval/repos.yaml`` entry whose image ``run_tests`` uses (a dataset
item uses its own repository's entry). Specialists are not run on dataset items before the
stage 5 evaluation, so ``--item`` works only with ``--agent single``.

Exit codes: 0 complete, 5 partial, 3 quota_paused, 4 limit_reached, 1 error (2 for bad
arguments).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from reviewer.agents.single import ReviewResult, review
from reviewer.agents.specialists import NAMES as SPECIALISTS
from reviewer.agents.specialists import run_specialist
from reviewer.config import repo_root
from reviewer.models import QuotaRouter
from reviewer.tools import ReviewContext, SandboxSpec

PAKISTAN = ZoneInfo("Asia/Karachi")
EXIT_CODES = {"complete": 0, "partial": 5, "quota_paused": 3, "limit_reached": 4, "error": 1}
AGENTS = ("single", *SPECIALISTS)


def make_router(evaluation: bool) -> QuotaRouter:
    return QuotaRouter(evaluation=evaluation)


def _eval_on_path() -> Path:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))  # ``eval`` is a top-level package beside ``src``
    return root


def sandbox_for_repo(name: str) -> SandboxSpec:
    """The ``run_tests`` sandbox of the ``eval/repos.yaml`` entry ``name``."""
    _eval_on_path()
    from eval.repos import sandbox_spec

    return sandbox_spec(name)


def item_inputs(item_id: str) -> tuple[ReviewContext, str, str]:
    """Materialize a dataset item and read only ``pr_title`` and ``pr_body`` from its
    ``meta.json`` (nothing else from the item reaches the agent). The context's test sandbox is
    the item's source repository's entry in ``eval/repos.yaml``."""
    root = _eval_on_path()
    from eval.repos import item_sandbox, materialize

    item_dir = root / "eval" / "dataset" / item_id
    if not (item_dir / "meta.json").is_file():
        raise FileNotFoundError(f"no dataset item {item_id!r} ({item_dir})")
    meta = json.loads((item_dir / "meta.json").read_bytes())
    title, body = str(meta["pr_title"]), str(meta["pr_body"])
    workspace = materialize(item_id)
    ctx = ReviewContext.from_paths(workspace, item_dir / "pr.diff", item_sandbox(item_id))
    return ctx, title, body


def format_reset(reset_at: datetime) -> str:
    pk = reset_at.astimezone(PAKISTAN).strftime("%Y-%m-%d %H:%M %Z")
    utc = reset_at.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M UTC")
    return f"{pk} (Pakistan time) / {utc}"


def format_table(result: ReviewResult) -> str:
    if not result.findings:
        return "No findings."
    rows = [("file:lines", "category", "severity", "conf", "title")]
    for f in result.findings:
        lines = str(f.line_start) if f.line_start == f.line_end else f"{f.line_start}-{f.line_end}"
        rows.append((f"{f.file}:{lines}", f.category, f.severity, f"{f.confidence:.2f}", f.title))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    out = []
    for r in rows:
        cells = [c.ljust(w) for c, w in zip(r[:4], widths, strict=True)]
        out.append("  ".join([*cells, r[4]]))
    return "\n".join(out)


def format_report(result: ReviewResult) -> str:
    lines = [format_table(result), ""]
    status = result.status
    if result.limit:
        status += f" ({result.limit} = {getattr(result.limits, result.limit)})"
    lines.append(f"status:  {status}")
    lines.append(f"agent:   {result.agent}")
    for note in result.notes:
        lines.append(f"note:    {note}")
    if result.summary:
        lines.append(f"summary: {result.summary}")
    if result.dropped:
        counts: dict[str, int] = {}
        for d in result.dropped:
            counts[d.reason] = counts.get(d.reason, 0) + 1
        lines.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    tools = ", ".join(f"{k} {v}" for k, v in result.tool_calls_by_tool.items()) or "none"
    lines += [
        f"model:   {result.model_id or '-'} (prompt {result.prompt_version})",
        f"calls:   {result.model_calls} model, {result.tool_calls} tool ({tools})",
        f"tokens:  {result.input_tokens} in, {result.output_tokens} out",
        f"time:    {result.net_duration_s:.1f}s net, {result.wait_s:.1f}s waiting on rate "
        f"limits, {result.duration_s:.1f}s total",
    ]
    if result.langsmith_run_id:
        lines.append(f"trace:   LangSmith run {result.langsmith_run_id}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reviewer", description="Review a Python pull request.")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review", help="review one pull request with one agent")
    r.add_argument("--agent", choices=AGENTS, default="single", help="which agent reviews")
    r.add_argument("--workspace", type=Path, help="checkout of the repository after the PR")
    r.add_argument("--diff", type=Path, help="the PR's unified diff")
    r.add_argument("--title", default="", help="pull request title")
    r.add_argument("--body-file", type=Path, help="file with the pull request description")
    r.add_argument("--focus", help="what a specialist should pay attention to")
    r.add_argument("--sandbox-repo", help="eval/repos.yaml entry whose image run_tests uses")
    r.add_argument("--item", help="dataset item id, for example item_001 (single agent only)")
    r.add_argument("--out", type=Path, help="write the full ReviewResult as JSON here")
    r.add_argument("--eval-mode", action="store_true", help="pinned evaluation model (D4)")
    r.add_argument("--debug", action="store_true", help="tracebacks and debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.item and (args.workspace or args.diff or args.title or args.body_file):
        parser.error("--item cannot be combined with --workspace, --diff, --title or --body-file")
    if not args.item and not (args.workspace and args.diff):
        parser.error("give --item, or both --workspace and --diff")
    if args.item and args.agent != "single":
        parser.error("specialists are not run on dataset items before the stage 5 evaluation")
    if args.focus and args.agent == "single":
        parser.error("--focus is for the specialists, not the single agent")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    load_dotenv(repo_root() / ".env")

    try:
        if args.item:
            ctx, title, body = item_inputs(args.item)
        else:
            sandbox = sandbox_for_repo(args.sandbox_repo) if args.sandbox_repo else None
            ctx = ReviewContext.from_paths(args.workspace, args.diff, sandbox)
            title = args.title
            body = args.body_file.read_bytes().decode("utf-8") if args.body_file else ""
        router = make_router(args.eval_mode)
        if args.agent == "single":
            result = review(ctx, title, body, router, item_id=args.item)
        else:
            result = run_specialist(
                args.agent,
                ctx,
                pr_title=title,
                pr_body=body,
                focus=args.focus,
                router=router,
                item_id=args.item,
                evaluation=args.eval_mode,
            )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        print(f"error: {type(e).__name__}: {e}".splitlines()[0], file=sys.stderr)
        return 1

    print(format_report(result))
    if result.status == "quota_paused" and result.reset_at is not None:
        print(f"quota paused: {result.model_id} resets at {format_reset(result.reset_at)}")
    elif result.status == "error":
        print(f"error: {result.error_class}: {result.error}".splitlines()[0], file=sys.stderr)
    if args.out:
        print(f"wrote {args.out}")
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
