"""``reviewer`` command line (stage 1B).

    uv run reviewer review --workspace PATH --diff PATH [--title TEXT] [--body-file PATH]
        [--out PATH] [--eval-mode] [--debug]
    uv run reviewer review --item item_001 [--out PATH] [--eval-mode] [--debug]

Exit codes: 0 complete, 3 quota_paused, 4 limit_reached, 1 error (2 for bad arguments).
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
from reviewer.config import repo_root
from reviewer.models import QuotaRouter
from reviewer.tools import ReviewContext

PAKISTAN = ZoneInfo("Asia/Karachi")
EXIT_CODES = {"complete": 0, "quota_paused": 3, "limit_reached": 4, "error": 1}


def make_router(evaluation: bool) -> QuotaRouter:
    return QuotaRouter(evaluation=evaluation)


def item_inputs(item_id: str) -> tuple[ReviewContext, str, str]:
    """Materialize a dataset item and read only ``pr_title`` and ``pr_body`` from its
    ``meta.json`` (nothing else from the item reaches the agent)."""
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))  # ``eval`` is a top-level package beside ``src``
    from eval.repos import materialize

    item_dir = root / "eval" / "dataset" / item_id
    if not (item_dir / "meta.json").is_file():
        raise FileNotFoundError(f"no dataset item {item_id!r} ({item_dir})")
    meta = json.loads((item_dir / "meta.json").read_bytes())
    title, body = str(meta["pr_title"]), str(meta["pr_body"])
    workspace = materialize(item_id)
    return ReviewContext.from_paths(workspace, item_dir / "pr.diff"), title, body


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
    r = sub.add_parser("review", help="review one pull request with the single agent")
    r.add_argument("--workspace", type=Path, help="checkout of the repository after the PR")
    r.add_argument("--diff", type=Path, help="the PR's unified diff")
    r.add_argument("--title", default="", help="pull request title")
    r.add_argument("--body-file", type=Path, help="file with the pull request description")
    r.add_argument("--item", help="dataset item id, for example item_001")
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
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    load_dotenv(repo_root() / ".env")

    try:
        if args.item:
            ctx, title, body = item_inputs(args.item)
        else:
            ctx = ReviewContext.from_paths(args.workspace, args.diff)
            title = args.title
            body = args.body_file.read_bytes().decode("utf-8") if args.body_file else ""
        result = review(ctx, title, body, make_router(args.eval_mode), item_id=args.item)
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
