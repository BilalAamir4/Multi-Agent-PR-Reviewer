"""``reviewer`` command line (stage 1B; specialists since stage 2A).

    uv run reviewer review [--agent NAME] --workspace PATH --diff PATH [--title TEXT]
        [--body-file PATH] [--focus TEXT] [--sandbox-repo NAME] [--security-mode local|remote]
        [--no-spawn] [--subagent-concurrency N] [--out PATH] [--eval-mode] [--debug]
    uv run reviewer review --item item_001 [--out PATH] [--eval-mode] [--debug]

``--agent`` is ``single`` (the default, the stage 1 baseline), one of the specialists
``bug_reviewer``, ``security_reviewer``, ``test_reviewer``, or ``orchestrator`` (stage 2B: plans,
calls the specialists as tools, merges their findings). ``--focus`` is for specialists only.
``--sandbox-repo`` names an ``eval/repos.yaml`` entry whose image ``run_tests`` uses (a dataset
item uses its own repository's entry). Specialists are not run on dataset items before the
stage 5 evaluation, and neither is the orchestrator (``eval.run --config orchestrator`` runs it
there), so ``--item`` works only with ``--agent single``. ``--security-mode`` (orchestrator and
``--agent security_reviewer`` only) overrides ``a2a.security_reviewer.mode`` of
``config/review.yaml`` for this run: ``remote`` calls the A2A security service (stage 3,
decision 0011; start it with ``scripts/run_a2a_security.py``), ``local`` runs it in this process.

``--no-spawn`` and ``--subagent-concurrency N`` (orchestrator and specialists; stage 4A,
decision 0012) override ``spawn.enabled`` and ``spawn.max_concurrency`` of ``config/review.yaml``
for this run: with spawning on, a specialist whose share of the change is large (D12) spawns
sub-agents over batches of its files. A remote security service applies its own config.

``--agent graph`` (stage 4B, decision 0013) runs the review graph (review, judge, present) with the
same options as ``--agent orchestrator`` plus ``--judge verifier|orchestrator|none`` (default
``verifier``): it prints the approval payload in order with each finding's verdict, the findings
the judge dropped and why, and the totals of each stage and overall.

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

from reviewer.agents.orchestrator import OrchestratorResult, orchestrate, run_security_remote
from reviewer.agents.single import ReviewResult, review
from reviewer.agents.specialists import NAMES as SPECIALISTS
from reviewer.agents.specialists import run_specialist
from reviewer.config import load_review_config, repo_root
from reviewer.models import QuotaRouter
from reviewer.tools import ReviewContext, SandboxSpec

PAKISTAN = ZoneInfo("Asia/Karachi")
EXIT_CODES = {"complete": 0, "partial": 5, "quota_paused": 3, "limit_reached": 4, "error": 1}
ORCHESTRATOR = "orchestrator"
GRAPH = "graph"
AGENTS = ("single", *SPECIALISTS, ORCHESTRATOR, GRAPH)
JUDGES = ("verifier", "orchestrator", "none")


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


def _retries_text(result: ReviewResult) -> str:
    """Per-call retries (decision 0010), for example ``2 retry_server, 1 retry_minute``."""
    return ", ".join(f"{n} {k}" for k, n in sorted(result.retries_by_reason.items()))


def format_hierarchy(result: ReviewResult, indent: str = "  ") -> list[str]:
    """One line per sub-agent of a specialist that spawned (status, findings, calls, tokens,
    time and waits), and the spawn decision, batches and fan-out timing (decision 0012)."""
    lines: list[str] = []
    sp = result.spawn
    if sp is None:
        return lines
    if not sp.spawned:
        return [
            f"{indent}{result.agent}: no sub-agents ({sp.py_files} .py file(s), about "
            f"{sp.estimated_tokens} tokens)"
        ]
    lines.append(
        f"{indent}{result.agent}: spawned {len(sp.batches)} sub-agent(s) because of "
        f"{' and '.join(sp.triggers)} ({sp.py_files} .py file(s), about {sp.estimated_tokens} "
        f"tokens), up to {sp.max_concurrency} at once"
    )
    for b in sp.batches:
        lines.append(f"{indent}  batch {b.index}: {', '.join(b.files)}")
    if sp.fan_out:
        f = sp.fan_out
        lines.append(
            f"{indent}  fan-out: {f.wall_s:.1f}s wall, {f.wait_s_wall:.1f}s wall-clock waiting "
            f"({f.wait_s_sum:.1f}s summed), {f.net_s:.1f}s net"
        )
    if sp.cross_file:
        c = sp.cross_file
        lines.append(
            f"{indent}  cross-file pass: {c.status}, {c.net_duration_s:.1f}s net + "
            f"{c.wait_s:.1f}s waiting, {c.model_calls} model call(s)"
        )
    for c in result.children:
        lines.append(
            f"{indent}  {c.agent}: {c.status}, {len(c.findings)} finding(s), {c.model_calls} "
            f"model / {c.tool_calls} tool calls, {c.input_tokens} in / {c.output_tokens} out, "
            f"{c.net_duration_s:.1f}s net + {c.wait_s:.1f}s waiting"
        )
    return lines


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
    if result.spawn is not None:
        lines.append("spawn:")
        lines += format_hierarchy(result)
    lines += [
        f"model:   {result.model_id or '-'} (prompt {result.prompt_version})",
        f"calls:   {result.model_calls} model, {result.tool_calls} tool ({tools})",
        f"tokens:  {result.input_tokens} in, {result.output_tokens} out",
        f"time:    {result.net_duration_s:.1f}s net, {result.wait_s:.1f}s waiting on rate "
        f"limits, {result.duration_s:.1f}s total",
    ]
    if result.retries_by_reason:
        lines.append(f"retries: {_retries_text(result)}")
    if result.langsmith_run_id:
        lines.append(f"trace:   LangSmith run {result.langsmith_run_id}")
    return "\n".join(lines)


def format_orchestrator_report(result: OrchestratorResult) -> str:
    """The orchestrator's plan, the merged findings with the agent that produced each, each
    specialist's status and the totals (calls, tokens, waits, net time)."""
    lines = ["plan:"]
    if result.plan:
        for e in result.plan:
            lines.append(f"  {e.specialist}: {'called' if e.called else 'skipped'}: {e.reason}")
    else:
        lines.append("  (no plan: the orchestrator did not finish)")
    lines.append("")
    if result.findings:
        rows = [("agent", "file:lines", "category", "severity", "conf", "title")]
        for f in result.findings:
            span = (
                str(f.line_start) if f.line_start == f.line_end else f"{f.line_start}-{f.line_end}"
            )
            rows.append(
                (
                    f.agent,
                    f"{f.file}:{span}",
                    f.category,
                    f.severity,
                    f"{f.confidence:.2f}",
                    f.title,
                )
            )
        widths = [max(len(r[i]) for r in rows) for i in range(5)]
        for r in rows:
            cells = [c.ljust(w) for c, w in zip(r[:5], widths, strict=True)]
            lines.append("  ".join([*cells, r[5]]))
    else:
        lines.append("No findings.")
    lines.append("")
    status = result.status
    if result.limit:
        status += f" ({result.limit} = {getattr(result.limits, result.limit)})"
    lines.append(f"status:  {status}")
    for note in result.notes:
        lines.append(f"note:    {note}")
    if result.summary:
        lines.append(f"summary: {result.summary}")
    if result.dropped:
        counts: dict[str, int] = {}
        for d in result.dropped:
            counts[d.reason] = counts.get(d.reason, 0) + 1
        lines.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines.append("specialists:")
    if not result.children:
        lines.append("  none called")
    for c in result.children:
        lines.append(
            f"  {c.agent}: {c.status}, {len(c.findings)} finding(s), {c.model_calls} model / "
            f"{c.tool_calls} tool calls, {c.input_tokens} in / {c.output_tokens} out, "
            f"{c.net_duration_s:.1f}s net + {c.wait_s:.1f}s waiting"
        )
        lines += format_hierarchy(c, "    ")
    lines += [
        f"model:   {result.model_id or '-'} (prompt {result.prompt_version})",
        f"calls:   {result.model_calls} model, {result.tool_calls} tool in total "
        f"(orchestrator alone: {result.own_model_calls} model, {result.own_tool_calls} tool)",
        f"tokens:  {result.input_tokens} in, {result.output_tokens} out in total "
        f"(orchestrator alone: {result.own_input_tokens} in, {result.own_output_tokens} out)",
        f"waits:   {result.wait_s_wall:.1f}s wall-clock (at least one wait in progress), "
        f"{result.wait_s_sum:.1f}s summed over all waits",
        f"time:    {result.net_duration_s:.1f}s net (total minus wall-clock waits), "
        f"{result.duration_s:.1f}s total",
    ]
    if result.retries_by_reason:
        lines.append(f"retries: {_retries_text(result)}")
    if result.langsmith_run_id:
        lines.append(f"trace:   LangSmith run {result.langsmith_run_id}")
    return "\n".join(lines)


def _stage_line(name: str, t: dict[str, float]) -> str:
    return (
        f"  {name:<7} {t['model_calls']:.0f} model / {t['tool_calls']:.0f} tool calls, "
        f"{t['input_tokens']:.0f} in / {t['output_tokens']:.0f} out, "
        f"{t['net_duration_s']:.1f}s net, waits {t['wait_s_wall']:.1f}s wall-clock "
        f"({t['wait_s_sum']:.1f}s summed), {t['duration_s']:.1f}s total"
    )


def format_graph_report(state: dict) -> str:
    """The approval payload in order with verdicts, the dropped findings with reasons, the status
    and the totals of each stage and overall (stage 4B)."""
    from reviewer.graph import stage_totals

    payload = state.get("approval_payload", [])
    lines = [f"review: {state['review_key']}  judge: {state['judge']}", ""]
    lines.append(f"approval payload ({len(payload)} finding(s), in order):")
    if not payload:
        lines.append("  (none)")
    for i, e in enumerate(payload, 1):
        a, b = e["line_start"], e["line_end"]
        span = str(a) if a == b else f"{a}-{b}"
        lines.append(
            f"  {i}. [{e['severity']}] {e['file']}:{span} {e['category']} "
            f"conf {e['confidence']:.2f} ({e['agent']}) {e['id']}"
        )
        lines.append(f"     {e['title']}")
        why = f" - {e['verdict_reason']}" if e["verdict_reason"] else ""
        lines.append(f"     verdict: {e['verdict']}{why}")
    step = state.get("judge_result") or {}
    dropped = step.get("dropped", [])
    lines.append("")
    lines.append(f"dropped by the judge ({len(dropped)}):")
    if not dropped:
        lines.append("  (none)")
    for d in dropped:
        draft = d["draft"]
        lines.append(
            f"  - {draft['file']}:{draft['line_start']}-{draft['line_end']} "
            f"{draft['category']} {draft['title']} [{d['reason']}]: {d['detail']}"
        )
    lines.append("")
    lines.append(f"status:  {state['status']}")
    for note in state.get("notes", []):
        lines.append(f"note:    {note}")
    summary = state.get("review_result", {}).get("summary")
    if summary:
        lines.append(f"summary: {summary}")
    totals = stage_totals(state)
    lines.append("stages:")
    for name in ("review", "judge", "total"):
        lines.append(_stage_line(name, totals[name]))
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reviewer", description="Review a Python pull request.")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review", help="review one pull request with one agent or the team")
    r.add_argument("--agent", choices=AGENTS, default="single", help="which agent reviews")
    r.add_argument("--workspace", type=Path, help="checkout of the repository after the PR")
    r.add_argument("--diff", type=Path, help="the PR's unified diff")
    r.add_argument("--title", default="", help="pull request title")
    r.add_argument("--body-file", type=Path, help="file with the pull request description")
    r.add_argument("--focus", help="what a specialist should pay attention to")
    r.add_argument("--sandbox-repo", help="eval/repos.yaml entry whose image run_tests uses")
    r.add_argument(
        "--security-mode",
        choices=("local", "remote"),
        help="run the security specialist locally or on the A2A service (overrides the config)",
    )
    r.add_argument(
        "--no-spawn", action="store_true", help="specialists never spawn sub-agents (this run)"
    )
    r.add_argument(
        "--subagent-concurrency",
        type=int,
        metavar="N",
        help="sub-agents running at once per specialist (default: spawn.max_concurrency)",
    )
    r.add_argument(
        "--judge",
        choices=JUDGES,
        help="graph only: who keeps or drops the merged findings (default: verifier)",
    )
    r.add_argument("--item", help="dataset item id, for example item_001 (single agent only)")
    r.add_argument("--out", type=Path, help="write the full ReviewResult as JSON here")
    r.add_argument("--eval-mode", action="store_true", help="pinned evaluation model (D4)")
    r.add_argument("--debug", action="store_true", help="tracebacks and debug logging")
    return parser


def run_graph_command(args, ctx: ReviewContext, title: str, body: str, config) -> int:
    """``--agent graph``: run the review graph, print its report, write the final state."""
    import uuid

    from langsmith.utils import tracing_is_enabled

    from reviewer.graph import GraphRuntime, make_review_key, run_graph

    state = {
        "review_key": make_review_key(ctx.diff_text),
        "pr_title": title,
        "pr_body": body,
        "workspace": str(args.workspace),
        "diff_path": str(args.diff),
        "sandbox_repo": args.sandbox_repo,
        "security_mode": args.security_mode,
        "spawn_enabled": config.spawn.enabled,
        "judge": args.judge or "verifier",
        "evaluation": args.eval_mode,
    }
    run_id = uuid.uuid4()
    final = run_graph(state, GraphRuntime(config=config), config={"run_id": run_id})
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(
            (json.dumps(final, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        )
    print(format_graph_report(final))
    if tracing_is_enabled():
        print(f"trace:   LangSmith root run review-graph {run_id}")
    review = final.get("review_result", {})
    step = final.get("judge_result") or {}
    for r in (step, review):
        if r.get("status") == "quota_paused" and r.get("reset_at"):
            print(f"quota paused: {r.get('model_id')} resets at {r['reset_at']}")
            break
    if final["status"] == "error":
        err = step.get("error") or review.get("error")
        cls = step.get("error_class") or review.get("error_class")
        print(f"error: {cls}: {err}".splitlines()[0], file=sys.stderr)
    if args.out:
        print(f"wrote {args.out}")
    return EXIT_CODES[final["status"]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.item and (args.workspace or args.diff or args.title or args.body_file):
        parser.error("--item cannot be combined with --workspace, --diff, --title or --body-file")
    if not args.item and not (args.workspace and args.diff):
        parser.error("give --item, or both --workspace and --diff")
    if args.item and args.agent != "single":
        parser.error(
            "specialists and the orchestrator are not run on dataset items before the stage 5 "
            "evaluation"
        )
    if args.focus and args.agent in ("single", ORCHESTRATOR, GRAPH):
        parser.error("--focus is for the specialists, not the single agent or the orchestrator")
    if args.judge and args.agent != GRAPH:
        parser.error("--judge is for --agent graph")
    if args.agent == "single" and (args.no_spawn or args.subagent_concurrency is not None):
        parser.error(
            "--no-spawn and --subagent-concurrency are for the orchestrator and specialists"
        )
    if args.subagent_concurrency is not None and args.subagent_concurrency < 1:
        parser.error("--subagent-concurrency must be at least 1")
    if args.security_mode and args.agent not in (ORCHESTRATOR, GRAPH, "security_reviewer"):
        parser.error("--security-mode is for --agent orchestrator, graph or security_reviewer")
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
        config = load_review_config()
        if args.security_mode:
            config = config.with_security_mode(args.security_mode)
        if args.no_spawn or args.subagent_concurrency is not None:
            config = config.with_spawn(
                enabled=False if args.no_spawn else None,
                max_concurrency=args.subagent_concurrency,
            )
        if args.agent == GRAPH:
            return run_graph_command(args, ctx, title, body, config)
        if args.agent == "single":
            result = review(ctx, title, body, router, item_id=args.item)
        elif args.agent == ORCHESTRATOR:
            result = orchestrate(ctx, title, body, router, item_id=args.item, config=config)
        elif args.agent == "security_reviewer" and args.security_mode == "remote":
            result = run_security_remote(
                ctx,
                pr_title=title,
                pr_body=body,
                focus=args.focus,
                evaluation=args.eval_mode,
                config=config,
                item_id=args.item,
            )
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
                config=config,
            )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes((result.model_dump_json(indent=2) + "\n").encode("utf-8"))
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        print(f"error: {type(e).__name__}: {e}".splitlines()[0], file=sys.stderr)
        return 1

    print(
        format_orchestrator_report(result)
        if isinstance(result, OrchestratorResult)
        else format_report(result)
    )
    if result.status == "quota_paused" and result.reset_at is not None:
        print(f"quota paused: {result.model_id} resets at {format_reset(result.reset_at)}")
    elif result.status == "error":
        print(f"error: {result.error_class}: {result.error}".splitlines()[0], file=sys.stderr)
    if args.out:
        print(f"wrote {args.out}")
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
