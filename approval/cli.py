"""Stage 6 command line (decision 0016)::

    uv run python -m approval review-pr --repo OWNER/NAME --pr N
        [--judge verifier|orchestrator|none] [--security-mode local|remote] [--sandbox-repo NAME]
        [--eval-mode] [--key-suffix S]
    uv run python -m approval approve --key KEY [--dry-run]
    uv run python -m approval resume --key KEY
    uv run python -m approval status --key KEY

``review-pr`` runs the review graph up to the approval interrupt. ``approve`` walks through the
findings, saves progress in ``.reviewer/decisions/<key>.json`` and resumes the graph with the
decisions, which posts one GitHub review. ``GITHUB_TOKEN`` comes from ``.env``; it is never
printed (every error text is scrubbed).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from approval.decisions import DecisionError, apply_decisions, validate_decisions
from approval.github import GitHubClient, scrub
from approval.graph import (
    NODE_ORDER,
    ApprovalRuntime,
    build_approval_graph,
    thread_config,
)
from approval.posting import build_review, marker_for
from approval.store import (
    close_checkpointer,
    default_state_dir,
    load_progress,
    open_checkpointer,
    save_progress,
    workspace_dir,
)
from approval.workspace import GitRunner, checkout_head, run_git
from reviewer.config import load_review_config, repo_root
from reviewer.graph import GraphRuntime, make_review_key

JUDGES = ("verifier", "orchestrator", "none")
EXIT_CODES = {"complete": 0, "partial": 5, "quota_paused": 3, "limit_reached": 4, "error": 1}


def _client_from_env() -> GitHubClient:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set (put it in .env)")
    return GitHubClient(token)


@dataclass
class Deps:
    """The things tests replace: GitHub, the review runtime, git and the keyboard."""

    github: Callable[[], GitHubClient] = _client_from_env
    runtime: Callable[[], GraphRuntime] = GraphRuntime
    git: GitRunner = run_git
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = print
    err: Callable[[str], None] = field(default=lambda s: print(s, file=sys.stderr))


# --- helpers ---------------------------------------------------------------------------------


def _open(state_dir: Path, deps: Deps):
    saver = open_checkpointer(state_dir)
    graph = build_approval_graph(saver, deps.runtime(), ApprovalRuntime(github=deps.github))
    return graph, saver


def pending_payload(snap) -> list[dict[str, Any]] | None:
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def last_completed(snap) -> str | None:
    if not snap.next:
        return NODE_ORDER[-1]
    i = NODE_ORDER.index(snap.next[0])
    return NODE_ORDER[i - 1] if i else None


def _span(e: dict[str, Any]) -> str:
    a, b = e["line_start"], e["line_end"]
    return str(a) if a == b else f"{a}-{b}"


def show_finding(deps: Deps, i: int, n: int, e: dict[str, Any]) -> None:
    out = deps.out
    out(f"\n--- finding {i}/{n} ({e['id']}) ---")
    out(f"file:      {e['file']}:{_span(e)}")
    out(f"severity:  {e['severity']}   category: {e['category']}   confidence: {e['confidence']}")
    out(f"title:     {e['title']}")
    out(f"explain:   {e['explanation']}")
    out("evidence:")
    for line in e["evidence"].splitlines() or [""]:
        out(f"    {line}")
    why = f" ({e['verdict_reason']})" if e.get("verdict_reason") else ""
    out(f"found by:  {e['agent']}   verdict: {e['verdict']}{why}")


def _ask_edit(deps: Deps, e: dict[str, Any]) -> dict[str, str]:
    edited: dict[str, str] = {}
    for key in ("title", "explanation", "severity"):
        while True:
            new = deps.ask(f"new {key} (blank keeps it): ").strip()
            if not new or new == e[key]:
                break
            if key == "severity" and new not in ("high", "medium", "low"):
                deps.out("severity must be high, medium or low")
                continue
            edited[key] = new
            break
    return edited


def _summary(deps: Deps, payload, decisions) -> None:
    deps.out("\nsummary:")
    for e in payload:
        d = decisions[e["id"]]
        extra = f" ({', '.join(sorted(d['edited']))} edited)" if d["edited"] else ""
        deps.out(f"  {d['action']:<8} {e['file']}:{_span(e)} {e['title']}{extra}")


def _ordered(payload, decisions) -> list[dict[str, Any]]:
    return [decisions[e["id"]] for e in payload]


# --- commands --------------------------------------------------------------------------------


def cmd_review_pr(args, deps: Deps) -> int:
    owner, _, name = args.repo.partition("/")
    if not owner or not name:
        raise ValueError("--repo must be OWNER/NAME")
    state_dir: Path = args.state_dir
    client = deps.github()
    try:
        pr = client.get_pr(owner, name, args.pr)
        diff = client.get_pr_diff(owner, name, args.pr)
    finally:
        client.close()
    key = make_review_key(diff, repo=args.repo, pr=str(args.pr), head_sha=pr["head_sha"])
    if args.key_suffix:
        key = f"{key}+{args.key_suffix}"
    graph, saver = _open(state_dir, deps)
    try:
        config = thread_config(key)
        if graph.get_state(config).values:
            deps.out(f"review {key} already exists; use status, resume or approve")
            return 0
        workspace, diff_path = checkout_head(
            pr["clone_url"], pr["head_sha"], workspace_dir(state_dir, key), diff, deps.git
        )
        load_dotenv(repo_root() / ".env")
        state = {
            "review_key": key,
            "pr_title": pr["title"],
            "pr_body": pr["body"],
            "workspace": str(workspace),
            "diff_path": str(diff_path),
            "sandbox_repo": args.sandbox_repo,
            "security_mode": args.security_mode,
            "spawn_enabled": load_review_config().spawn.enabled,
            "judge": args.judge,
            "evaluation": args.eval_mode,
            "repo": args.repo,
            "pr_number": args.pr,
            "head_sha": pr["head_sha"],
            "base_sha": pr["base_sha"],
            "pr_url": pr["html_url"],
        }
        final = graph.invoke(state, config)
        snap = graph.get_state(config)
        payload = pending_payload(snap)
    finally:
        close_checkpointer(saver)
    deps.out(f"review_key: {key}")
    deps.out(f"status:     {final.get('status')}")
    for note in final.get("notes", []):
        deps.out(f"note:       {note}")
    if payload is None:
        deps.out("nothing to approve (no findings, or the review did not finish)")
        deps.out(f"posted:     {final.get('posted')}")
        return EXIT_CODES.get(final.get("status", "error"), 1)
    deps.out(f"findings:   {len(payload)} waiting for approval")
    deps.out(f'next:       uv run python -m approval approve --key "{key}"')
    return 0


def cmd_status(args, deps: Deps) -> int:
    graph, saver = _open(args.state_dir, deps)
    try:
        config = thread_config(args.key)
        snap = graph.get_state(config)
        if not snap.values:
            deps.err(f"no review with key {args.key}")
            return 1
        payload = pending_payload(snap)
        v = snap.values
        deps.out(f"review_key:     {args.key}")
        deps.out(f"last completed: {last_completed(snap)}")
        deps.out(f"next node:      {', '.join(snap.next) or '(finished)'}")
        deps.out(f"review status:  {v.get('status')}")
        if payload is not None:
            done = len(load_progress(args.state_dir, args.key))
            deps.out(f"interrupt:      waiting for approval of {len(payload)} finding(s)")
            deps.out(f"decisions:      {done}/{len(payload)} saved")
        else:
            deps.out("interrupt:      none")
        deps.out(f"posted:         {json.dumps(v.get('posted'))}")
    finally:
        close_checkpointer(saver)
    return 0


def cmd_resume(args, deps: Deps) -> int:
    graph, saver = _open(args.state_dir, deps)
    try:
        config = thread_config(args.key)
        snap = graph.get_state(config)
        if not snap.values:
            deps.err(f"no review with key {args.key}")
            return 1
        if pending_payload(snap) is not None:
            deps.out(
                f"review is waiting for approval; run: "
                f'uv run python -m approval approve --key "{args.key}"'
            )
            return 0
        if not snap.next:
            deps.out("review already finished; nothing to resume")
            return 0
        deps.out(f"resuming after {last_completed(snap)}; running {', '.join(snap.next)}")
        final = graph.invoke(None, config)
        snap = graph.get_state(config)
        payload = pending_payload(snap)
    finally:
        close_checkpointer(saver)
    deps.out(f"status: {final.get('status')}")
    if payload is not None:
        deps.out(f"findings: {len(payload)} waiting for approval")
        deps.out(f'next: uv run python -m approval approve --key "{args.key}"')
    return 0


def _print_plan(deps: Deps, key: str, approved, diff_path: str) -> None:
    plan = build_review(approved, Path(diff_path).read_bytes().decode("utf-8"), marker_for(key))
    deps.out("\nwould post one review (event COMMENT):")
    deps.out(plan.body)
    for c in plan.comments:
        where = f"{c['path']}:{c.get('start_line', c['line'])}-{c['line']}"
        deps.out(f"\n[inline {where}]\n{c['body']}")


def cmd_approve(args, deps: Deps) -> int:
    state_dir: Path = args.state_dir
    graph, saver = _open(state_dir, deps)
    try:
        config = thread_config(args.key)
        snap = graph.get_state(config)
        if not snap.values:
            deps.err(f"no review with key {args.key}")
            return 1
        payload = pending_payload(snap)
        if payload is None:
            return _approve_finished(args, deps, graph, config, snap)
        progress = load_progress(state_dir, args.key)
        n = len(payload)
        for i, e in enumerate(payload, 1):
            if e["id"] in progress:
                continue
            show_finding(deps, i, n, e)
            while True:
                answer = deps.ask("[a]pprove / [e]dit / [r]eject / [s]kip for now / [q]uit: ")
                answer = answer.strip().lower()[:1]
                if answer in ("a", "r", "e", "s", "q"):
                    break
            if answer == "q":
                deps.out("progress saved; run approve again to continue")
                return 0
            if answer == "s":
                continue
            if answer == "e":
                edited = _ask_edit(deps, e)
                if not edited:
                    deps.out("nothing changed; skipping for now")
                    continue
                progress[e["id"]] = {"finding_id": e["id"], "action": "edit", "edited": edited}
            else:
                action = "approve" if answer == "a" else "reject"
                progress[e["id"]] = {"finding_id": e["id"], "action": action, "edited": {}}
            save_progress(state_dir, args.key, progress)
        missing = [e["id"] for e in payload if e["id"] not in progress]
        if missing:
            deps.out(f"{len(missing)} finding(s) undecided; run approve again to continue")
            return 0
        _summary(deps, payload, progress)
        decisions = validate_decisions(_ordered(payload, progress), payload)
        approved = apply_decisions(decisions, payload)
        if args.dry_run:
            _print_plan(deps, args.key, approved, snap.values["diff_path"])
            deps.out("\n(dry run: nothing was sent to GitHub, the review was not resumed)")
            return 0
        if deps.ask("post these decisions to GitHub? [y/N]: ").strip().lower() != "y":
            deps.out("not posted; run approve again when ready")
            return 0
        final = graph.invoke(Command(resume=decisions), config)
    finally:
        close_checkpointer(saver)
    return _report_posted(deps, final.get("posted"))


def _report_posted(deps: Deps, posted: dict[str, Any] | None) -> int:
    posted = posted or {}
    if "skipped" in posted:
        deps.out(f"nothing posted: {posted['skipped']}")
    elif posted.get("existed"):
        deps.out(f"already posted (marker found), not posted again: {posted.get('url')}")
    else:
        deps.out(
            f"posted review {posted.get('review_id')}: {posted.get('url')} "
            f"({len(posted.get('inline', []))} inline, {len(posted.get('in_summary', []))} "
            "in the summary)"
        )
    return 0


def _approve_finished(args, deps: Deps, graph, config, snap) -> int:
    """The review already ran through ``post``. ``--dry-run`` shows what would be posted; without
    it, ``post`` runs again from its checkpoint (it finds its marker and posts nothing twice)."""
    v = snap.values
    if "approved_findings" not in v:
        deps.err("this review is not at the approval step; see status or resume")
        return 1
    if args.dry_run:
        if not v["approved_findings"]:
            deps.out("nothing would be posted: no approved findings")
        else:
            _print_plan(deps, args.key, v["approved_findings"], v["diff_path"])
            deps.out("\n(dry run: nothing was sent to GitHub)")
        return 0
    redo = next((s for s in graph.get_state_history(config) if s.next == ("post",)), None)
    if redo is None:
        deps.err("no checkpoint before the post node")
        return 1
    final = graph.invoke(None, redo.config)
    return _report_posted(deps, final.get("posted"))


# --- entry point -----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="approval", description=__doc__.split("\n")[0])
    parser.add_argument("--state-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review-pr", help="review a pull request up to the approval step")
    r.add_argument("--repo", required=True, help="OWNER/NAME")
    r.add_argument("--pr", type=int, required=True)
    r.add_argument("--judge", choices=JUDGES, default="verifier")
    r.add_argument("--security-mode", choices=("local", "remote"))
    r.add_argument("--sandbox-repo")
    r.add_argument("--eval-mode", action="store_true")
    r.add_argument("--key-suffix", help="makes a separate review of the same PR (for demos)")
    a = sub.add_parser("approve", help="decide on each finding, then post the approved ones")
    a.add_argument("--key", required=True)
    a.add_argument("--dry-run", action="store_true", help="show what would be posted; no GitHub")
    for name, text in (("resume", "continue a review after a crash"), ("status", "show progress")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--key", required=True)
    return parser


def main(argv: Sequence[str] | None = None, deps: Deps | None = None) -> int:
    args = _build_parser().parse_args(argv)
    deps = deps or Deps()
    args.state_dir = args.state_dir or default_state_dir()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    load_dotenv(repo_root() / ".env")
    handlers = {
        "review-pr": cmd_review_pr,
        "approve": cmd_approve,
        "resume": cmd_resume,
        "status": cmd_status,
    }
    try:
        return handlers[args.command](args, deps)
    except (Exception, KeyboardInterrupt) as e:
        token = os.environ.get("GITHUB_TOKEN")
        if args.debug:
            print(scrub(traceback.format_exc(), token), file=sys.stderr)
        if isinstance(e, KeyboardInterrupt):
            deps.err("interrupted; continue with: resume --key <key>")
            return 130
        kind = "decision error" if isinstance(e, DecisionError) else type(e).__name__
        first = scrub(e, token).splitlines()[0] if str(e) else ""
        deps.err(f"error: {kind}: {first}")
        return 1
