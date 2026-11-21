"""Live parity of the security specialist over A2A against the local one (stage 3, decision 0011).

    uv run python scripts/run_a2a_security.py          # the service must be running
    uv run python scripts/check_a2a_parity.py --probe  # 1 model call: is the API answering?
    uv run python scripts/check_a2a_parity.py [--fixtures dev_security_pr ...] [--max-calls N]

For each dev fixture (``dev_security_pr``, ``dev_risky_pr``, ``dev_pr``) it runs the security
specialist locally twice and remotely once with the same fixed ``focus``, in evaluation mode
(``gemini-3.1-flash-lite``), and writes all three runs to ``docs/a2a-parity.json``. This spends
model calls (free-tier quota): do not loop it.

Pass rule (decision 0011): findings are compared as sets of ``(file, category, overlapping line
ranges)``. The difference between two runs is the number of findings in either run that have no
partner in the other (a maximum matching, each finding used once). For each fixture the remote
run passes when it matches one of the two local runs exactly, or when its difference from the
nearest local run is no larger than the difference between the two local runs: remote must not
differ from local more than local differs from itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # ``tests`` is a top-level package beside ``src``
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "docs" / "a2a-parity.json"
# One fixed focus per fixture, in the style the orchestrator gives (files or functions, and why).
FOCUS = {
    "dev_security_pr": "The new search query in vault/db.py and the admin login in vault/admin.py.",
    "dev_risky_pr": "The SQL in ledger/store.py and the sync URL it builds.",
    "dev_pr": "The coupon handling in shop/cart.py and shop/coupons.py.",
}
FIXTURES = tuple(FOCUS)
PROBE_WAIT_S = 600  # a 503 means wait 10 minutes ...
PROBE_TRIES = 3  # ... at most 3 times, then stop


# --- The comparison rule ---------------------------------------------------------------------


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same file, same category, line ranges that overlap."""
    return (
        a["file"] == b["file"]
        and a["category"] == b["category"]
        and a["line_start"] <= b["line_end"]
        and b["line_start"] <= a["line_end"]
    )


def matching_size(xs: list[dict[str, Any]], ys: list[dict[str, Any]]) -> int:
    """Size of a maximum matching between ``xs`` and ``ys`` (augmenting paths)."""
    owner: dict[int, int] = {}  # index in ys -> index in xs

    def augment(i: int, seen: set[int]) -> bool:
        for j, y in enumerate(ys):
            if j in seen or not overlaps(xs[i], y):
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(xs)))


def difference(xs: list[dict[str, Any]], ys: list[dict[str, Any]]) -> int:
    """Findings without a partner on the other side, both sides added up."""
    m = matching_size(xs, ys)
    return (len(xs) - m) + (len(ys) - m)


def parity(local_a: list, local_b: list, remote: list) -> dict[str, Any]:
    """The pass rule for one fixture."""
    d_aa = difference(local_a, local_b)
    d_ra, d_rb = difference(remote, local_a), difference(remote, local_b)
    nearest = min(d_ra, d_rb)
    exact = nearest == 0
    return {
        "local_vs_local": d_aa,
        "remote_vs_local_1": d_ra,
        "remote_vs_local_2": d_rb,
        "nearest": nearest,
        "matches_a_local_exactly": exact,
        "passed": exact or nearest <= d_aa,
    }


# --- Running ---------------------------------------------------------------------------------


def summarize(response: dict[str, Any], seconds: float) -> dict[str, Any]:
    counts = response["counts"]
    return {
        "status": response["status"],
        "error": response["error"],
        "model_id": response["model_id"],
        "prompt_version": response["prompt_version"],
        "findings": [
            {
                k: f[k]
                for k in (
                    "file",
                    "category",
                    "line_start",
                    "line_end",
                    "severity",
                    "confidence",
                    "title",
                )
            }
            for f in response["findings"]
        ],  # fmt: skip
        "dropped": [d["reason"] for d in response["dropped"]],
        "model_calls": counts["model_calls"],
        "tool_calls": counts["tool_calls"],
        "input_tokens": counts["input_tokens"],
        "output_tokens": counts["output_tokens"],
        "net_duration_s": counts["net_duration_s"],
        "wait_s": counts["wait_s"],
        "duration_s": counts["duration_s"],
        "retries_by_reason": response["retries_by_reason"],
        "wall_s": round(seconds, 1),
        "service": response.get("service"),
        "transport": response.get("transport"),
    }


def probe() -> int:
    """One model call to the evaluation model; waits 10 minutes on a 503, three times at most."""
    from reviewer.models import ErrorKind, QuotaRouter, classify

    router = QuotaRouter(evaluation=True)
    model_id = router.config.evaluation.model
    for attempt in range(1, PROBE_TRIES + 1):
        try:
            router.build(model_id).invoke("Reply with the single word OK.")
        except Exception as e:
            kind = classify(e)
            print(f"probe {attempt}: {type(e).__name__} ({kind.name}): {str(e)[:200]}")
            if kind is ErrorKind.SERVER and attempt < PROBE_TRIES:
                print(f"waiting {PROBE_WAIT_S} s")
                time.sleep(PROBE_WAIT_S)
                continue
            return 1
        print(f"probe ok on {model_id} (1 model call)")
        return 0
    return 1


def run_fixture(name: str, base_url: str, tmp: Path, save=None) -> dict[str, Any]:
    from reviewer.a2a_client import call_remote_security, discover_assistant_id
    from reviewer.agents.specialists import run_security_review
    from tests.dev_pr import make_fixture_workspace, pr_text

    ctx = make_fixture_workspace(name, tmp)
    title, body = pr_text(name)
    focus = FOCUS[name]
    discover_assistant_id(base_url)  # fail before any model call if the service is down
    runs: dict[str, Any] = {}
    for label in ("local_1", "local_2"):
        started = time.monotonic()
        response = run_security_review(
            {
                "workspace": str(ctx.workspace),
                "diff_text": ctx.diff_text,
                "pr_title": title,
                "pr_body": body,
                "focus": focus,
                "evaluation": True,
            }
        )
        runs[label] = summarize(response, time.monotonic() - started)
        if save:
            save(runs)  # a later failure must not lose the calls already spent
        print(f"  {label}: {runs[label]['status']}, {len(runs[label]['findings'])} finding(s), "
              f"{runs[label]['model_calls']} model calls")  # fmt: skip
    started = time.monotonic()
    response = call_remote_security(
        ctx, pr_title=title, pr_body=body, focus=focus, evaluation=True, base_url=base_url
    )
    runs["remote"] = summarize(response, time.monotonic() - started)
    if save:
        save(runs)
    print(f"  remote: {runs['remote']['status']}, {len(runs['remote']['findings'])} finding(s), "
          f"{runs['remote']['model_calls']} model calls")  # fmt: skip
    verdict = parity(runs["local_1"]["findings"], runs["local_2"]["findings"],
                     runs["remote"]["findings"])  # fmt: skip
    all_complete = all(r["status"] == "complete" for r in runs.values())
    return {"focus": focus, "runs": runs, "parity": verdict, "all_complete": all_complete}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true", help="one model call, then exit")
    parser.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    parser.add_argument("--base-url", default="http://127.0.0.1:2025")
    parser.add_argument("--max-calls", type=int, default=60, help="stop before a fixture if the "
                        "model calls so far reach this")  # fmt: skip
    args = parser.parse_args(argv)
    load_dotenv(REPO / ".env")
    if args.probe:
        return probe()
    results: dict[str, Any] = {}
    if OUT.is_file():
        results = json.loads(OUT.read_bytes().decode("utf-8"))
    used = sum(r["model_calls"] for f in results.values() for r in f.get("runs", {}).values())
    for name in args.fixtures:
        if used >= args.max_calls:
            print(f"stopping: {used} model calls used (--max-calls {args.max_calls})")
            break
        print(f"{name}: focus {FOCUS[name]!r}")

        def save(runs, name=name):
            partial = {**results, name: {"focus": FOCUS[name], "runs": dict(runs), "partial": True}}
            OUT.write_bytes((json.dumps(partial, indent=2) + "\n").encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            results[name] = run_fixture(name, args.base_url, Path(tmp), save)
        used = sum(r["model_calls"] for f in results.values() for r in f["runs"].values())
        v = results[name]["parity"]
        print(f"  parity: {v} -> {'PASS' if v['passed'] else 'FAIL'}; model calls so far {used}")
        OUT.write_bytes((json.dumps(results, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {OUT}")
    ok = all(f.get("parity", {}).get("passed") and f.get("all_complete") for f in results.values())
    return 0 if ok and results else 1


if __name__ == "__main__":
    sys.exit(main())
