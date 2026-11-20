"""Baseline test runs for the dataset source repositories (stage 0C, task 6).

For each repository in ``eval/repos.yaml``, all in its sandbox image (D2):

1. ``RUNS`` runs of ``test_command`` on ``snapshot_files(name, image_commit)``. Passed, failed,
   errors and skipped come from pytest's summary line; the duration is the sandbox wall time
   (container start to exit). Requirement: exit code 0, no failures or errors, and a median
   under ``MAX_MEDIAN_S`` seconds.
2. One run at the newest eligible large commit and one at the oldest eligible commit of any
   bucket (from ``eval/candidates/<name>.json``), to check the image across the range 0D uses.

Results go under ``baseline`` in ``eval/candidates/<name>.json``, and each bucket is filtered to
the range the image supports (``eligible_since`` in ``eval/repos.yaml``, if set).

Usage:
    uv run python -m eval.check_repos [name ...] [--runs 3]

Exit code 0 if every repository meets the requirements and both range checks pass, else 1.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime

from eval.find_candidates import BUCKETS, bucket_counts, candidates_path
from eval.repos import RepoSpec, get_repo, load_repos, run_repo_tests, take_snapshot
from reviewer.sandbox import SandboxResult

RUNS = 3
MAX_MEDIAN_S = 60.0

_SUMMARY_LINE = re.compile(r"\bin [\d.]+s\b")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)")
_ALIASES = {"error": "errors", "warning": "warnings"}


@dataclass
class TestCounts:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0
    warnings: int = 0
    summary: str = ""  # the summary line itself; empty if none was found


def parse_pytest_summary(stdout: str) -> TestCounts:
    """Counts from pytest's last summary line (``1 failed, 2 passed, 3 skipped in 1.2s``)."""
    counts = TestCounts()
    lines = [ln for ln in stdout.splitlines() if _SUMMARY_LINE.search(ln) and _COUNT.search(ln)]
    if not lines:
        return counts
    line = lines[-1]
    counts.summary = line.strip("= ").strip()
    for n, word in _COUNT.findall(line):
        setattr(counts, _ALIASES.get(word, word), int(n))
    return counts


@dataclass
class RunResult:
    commit: str
    exit_code: int | None
    timed_out: bool
    duration_s: float
    counts: TestCounts

    @property
    def ok(self) -> bool:
        c = self.counts
        return (
            self.exit_code == 0
            and not self.timed_out
            and c.failed == 0
            and c.errors == 0
            and c.passed > 0
        )

    def to_json(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 2)
        d["ok"] = self.ok
        return d


def _tail(res: SandboxResult, n: int = 40) -> str:
    return "\n".join((res.stdout + "\n" + res.stderr).strip().splitlines()[-n:])


def run_at(name: str, commit: str) -> tuple[RunResult, SandboxResult, list[tuple[str, str]]]:
    snap = take_snapshot(name, commit)
    res = run_repo_tests(name, snap.files)
    run = RunResult(
        commit, res.exit_code, res.timed_out, res.duration_s, parse_pytest_summary(res.stdout)
    )
    return run, res, snap.skipped


def _range_commits(cands: dict) -> tuple[dict | None, dict | None]:
    large = cands["buckets"]["large"]
    every = [c for b in BUCKETS for c in cands["buckets"][b]]
    newest_large = max(large, key=lambda c: c["date"]) if large else None
    oldest = min(every, key=lambda c: c["date"]) if every else None
    return newest_large, oldest


def filter_buckets(cands: dict, since: date | None) -> None:
    """Keep only commits on or after ``since`` (the range the image supports)."""
    if since is None:
        return
    for b in BUCKETS:
        cands["buckets"][b] = [
            c for c in cands["buckets"][b] if c["date"][:10] >= since.isoformat()
        ]


def check_repo(name: str, spec: RepoSpec, runs: int) -> bool:
    path = candidates_path(name)
    cands = json.loads(path.read_bytes())
    filter_buckets(cands, spec.eligible_since)
    cands["counts"] = bucket_counts(cands["buckets"])

    print(f"== {name} ({spec.image})")
    base_runs: list[RunResult] = []
    skipped: list[tuple[str, str]] = []
    for i in range(runs):
        run, res, skipped = run_at(name, spec.image_commit)
        base_runs.append(run)
        print(f"  run {i + 1}: exit={run.exit_code} {run.duration_s:.1f}s  {run.counts.summary}")
        if not run.ok:
            print(_tail(res))
    durations = [r.duration_s for r in base_runs]
    median = statistics.median(durations)
    base_ok = all(r.ok for r in base_runs) and median < MAX_MEDIAN_S
    print(f"  median {median:.1f}s  {'OK' if base_ok else 'NOT OK'}")

    range_checks = {}
    newest_large, oldest = _range_commits(cands)
    for label, c in (("newest_large", newest_large), ("oldest_eligible", oldest)):
        if c is None:
            range_checks[label] = None
            print(f"  {label}: no eligible commit")
            continue
        run, res, _ = run_at(name, c["sha"])
        range_checks[label] = {"date": c["date"], "subject": c["subject"], **run.to_json()}
        print(
            f"  {label} {c['sha'][:12]} {c['date'][:10]}: exit={run.exit_code} "
            f"{run.duration_s:.1f}s  {run.counts.summary}"
        )
        if not run.ok:
            print(_tail(res))
    range_ok = all(v is None or v["ok"] for v in range_checks.values())

    cands["baseline"] = {
        "checked": datetime.now(UTC).isoformat(timespec="seconds"),
        "image": spec.image,
        "image_commit": spec.image_commit,
        "test_command": spec.test_command,
        "snapshot_skipped": [list(s) for s in skipped],
        "runs": [r.to_json() for r in base_runs],
        "median_duration_s": round(median, 2),
        "meets_requirements": base_ok,
        "range_checks": range_checks,
        "supported_since": spec.eligible_since.isoformat() if spec.eligible_since else None,
    }
    path.write_bytes((json.dumps(cands, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return base_ok and range_ok


def narrow(name: str) -> str | None:
    """Binary-search the indexed commits (all buckets, by date) for the oldest one whose tests
    pass in the image, assuming old commits fail and new ones pass. Prints every run and returns
    that commit's date (``YYYY-MM-DD``) to use as ``eligible_since``, or None if none passes.
    The result is a proposal: re-run ``check_repos`` after setting it to confirm the boundary."""
    cands = json.loads(candidates_path(name).read_bytes())
    commits = sorted((c for b in BUCKETS for c in cands["buckets"][b]), key=lambda c: c["date"])
    cache: dict[int, bool] = {}

    def ok(i: int) -> bool:
        if i not in cache:
            run, _, _ = run_at(name, commits[i]["sha"])
            cache[i] = run.ok
            c = commits[i]
            print(f"  [{i}] {c['sha'][:12]} {c['date'][:10]} ok={run.ok}  {run.counts.summary}")
        return cache[i]

    print(f"== narrowing {name}: {len(commits)} commits")
    if not commits or not ok(len(commits) - 1):
        return None
    lo, hi = 0, len(commits) - 1  # invariant: commits[hi] passes
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    since = commits[hi]["date"][:10]
    print(f"  oldest passing: {commits[hi]['sha'][:12]} {since} {commits[hi]['subject']!r}")
    return since


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline sandboxed test runs per repository.")
    parser.add_argument("names", nargs="*", help="repositories (default: all in eval/repos.yaml)")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument(
        "--narrow",
        action="store_true",
        help="only propose eligible_since: binary-search for the oldest passing commit",
    )
    args = parser.parse_args(argv)
    names = args.names or list(load_repos())
    if args.narrow:
        found = {name: narrow(name) for name in names}
        print("proposed eligible_since:", found)
        return 0 if all(found.values()) else 1
    results = {name: check_repo(name, get_repo(name), args.runs) for name in names}
    print("summary:", {n: "OK" if ok else "NOT OK" for n, ok in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
