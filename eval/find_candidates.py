"""Index the commits of a cloned upstream repository that are eligible to become dataset items.

Only ``git`` runs here; no upstream code is executed on the host.

Eligible commits (stage 0C):

- non-merge commits reachable from the default branch, committed in the last ``--years`` years;
- not authored by a bot;
- subject does not match ``EXCLUDED_SUBJECT``;
- at least one non-test ``.py`` file gains lines (somewhere to plant a bug, D9a);
- at most ``MAX_LINES_CHANGED`` lines added plus deleted;
- no binary changes and no renames scored under ``MIN_RENAME_SCORE``.

Each commit is bucketed by ``eval.ground_truth.size_bucket`` on the text of
``git diff <parent> <commit>``, the same function the scorer uses.

Usage:
    uv run python -m eval.find_candidates <name> [--repo-dir eval/repos/<name>] [--years 4]
        [--since YYYY-MM-DD]

Output: ``eval/candidates/<name>.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from eval.ground_truth import changed_py_file_count, size_bucket
from eval.repos import REPOS_YAML, load_repos
from eval.sweep import SWEEP_THRESHOLD, sweep_stats
from reviewer.config import repo_root

MAX_LINES_CHANGED = 1500
MIN_RENAME_SCORE = 90
DEFAULT_YEARS = 4
BUCKETS = ("small", "medium", "large")
TEST_DIRS = frozenset({"tests", "test", "testing"})

BOT_MARKERS = ("[bot]", "dependabot", "pre-commit-ci", "renovate")
EXCLUDED_SUBJECT = re.compile(
    r"format|lint|typo|bump|release|version|merge|revert|changelog|docs?:|style|pre-commit",
    re.IGNORECASE,
)

# Fixed diff options so the text does not depend on the user's git config.
DIFF_OPTS = ("--no-color", "--no-ext-diff", "--no-textconv", "-M")
FIELD_SEP, RECORD_SEP = "\x1f", "\x1e"
LOG_FORMAT = f"%H{FIELD_SEP}%P{FIELD_SEP}%an{FIELD_SEP}%ae{FIELD_SEP}%cI{FIELD_SEP}%s{RECORD_SEP}"


@dataclass(frozen=True)
class LogEntry:
    sha: str
    parents: tuple[str, ...]
    author_name: str
    author_email: str
    date: str  # committer date, ISO 8601
    subject: str


@dataclass(frozen=True)
class NumstatEntry:
    path: str  # new path for renames
    added: int | None  # None for binary
    deleted: int | None
    old_path: str | None = None


@dataclass(frozen=True)
class StatusEntry:
    status: str  # A, M, D, T, R, C
    score: int | None  # similarity for R/C
    path: str
    old_path: str | None = None


@dataclass
class Candidate:
    sha: str
    parent: str
    date: str
    subject: str
    py_files_changed: int
    lines_changed: int
    test_files_changed: bool
    sweep: bool  # mostly typing, imports, decorators, strings or whitespace (eval/sweep.py)
    sweep_ratio: float


@dataclass
class ScanStats:
    scanned: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


# --- pure helpers (unit-tested on synthetic input) ---------------------------------------------


def parse_log(text: str) -> list[LogEntry]:
    """Parse ``git log --format=LOG_FORMAT`` output."""
    out = []
    for rec in text.split(RECORD_SEP):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, parents, name, email, date, subject = rec.split(FIELD_SEP)
        out.append(LogEntry(sha, tuple(parents.split()), name, email, date, subject))
    return out


def parse_numstat_z(text: str) -> list[NumstatEntry]:
    """Parse ``git diff --numstat -z -M``. Renames are ``a\\td\\t\\0old\\0new\\0``."""
    tokens = text.split("\0")
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok:
            continue
        added, deleted, path = tok.split("\t", 2)
        old = None
        if path == "":  # rename or copy: the next two tokens are old and new path
            old, path = tokens[i], tokens[i + 1]
            i += 2
        a = None if added == "-" else int(added)
        d = None if deleted == "-" else int(deleted)
        out.append(NumstatEntry(path, a, d, old))
    return out


def parse_name_status_z(text: str) -> list[StatusEntry]:
    """Parse ``git diff --name-status -z -M``."""
    tokens = text.split("\0")
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok:
            continue
        letter, score = tok[0], tok[1:]
        if letter in "RC":
            old, new = tokens[i], tokens[i + 1]
            i += 2
            out.append(StatusEntry(letter, int(score) if score else None, new, old))
        else:
            out.append(StatusEntry(letter, None, tokens[i]))
            i += 1
    return out


def is_bot(name: str, email: str) -> bool:
    text = f"{name}\n{email}".lower()
    return any(m in text for m in BOT_MARKERS)


def subject_excluded(subject: str) -> bool:
    return EXCLUDED_SUBJECT.search(subject) is not None


def is_test_path(path: str) -> bool:
    """A directory segment exactly ``tests``, ``test`` or ``testing``, or a file named
    ``test_*.py``, ``*_test.py`` or ``conftest.py``. Package modules such as
    ``click/testing.py`` are not tests."""
    p = PurePosixPath(path)
    if any(seg in TEST_DIRS for seg in p.parts[:-1]):
        return True
    name = p.name
    return (
        (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def is_py(path: str) -> bool:
    return PurePosixPath(path).suffix == ".py"


def reject_reason_meta(entry: LogEntry) -> str | None:
    """Checks that need only ``git log``."""
    if len(entry.parents) != 1:
        return "merge_or_root"
    if is_bot(entry.author_name, entry.author_email):
        return "bot_author"
    if subject_excluded(entry.subject):
        return "excluded_subject"
    return None


def reject_reason_files(numstat: list[NumstatEntry], status: list[StatusEntry]) -> str | None:
    """Checks on the file list of one commit."""
    if any(n.added is None or n.deleted is None for n in numstat):
        return "binary_change"
    if any(s.status == "R" and (s.score or 0) < MIN_RENAME_SCORE for s in status):
        return "low_score_rename"
    if not any(is_non_test_py(n.path) and (n.added or 0) > 0 for n in numstat):
        return "no_non_test_py_change"
    if sum((n.added or 0) + (n.deleted or 0) for n in numstat) > MAX_LINES_CHANGED:
        return "too_many_lines"
    return None


def is_non_test_py(path: str) -> bool:
    return is_py(path) and not is_test_path(path)


def make_candidate(entry: LogEntry, numstat: list[NumstatEntry], diff_text: str) -> Candidate:
    sweep = sweep_stats(diff_text, is_non_test_py)
    return Candidate(
        sha=entry.sha,
        parent=entry.parents[0],
        date=entry.date,
        subject=entry.subject,
        py_files_changed=changed_py_file_count(diff_text),
        lines_changed=sum((n.added or 0) + (n.deleted or 0) for n in numstat),
        test_files_changed=any(is_test_path(n.path) and is_py(n.path) for n in numstat),
        sweep=sweep.is_sweep,
        sweep_ratio=round(sweep.ratio, 3),
    )


def bucket_candidates(items: Iterable[tuple[Candidate, str]]) -> dict[str, list[Candidate]]:
    """Group (candidate, diff text) pairs by ``size_bucket``; newest first within each."""
    out: dict[str, list[Candidate]] = {b: [] for b in BUCKETS}
    for cand, diff_text in items:
        out[size_bucket(diff_text)].append(cand)
    for lst in out.values():
        lst.sort(key=lambda c: c.date, reverse=True)
    return out


# --- git ---------------------------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return res.stdout.decode("utf-8", "replace")


def default_branch(repo: Path) -> str:
    ref = git(repo, "symbolic-ref", "--short", "refs/remotes/origin/HEAD").strip()
    return ref.removeprefix("origin/")


def diff_text(repo: Path, parent: str, sha: str) -> str:
    return git(repo, "diff", *DIFF_OPTS, parent, sha)


def scan(repo: Path, branch: str, since: str) -> tuple[dict[str, list[Candidate]], ScanStats]:
    log = git(repo, "log", "--no-merges", f"--since={since}", f"--format={LOG_FORMAT}",
              f"refs/remotes/origin/{branch}")  # fmt: skip
    stats = ScanStats()
    pairs: list[tuple[Candidate, str]] = []
    for entry in parse_log(log):
        stats.scanned += 1
        reason = reject_reason_meta(entry)
        if reason is None:
            parent = entry.parents[0]
            numstat = parse_numstat_z(git(repo, "diff", *DIFF_OPTS, "--numstat", "-z", parent,
                                          entry.sha))  # fmt: skip
            status = parse_name_status_z(git(repo, "diff", *DIFF_OPTS, "--name-status", "-z",
                                             parent, entry.sha))  # fmt: skip
            reason = reject_reason_files(numstat, status)
        if reason is not None:
            stats.reject(reason)
            continue
        text = diff_text(repo, entry.parents[0], entry.sha)
        pairs.append((make_candidate(entry, numstat, text), text))
    return bucket_candidates(pairs), stats


def years_before(when: datetime, years: int) -> datetime:
    """Same calendar date ``years`` earlier (Feb 29 becomes Feb 28)."""
    try:
        return when.replace(year=when.year - years)
    except ValueError:
        return when.replace(year=when.year - years, day=28)


def bucket_counts(buckets: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Per bucket (commits as JSON dicts): all, sweep, non-sweep and with test changes."""
    return {
        b: {
            "all": len(lst),
            "sweep": sum(c["sweep"] for c in lst),
            "non_sweep": sum(not c["sweep"] for c in lst),
            "with_test_changes": sum(c["test_files_changed"] for c in lst),
        }
        for b, lst in buckets.items()
    }


def configured_since(name: str) -> str | None:
    """``eligible_since`` from ``eval/repos.yaml`` (the range the sandbox image supports)."""
    if not REPOS_YAML.is_file():
        return None
    spec = load_repos().get(name)
    return spec.eligible_since.isoformat() if spec and spec.eligible_since else None


def candidates_path(name: str) -> Path:
    return repo_root() / "eval" / "candidates" / f"{name}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index eligible commits of a cloned repository.")
    parser.add_argument("name")
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--since", default=None, help="YYYY-MM-DD; overrides --years")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    repo = args.repo_dir or repo_root() / "eval" / "repos" / args.name
    if not (repo / ".git").exists():
        print(f"not a clone: {repo}", file=sys.stderr)
        return 2
    now = datetime.now(UTC)
    since = args.since or years_before(now, args.years).date().isoformat()
    supported = configured_since(args.name)
    if supported and supported > since:
        since = supported
    branch = default_branch(repo)
    buckets, stats = scan(repo, branch, since)

    out_path = args.out or candidates_path(args.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(out_path.read_bytes()) if out_path.is_file() else {}
    data = {
        "name": args.name,
        "default_branch": branch,
        "generated": now.isoformat(timespec="seconds"),
        "since": since,
        "head": git(repo, "rev-parse", f"refs/remotes/origin/{branch}").strip(),
        "scan": {"scanned": stats.scanned, "rejected": dict(sorted(stats.rejected.items()))},
        "sweep_threshold": SWEEP_THRESHOLD,
        "counts": {},
        "buckets": {b: [asdict(c) for c in buckets[b]] for b in BUCKETS},
    }
    data["counts"] = bucket_counts(data["buckets"])
    if "baseline" in previous:  # written by eval/check_repos.py; keep it
        data["baseline"] = previous["baseline"]
    out_path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    print(f"{args.name}: branch={branch} since={since} scanned={stats.scanned}")
    print(f"  rejected: {data['scan']['rejected']}")
    for b, c in data["counts"].items():
        print(f"  {b:6}: {c}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
