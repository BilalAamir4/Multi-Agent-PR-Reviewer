"""Commit filter and bucketing in eval/find_candidates.py, on synthetic git output."""

import pytest

from eval import find_candidates as fc
from eval.find_candidates import (
    FIELD_SEP,
    RECORD_SEP,
    LogEntry,
    NumstatEntry,
    StatusEntry,
    bucket_candidates,
    is_bot,
    is_test_path,
    make_candidate,
    parse_log,
    parse_name_status_z,
    parse_numstat_z,
    reject_reason_files,
    reject_reason_meta,
    subject_excluded,
)

SHA, PARENT = "a" * 40, "b" * 40


def log_record(name="Ann", email="ann@example.com", subject="Fix parser", parents=PARENT):
    fields = [SHA, parents, name, email, "2026-01-02T03:04:05+00:00", subject]
    return FIELD_SEP.join(fields) + RECORD_SEP


def entry(**kw) -> LogEntry:
    return parse_log(log_record(**kw))[0]


def py_diff(n_files: int, prefix: str = "pkg/m") -> str:
    parts = []
    for i in range(n_files):
        path = f"{prefix}{i}.py"
        parts.append(
            f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
        )
    return "".join(parts)


# --- parsing ---------------------------------------------------------------------------------


def test_parse_log():
    text = "\n" + log_record() + "\n" + log_record(subject="Two", parents=f"{PARENT} {'c' * 40}")
    a, b = parse_log(text)
    assert a.sha == SHA and a.parents == (PARENT,) and a.subject == "Fix parser"
    assert len(b.parents) == 2


def test_parse_numstat_z_plain_binary_and_rename():
    text = "3\t1\tpkg/a.py\0-\t-\tlogo.png\0" + "2\t2\t\0pkg/old.py\0pkg/new.py\0"
    a, img, ren = parse_numstat_z(text)
    assert a == NumstatEntry("pkg/a.py", 3, 1)
    assert img.added is None and img.deleted is None
    assert ren == NumstatEntry("pkg/new.py", 2, 2, "pkg/old.py")


def test_parse_name_status_z():
    text = "M\0pkg/a.py\0R087\0pkg/old.py\0pkg/new.py\0A\0tests/test_x.py\0"
    m, r, a = parse_name_status_z(text)
    assert m == StatusEntry("M", None, "pkg/a.py")
    assert r == StatusEntry("R", 87, "pkg/new.py", "pkg/old.py")
    assert a.status == "A"


# --- metadata filter -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "email"),
    [
        ("dependabot[bot]", "49699333+dependabot[bot]@users.noreply.github.com"),
        ("pre-commit-ci[bot]", "66853113+pre-commit-ci[bot]@users.noreply.github.com"),
        ("Renovate Bot", "bot@renovateapp.com"),
        ("Someone", "dependabot@example.com"),
    ],
)
def test_bot_authors_rejected(name, email):
    assert is_bot(name, email)
    assert reject_reason_meta(entry(name=name, email=email)) == "bot_author"


def test_human_author_kept():
    assert not is_bot("Ann Robot", "ann@example.com")
    assert reject_reason_meta(entry()) is None


@pytest.mark.parametrize(
    "subject",
    [
        "Format with black",
        "Fix typo in error",
        "Bump version to 1.2",
        "Release 4.0",
        "Merge branch main",
        'Revert "Add x"',
        "Update CHANGELOG",
        "docs: explain options",
        "Doc: explain options",
        "Code style",
        "pre-commit autoupdate",
        "Lint fixes",
        "conversion fix",  # contains "version": the rule is a plain substring match
    ],
)
def test_excluded_subjects(subject):
    assert subject_excluded(subject)
    assert reject_reason_meta(entry(subject=subject)) == "excluded_subject"


@pytest.mark.parametrize("subject", ["Fix parser crash", "Add if_exists to drop_index"])
def test_allowed_subjects(subject):
    assert not subject_excluded(subject)


def test_merge_commit_rejected():
    assert reject_reason_meta(entry(parents=f"{PARENT} {'c' * 40}")) == "merge_or_root"


# --- file filter -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "is_test"),
    [
        ("tests/test_core.py", True),
        ("tests/helpers.py", True),
        ("src/pkg/test/util.py", True),
        ("alembic/testing/fixtures.py", True),
        ("pkg/test_utils.py", True),
        ("pkg/core_test.py", True),
        ("conftest.py", True),
        ("src/click/testing.py", False),  # a package module, not a test
        ("pkg/testing_utils.py", False),
        ("pkg/tests.py", False),
        ("pkg/contest.py", False),
        ("testsuite/run.py", False),
        ("src/pkg/core.py", False),
    ],
)
def test_is_test_path(path, is_test):
    assert is_test_path(path) is is_test


def test_click_testing_module_is_not_a_test():
    assert not is_test_path("src/click/testing.py")
    assert reject_reason_files([NumstatEntry("src/click/testing.py", 5, 0)], []) is None


def test_line_limit_inclusive():
    at_limit = [NumstatEntry("pkg/a.py", 1000, 400), NumstatEntry("README.md", 50, 50)]
    over = [NumstatEntry("pkg/a.py", 1000, 401), NumstatEntry("README.md", 50, 50)]
    assert reject_reason_files(at_limit, []) is None
    assert reject_reason_files(over, []) == "too_many_lines"


def test_test_only_commit_rejected():
    files = [NumstatEntry("tests/test_a.py", 10, 2), NumstatEntry("conftest.py", 1, 0)]
    assert reject_reason_files(files, []) == "no_non_test_py_change"


def test_non_py_only_rejected():
    assert reject_reason_files([NumstatEntry("docs/a.rst", 10, 0)], []) == "no_non_test_py_change"


def test_pure_deletion_of_py_rejected():
    assert reject_reason_files([NumstatEntry("pkg/a.py", 0, 7)], []) == "no_non_test_py_change"


def test_low_score_rename_rejected():
    files = [NumstatEntry("pkg/new.py", 5, 5, "pkg/old.py")]
    low = [StatusEntry("R", 89, "pkg/new.py", "pkg/old.py")]
    ok = [StatusEntry("R", 90, "pkg/new.py", "pkg/old.py")]
    assert reject_reason_files(files, low) == "low_score_rename"
    assert reject_reason_files(files, ok) is None


def test_binary_change_rejected():
    files = [NumstatEntry("pkg/a.py", 1, 1), NumstatEntry("logo.png", None, None)]
    assert reject_reason_files(files, []) == "binary_change"


# --- bucketing -------------------------------------------------------------------------------


def test_bucketing_goes_through_size_bucket(monkeypatch):
    seen = []

    def spy(diff_text):
        seen.append(diff_text)
        return "large"

    monkeypatch.setattr(fc, "size_bucket", spy)
    cand = make_candidate(entry(), [NumstatEntry("pkg/m0.py", 1, 1)], py_diff(1))
    out = bucket_candidates([(cand, "DIFF TEXT")])
    assert seen == ["DIFF TEXT"]
    assert out["large"] == [cand] and out["small"] == [] and out["medium"] == []


@pytest.mark.parametrize(
    ("n", "bucket"), [(3, "small"), (4, "medium"), (7, "medium"), (8, "large")]
)
def test_bucket_boundaries_match_size_bucket(n, bucket):
    text = py_diff(n)
    cand = make_candidate(entry(), [NumstatEntry(f"pkg/m{i}.py", 1, 1) for i in range(n)], text)
    assert cand.py_files_changed == n
    assert bucket_candidates([(cand, text)])[bucket] == [cand]


def test_candidate_fields():
    numstat = [NumstatEntry("pkg/m0.py", 1, 1), NumstatEntry("tests/test_m.py", 4, 0)]
    cand = make_candidate(entry(), numstat, py_diff(1))
    assert cand.parent == PARENT and cand.lines_changed == 6 and cand.test_files_changed
    assert cand.sweep is False  # "x = 1" -> "x = 2" is a logic change


def test_buckets_sorted_newest_first():
    old = make_candidate(entry(), [NumstatEntry("pkg/m0.py", 1, 1)], py_diff(1))
    new = make_candidate(entry(), [NumstatEntry("pkg/m0.py", 1, 1)], py_diff(1))
    old.date, new.date = "2023-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"
    assert bucket_candidates([(old, py_diff(1)), (new, py_diff(1))])["small"] == [new, old]
