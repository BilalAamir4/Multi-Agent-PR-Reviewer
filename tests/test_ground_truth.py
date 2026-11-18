import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.ground_truth import (
    DatasetError,
    GroundTruth,
    PlantedBug,
    default_accepted_kinds,
    load_item,
    size_bucket,
)

FIXTURES = Path(__file__).parent / "fixtures"
ITEM_001 = FIXTURES / "dataset" / "item_001"

BUG = {
    "bug_id": "b1",
    "file": "pkg/calc.py",
    "line_start": 11,
    "line_end": 12,
    "kind": "missing_none_check",
    "accepted_kinds": ["missing_none_check", "unhandled_exception"],
    "description": "No None check.",
}
GT = {
    "item_id": "item_001",
    "source_repo": "owner/name",
    "base_commit": "0123456789abcdef0123456789abcdef01234567",
    "is_clean": False,
    "bugs": [BUG],
}


def bug(**kw):
    return PlantedBug(**{**BUG, **kw})


def gt(**kw):
    return GroundTruth(**{**GT, **kw})


# --- load_item -------------------------------------------------------------------------------


def test_valid_item_loads():
    truth, diff = load_item(ITEM_001)
    assert truth.item_id == "item_001"
    assert [b.bug_id for b in truth.bugs] == ["b1", "b2"]
    assert diff.startswith("diff --git a/pkg/calc.py")
    assert "\r" not in diff


def test_diff_line_endings_preserved(tmp_path):
    item = tmp_path / "item_001"
    shutil.copytree(ITEM_001, item)
    raw = b"diff --git a/x.py b/x.py\r\n"
    (item / "pr.diff").write_bytes(raw)
    _, diff = load_item(item)
    assert diff == raw.decode("utf-8")


@pytest.mark.parametrize("missing", ["pr.diff", "ground_truth.json"])
def test_missing_file(tmp_path, missing):
    item = tmp_path / "item_001"
    shutil.copytree(ITEM_001, item)
    (item / missing).unlink()
    with pytest.raises(DatasetError, match=missing):
        load_item(item)


def test_item_id_must_match_folder(tmp_path):
    item = tmp_path / "item_009"
    shutil.copytree(ITEM_001, item)
    with pytest.raises(DatasetError, match="does not match folder"):
        load_item(item)


def test_invalid_json_is_dataset_error(tmp_path):
    item = tmp_path / "item_001"
    shutil.copytree(ITEM_001, item)
    data = json.loads((item / "ground_truth.json").read_text(encoding="utf-8"))
    data["unexpected"] = 1
    (item / "ground_truth.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DatasetError):
        load_item(item)


# --- PlantedBug validators ------------------------------------------------------------------


def test_valid_bug():
    assert bug().kind == "missing_none_check"


def test_bug_kind_other_rejected():
    with pytest.raises(ValidationError):
        bug(kind="other", accepted_kinds=["other"])


def test_accepted_kinds_must_include_kind():
    with pytest.raises(ValidationError):
        bug(accepted_kinds=["unhandled_exception"])


def test_accepted_kinds_no_other():
    with pytest.raises(ValidationError):
        bug(accepted_kinds=["missing_none_check", "other"])


def test_accepted_kinds_no_duplicates():
    with pytest.raises(ValidationError):
        bug(accepted_kinds=["missing_none_check", "missing_none_check"])


def test_bug_line_start_positive():
    with pytest.raises(ValidationError):
        bug(line_start=0, line_end=1)


def test_bug_line_end_before_start():
    with pytest.raises(ValidationError):
        bug(line_start=12, line_end=11)


@pytest.mark.parametrize("path", ["", "/abs.py", "C:/x.py", "a\\b.py", "../x.py"])
def test_bug_file_rules(path):
    with pytest.raises(ValidationError):
        bug(file=path)


def test_bug_unknown_field():
    with pytest.raises(ValidationError):
        PlantedBug(**BUG, severity="high")


@pytest.mark.parametrize("field", ["bug_id", "description"])
def test_bug_blank_text(field):
    with pytest.raises(ValidationError):
        bug(**{field: "  "})


# --- GroundTruth validators -----------------------------------------------------------------


def test_valid_ground_truth():
    assert gt().item_id == "item_001"


@pytest.mark.parametrize("item_id", ["item_1", "item_0001", "Item_001", "item_abc"])
def test_item_id_pattern(item_id):
    with pytest.raises(ValidationError):
        gt(item_id=item_id)


@pytest.mark.parametrize("repo", ["name", "owner/name/x", "", "owner/"])
def test_source_repo_pattern(repo):
    with pytest.raises(ValidationError):
        gt(source_repo=repo)


@pytest.mark.parametrize("sha", ["abc123", "g" * 40, "0" * 39, "0" * 41])
def test_base_commit_pattern(sha):
    with pytest.raises(ValidationError):
        gt(base_commit=sha)


def test_clean_with_bugs_rejected():
    with pytest.raises(ValidationError):
        gt(is_clean=True)


def test_clean_without_bugs_ok():
    assert gt(is_clean=True, bugs=[]).bugs == []


def test_not_clean_without_bugs_rejected():
    with pytest.raises(ValidationError):
        gt(bugs=[])


def test_more_than_three_bugs_rejected():
    bugs = [{**BUG, "bug_id": f"b{i}"} for i in range(1, 5)]
    with pytest.raises(ValidationError):
        gt(bugs=bugs)


def test_three_bugs_ok():
    bugs = [{**BUG, "bug_id": f"b{i}"} for i in range(1, 4)]
    assert len(gt(bugs=bugs).bugs) == 3


def test_duplicate_bug_ids_rejected():
    with pytest.raises(ValidationError):
        gt(bugs=[BUG, BUG])


def test_ground_truth_unknown_field():
    with pytest.raises(ValidationError):
        GroundTruth(**GT, notes="x")


# --- size_bucket -----------------------------------------------------------------------------


def _modified(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )


def _deleted(path: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "deleted file mode 100644\n"
        f"--- a/{path}\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-x = 1\n"
    )


def _diff(n_py: int, extra: str = "") -> str:
    return "".join(_modified(f"pkg/m{i}.py") for i in range(n_py)) + extra


@pytest.mark.parametrize(
    ("n", "bucket"),
    [(0, "small"), (1, "small"), (3, "small"), (4, "medium"), (7, "medium"), (8, "large")],
)
def test_size_bucket_boundaries(n, bucket):
    assert size_bucket(_diff(n)) == bucket


def test_size_bucket_ignores_deleted_and_non_py():
    diff = _diff(3, _deleted("pkg/gone.py") + _modified("docs/readme.md"))
    assert size_bucket(diff) == "small"


def test_size_bucket_counts_added_file():
    added = "diff --git a/pkg/new.py b/pkg/new.py\nnew file mode 100644\n--- /dev/null\n"
    added += "+++ b/pkg/new.py\n@@ -0,0 +1,1 @@\n+x = 1\n"
    assert size_bucket(_diff(3, added)) == "medium"


def test_fixture_bucket():
    _, diff = load_item(ITEM_001)
    assert size_bucket(diff) == "small"


# --- equivalences ----------------------------------------------------------------------------


def test_default_accepted_kinds_with_group():
    assert default_accepted_kinds("missing_none_check") == [
        "missing_none_check",
        "unhandled_exception",
    ]


def test_default_accepted_kinds_without_group():
    assert default_accepted_kinds("off_by_one") == ["off_by_one"]


def test_default_accepted_kinds_rejects_other():
    with pytest.raises(ValueError):
        default_accepted_kinds("other")
