"""eval/dataset_stats.py counting functions and the target check (no Docker, no model calls)."""

import json

from eval.dataset_stats import (
    BugInfo,
    ItemInfo,
    bucket_by_type,
    check_targets,
    cross_file_bugs,
    items_per_repo,
    kind_counts,
    load_items,
    load_notes,
    multi_line_bugs,
    session_of,
)


def bug(bug_id, kind, file="a.py", start=10, end=10, cross=None):
    return BugInfo(bug_id, kind, file, start, end, cross)


def item(item_id, repo, bucket, bugs=(), py_files=2):
    return ItemInfo(item_id, repo, bucket, py_files, 50, tuple(bugs))


ITEMS = [
    item("item_001", "click", "small", [bug("b1", "off_by_one")]),
    item("item_002", "click", "small"),
    item(
        "item_006",
        "alembic",
        "large",
        [bug("b1", "off_by_one", "a.py", 5, 6), bug("b2", "sql_injection", "b.py", 9, 9, "c.py")],
        py_files=9,
    ),
    item("item_009", "alembic", "large", py_files=9),
]


def test_session_of():
    assert session_of("item_003") == "0D"
    assert session_of("item_006") == "E1"
    assert session_of("item_014") == "E2"
    assert session_of("item_099") is None


def test_bucket_by_type():
    assert bucket_by_type(ITEMS) == {
        ("large", "clean"): 1,
        ("large", "planted"): 1,
        ("small", "clean"): 1,
        ("small", "planted"): 1,
    }


def test_kind_counts_overall_and_per_session():
    assert kind_counts(ITEMS) == {"off_by_one": 2, "sql_injection": 1}
    assert kind_counts(ITEMS, "0D") == {"off_by_one": 1}
    assert kind_counts(ITEMS, "E1") == {"off_by_one": 1, "sql_injection": 1}
    assert kind_counts(ITEMS, "E2") == {}


def test_items_per_repo():
    assert items_per_repo(ITEMS) == {"alembic": 2, "click": 2}


def test_multi_line_and_cross_file():
    assert multi_line_bugs(ITEMS) == [("item_006", "b1")]
    assert cross_file_bugs(ITEMS) == [("item_006", "b2")]
    assert bug("b", "off_by_one", start=3, end=3).multi_line is False


def test_check_targets_accepts_matching_items_and_flags_mismatches():
    ok = [
        item("item_001", "click", "small", [bug("b1", "off_by_one")]),
        item("item_002", "click", "small"),
    ]
    assert check_targets(ok) == []
    wrong = [item("item_002", "click", "small", [bug("b1", "off_by_one")])]
    assert any("item_002" in p and "target" in p for p in check_targets(wrong))
    unknown = [item("item_099", "click", "small")]
    assert any("not in the target table" in p for p in check_targets(unknown))


def test_check_targets_needs_two_files_for_two_bugs_and_limits_repos():
    two_in_one = [
        item("item_003", "x", "medium", [bug("b1", "a", "f.py"), bug("b2", "b", "f.py")], 4)
    ]
    problems = check_targets(two_in_one)
    assert any("only 1 file" in p for p in problems)
    many = [item(f"item_{n:03d}", "click", "small") for n in (2, 9, 12)] * 3
    assert any("click has 9 items" in p for p in check_targets(many))


def test_load_items_and_notes(tmp_path):
    ds = tmp_path / "dataset" / "item_006"
    ds.mkdir(parents=True)
    gt = {
        "item_id": "item_006",
        "bugs": [
            {"bug_id": "b1", "kind": "off_by_one", "file": "a.py", "line_start": 4, "line_end": 5},
        ],
    }
    meta = {"repo": "alembic", "bucket": "large", "py_files_changed": 9, "lines_changed": 120}
    (ds / "ground_truth.json").write_bytes(json.dumps(gt).encode())
    (ds / "meta.json").write_bytes(json.dumps(meta).encode())
    notes = tmp_path / "notes.yaml"
    notes.write_bytes(b"cross_file:\n  item_006/b1: other.py\n")
    loaded = load_items(tmp_path / "dataset", load_notes(notes))
    assert loaded[0].bugs[0].cross_file == "other.py"
    assert loaded[0].bugs[0].multi_line is True
    assert loaded[0].type == "planted"
    assert load_notes(tmp_path / "missing.yaml") == {}
