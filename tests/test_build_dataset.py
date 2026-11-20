"""eval/build_dataset.py: find/replace, line ranges and the git build, on synthetic repos."""

import pytest
from pydantic import ValidationError

from eval.build_dataset import (
    PlantError,
    Recipe,
    accepted_kinds_for,
    apply_plants,
    enclosing_test,
    line_of,
    locate_find,
    split_lines,
)
from tests.dataset_helpers import (
    LIB_AT_COMMIT,
    TEST_AT_COMMIT,
    bug,
    build,
    git,
    make_repo,
    recipe,
)

FILES = {"lib.py": LIB_AT_COMMIT.encode(), "test_lib.py": TEST_AT_COMMIT.encode()}


# --- pure helpers ----------------------------------------------------------------------------


def test_split_lines_keeps_endings_and_splits_on_newline_only():
    assert split_lines("a\nb\r\nc") == ["a\n", "b\r\n", "c"]
    assert split_lines("a\n") == ["a\n"]
    assert split_lines("x\x0cy\n") == ["x\x0cy\n"]  # form feed is not a line break for git
    assert split_lines("") == []


def test_line_of():
    text = "a\nbb\nccc\n"
    assert [line_of(text, i) for i in (0, 1, 2, 4, 5, 8)] == [1, 1, 2, 2, 3, 3]


def test_locate_find_spans_lines():
    span = locate_find(LIB_AT_COMMIT, "if value < low:\n        return low\n", "b1", "lib.py")
    assert (span.first_line, span.last_line) == (8, 9)
    mid = locate_find(LIB_AT_COMMIT, "value < low", "b1", "lib.py")
    assert (mid.first_line, mid.last_line) == (8, 8)


# --- find/replace ----------------------------------------------------------------------------


def test_single_line_plant_range():
    res = apply_plants(FILES, [bug()])
    assert res.ranges == {"b1": (8, 8)}
    assert "if value <= low:" in res.planted["lib.py"]
    assert res.original["lib.py"] == LIB_AT_COMMIT


def test_find_must_occur_exactly_once():
    with pytest.raises(PlantError, match=r"b1: find text occurs 0 times"):
        apply_plants(FILES, [bug(find="not there\n")])
    with pytest.raises(PlantError, match=r"b2: find text occurs 4 times"):
        apply_plants(FILES, [bug(bug_id="b2", find="return ")])


def test_find_counted_again_after_earlier_plants():
    # b1 creates a second copy of b2's find text
    b1 = bug(find="return min(value, high)\n", replace="return min(value, high)  # x\n")
    b2 = bug(bug_id="b2", find="return min(value, high)  #", replace="return max(value, high)  #")
    with pytest.raises(PlantError, match="b2: find text occurs 0 times"):
        apply_plants(FILES, [b1, b2])


def test_overlapping_finds_rejected():
    b1 = bug(
        find="if value < low:\n        return low\n",
        replace="if value <= low:\n        return low\n",
    )
    b2 = bug(bug_id="b2", find="return low\n", replace="return high\n")
    with pytest.raises(PlantError, match="overlaps"):
        apply_plants(FILES, [b1, b2])


def test_file_must_be_changed_by_the_commit():
    with pytest.raises(PlantError, match="b1: other.py is not a file the commit"):
        apply_plants(FILES, [bug(file="other.py")])


def test_identical_replace_rejected():
    with pytest.raises(PlantError, match="identical"):
        apply_plants(FILES, [bug(replace="if value < low:\n")])


def test_multi_line_replace_range_covers_new_lines():
    b = bug(
        find="return min(value, high)\n", replace="result = min(value, high)\n    return result\n"
    )
    assert apply_plants(FILES, [b]).ranges == {"b1": (10, 11)}


def test_pure_deletion_is_located_at_the_next_line():
    b = bug(
        kind="missing_none_check",
        find="    if value is None:\n        return None\n    if value",
        replace="    if value",
    )
    res = apply_plants(FILES, [b])
    # lines 6-7 deleted; the line now at 6 is "if value < low:"
    assert res.ranges == {"b1": (6, 6)}
    assert split_lines(res.planted["lib.py"])[5] == "    if value < low:\n"


def test_two_bugs_in_one_file_get_their_own_ranges():
    b1 = bug(find="if value is None:\n", replace="if value is False:\n", kind="wrong_comparison")
    b2 = bug(bug_id="b2", find="return min(value, high)\n", replace="return min(value, high + 1)\n")
    res = apply_plants(FILES, [b1, b2])
    assert res.ranges == {"b1": (6, 6), "b2": (10, 10)}


def test_recipe_validation():
    with pytest.raises(ValidationError):
        bug(kind="other")
    data = {**recipe("a" * 40, [bug()]).model_dump(), "is_clean": True}
    with pytest.raises(ValidationError, match="is_clean"):
        Recipe.model_validate(data)


def test_extra_accepted_kinds_need_a_reason():
    with pytest.raises(ValidationError, match="extra_accepted_reason is required"):
        bug(extra_accepted_kinds=["missing_none_check"])
    with pytest.raises(ValidationError, match="extra_accepted_reason is required"):
        bug(extra_accepted_kinds=["missing_none_check"], extra_accepted_reason="  ")
    with pytest.raises(ValidationError, match="only allowed with extra_accepted_kinds"):
        bug(extra_accepted_reason="Why.")


@pytest.mark.parametrize(
    "kinds", [["other"], ["off_by_one"], ["missing_none_check", "missing_none_check"]]
)
def test_extra_accepted_kinds_rejects_other_own_kind_and_duplicates(kinds):
    # bug() defaults to kind off_by_one
    with pytest.raises(ValidationError):
        bug(extra_accepted_kinds=kinds, extra_accepted_reason="Why.")


def test_accepted_kinds_for_appends_extras_in_order():
    b = bug(
        kind="missing_none_check",
        extra_accepted_kinds=["wrong_comparison", "unhandled_exception"],
        extra_accepted_reason="Why.",
    )
    # unhandled_exception already comes from the D8a group, so it is not repeated.
    assert accepted_kinds_for(b) == [
        "missing_none_check",
        "unhandled_exception",
        "wrong_comparison",
    ]
    assert accepted_kinds_for(bug()) == ["off_by_one"]


def test_tests_key_ignores_accepted_kinds(monkeypatch):
    from eval import build_dataset

    tests_key = build_dataset.tests_key

    class Spec:
        image = "img"
        test_command = ["pytest"]

    monkeypatch.setattr(build_dataset, "get_repo", lambda name: Spec)
    plain = recipe("a" * 40, [bug()])
    extra = recipe(
        "a" * 40,
        [bug(extra_accepted_kinds=["wrong_comparison"], extra_accepted_reason="Why.")],
    )
    assert tests_key(plain, "diff") == tests_key(extra, "diff")
    assert tests_key(plain, "diff") != tests_key(recipe("a" * 40, [bug(replace="x\n")]), "diff")


def test_enclosing_test():
    src = "class TestX:\n    def test_a(self):\n        assert 1\n\n\ndef test_b():\n    x = 1\n"
    assert enclosing_test(src, "t.py", 3) == "t.py::TestX::test_a"
    assert enclosing_test(src, "t.py", 7) == "t.py::test_b"
    assert enclosing_test(src, "t.py", 1) is None


# --- git build -------------------------------------------------------------------------------


def test_build_in_repo_produces_diffs_and_removes_worktree(tmp_path):
    repo, commit = make_repo(tmp_path)
    built, gt = build(tmp_path, repo, recipe(commit, [bug()]))
    assert built.parent == git(repo, "rev-parse", "HEAD~1")
    assert "+    if value <= low:\n" in built.pr_diff
    assert "diff --git a/test_lib.py b/test_lib.py" in built.pr_diff
    assert "-    if value < low:\n+    if value <= low:\n" in built.plant_patch
    assert gt.bugs[0].line_start == gt.bugs[0].line_end == 8
    assert gt.base_commit == built.parent
    assert set(built.static) == {"b1"} and "flagged" in built.static["b1"]
    assert list((tmp_path / "workspaces").iterdir()) == []
    assert "workspaces" not in git(repo, "worktree", "list")


def test_build_clean_item_matches_upstream_diff(tmp_path):
    from eval.build_dataset import upstream_diff

    repo, commit = make_repo(tmp_path)
    built, gt = build(tmp_path, repo, recipe(commit, []))
    assert built.pr_diff == upstream_diff(repo, built.parent, commit)
    assert built.plant_patch == "" and gt.is_clean and gt.bugs == []


def test_build_is_deterministic(tmp_path):
    repo, commit = make_repo(tmp_path)
    r = recipe(commit, [bug()])
    a, _ = build(tmp_path, repo, r)
    b, _ = build(tmp_path, repo, r)
    assert (a.pr_diff, a.plant_patch, a.plant.ranges) == (b.pr_diff, b.plant_patch, b.plant.ranges)
