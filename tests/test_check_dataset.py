"""eval/check_dataset.py checks 5, 6, 7, 8, 10 and 11 on synthetic repositories.

Token-like strings are assembled at run time so this file never contains one that secret
scanning would recognize.
"""

import secrets

import pytest

from eval.build_dataset import load_recipe
from eval.check_dataset import (
    check_accepted_kinds,
    check_extra_kinds_documented,
    check_giveaways,
    check_oracle,
    check_plant_real,
    check_plants_on_added_lines,
    check_secrets,
    check_separation,
    oracle_findings,
    oracle_range,
    read_extra_kinds_record,
)
from tests.dataset_helpers import bug, build, make_repo, recipe


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def write_item(tmp_path, built, gt):
    item_dir = tmp_path / "dataset" / gt.item_id
    item_dir.mkdir(parents=True)
    (item_dir / "pr.diff").write_bytes(built.pr_diff.encode())
    (item_dir / "ground_truth.json").write_bytes(gt.model_dump_json(indent=2).encode())
    return item_dir


def texts(built):
    out = {p: b.decode() for p, b in built.commit_files.items()}
    out.update(built.plant.planted)
    return out


def with_range(gt, start, end):
    b = gt.bugs[0].model_copy(update={"line_start": start, "line_end": end})
    return gt.model_copy(update={"bugs": [b]})


# --- check 5: plants on added lines ----------------------------------------------------------


def test_check5_passes_for_plant_on_added_line(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, gt = build(tmp_path, path, r)
    from eval.build_dataset import upstream_diff

    up = upstream_diff(path, built.parent, commit)
    assert check_plants_on_added_lines(r, built.plant, up, built.pr_diff, gt) == []


def test_check5_fails_for_plant_on_context_line(tmp_path, repo):
    path, commit = repo
    # "return sum(values)" was in the base commit, not added by the commit
    r = recipe(commit, [bug(find="return sum(values)\n", replace="return sum(values[1:])\n")])
    built, gt = build(tmp_path, path, r)
    from eval.build_dataset import upstream_diff

    up = upstream_diff(path, built.parent, commit)
    problems = check_plants_on_added_lines(r, built.plant, up, built.pr_diff, gt)
    assert any("find lines [2] were not added" in p for p in problems)


def test_check5_fails_when_range_is_not_added_in_pr_diff(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, gt = build(tmp_path, path, r)
    from eval.build_dataset import upstream_diff

    up = upstream_diff(path, built.parent, commit)
    problems = check_plants_on_added_lines(r, built.plant, up, built.pr_diff, with_range(gt, 1, 1))
    assert any("not added lines in pr.diff" in p for p in problems)


# --- check 6: the plant is real --------------------------------------------------------------


def test_check6_passes_for_exact_range(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, gt = build(tmp_path, path, r)
    assert check_plant_real(r, built.commit_files, gt) == []


@pytest.mark.parametrize(("start", "end"), [(7, 8), (8, 9), (9, 9)])
def test_check6_fails_for_loose_or_wrong_range(tmp_path, repo, start, end):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, gt = build(tmp_path, path, r)
    problems = check_plant_real(r, built.commit_files, with_range(gt, start, end))
    assert any("!= recomputed (8, 8)" in p for p in problems)


def test_check6_fails_when_find_is_missing(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, gt = build(tmp_path, path, r)
    broken = recipe(commit, [bug(find="nowhere\n")])
    assert any("occurs 0 times" in p for p in check_plant_real(broken, built.commit_files, gt))


# --- checks 7 and 8: separation and oracle ---------------------------------------------------


def test_check7_separation(tmp_path, repo):
    path, commit = repo
    b1 = bug(find="if value is None:\n", replace="if value is False:\n", kind="wrong_comparison")
    b2 = bug(bug_id="b2", find="return min(value, high)\n", replace="return min(value, high + 1)\n")
    built, gt = build(tmp_path, path, recipe(commit, [b1, b2]))
    problems = check_separation(gt)
    assert problems == ["b1 and b2 in lib.py are 4 lines apart (< 10)"]
    assert check_separation(gt.model_copy(update={"bugs": gt.bugs[:1]})) == []


def test_check7_needs_two_files_for_items_3_and_4(tmp_path, repo):
    path, commit = repo
    built, gt = build(tmp_path, path, recipe(commit, [bug()], item_id="item_003"))
    assert check_separation(gt) == ["bugs are in 1 file(s), need at least 2"]


def test_check8_oracle_passes_for_single_line_range(tmp_path, repo):
    path, commit = repo
    built, gt = build(tmp_path, path, recipe(commit, [bug()]))
    item_dir = write_item(tmp_path, built, gt)
    assert check_oracle(item_dir, gt, texts(built)) == []


@pytest.mark.parametrize(
    ("start", "end", "after", "before"),
    [
        (20, 20, (24, 24), (16, 16)),  # 1-line bug
        (20, 22, (26, 28), (14, 16)),  # 3-line bug: same length, exactly 4 lines away
        (3, 5, (9, 11), None),  # nothing can end at line -1
        (6, 8, (12, 14), (1, 2)),  # start clamped to line 1, end still exactly at 2
    ],
)
def test_oracle_range_moves_by_the_whole_range(start, end, after, before):
    assert oracle_range(start, end, "exact") == (start, end)
    assert oracle_range(start, end, "after") == after
    assert oracle_range(start, end, "before") == before


def test_check8_oracle_passes_for_three_line_range(tmp_path, repo):
    # The old rule (+4 lines on both ends) left a 3-line finding overlapping its own bug.
    path, commit = repo
    b = bug(
        find="return min(value, high)\n",
        replace="lo = low\n    hi = high\n    return min(value, hi)\n",
    )
    built, gt = build(tmp_path, path, recipe(commit, [b]))
    assert (gt.bugs[0].line_start, gt.bugs[0].line_end) == (10, 12)
    item_dir = write_item(tmp_path, built, gt)
    assert check_oracle(item_dir, gt, texts(built)) == []
    moved = oracle_findings(gt, texts(built), "after")
    assert [(f["line_start"], f["line_end"]) for f in moved] == [(16, 18)]
    moved = oracle_findings(gt, texts(built), "before")
    assert [(f["line_start"], f["line_end"]) for f in moved] == [(4, 6)]


def test_check8_oracle_fails_when_moved_finding_hits_a_neighbour(tmp_path, repo):
    # Two same-kind bugs 2 lines apart: each moved finding lands on the other bug.
    path, commit = repo
    b1 = bug()  # line 8
    b2 = bug(bug_id="b2", find="return min(value, high)\n", replace="return min(value, high + 1)\n")
    built, gt = build(tmp_path, path, recipe(commit, [b1, b2]))
    assert [(x.line_start, x.line_end) for x in gt.bugs] == [(8, 8), (10, 10)]
    item_dir = write_item(tmp_path, built, gt)
    assert check_oracle(item_dir, gt, texts(built)) == [
        "oracle moved to start at line_end + 4 still has 1 strict hit(s)",
        "oracle moved to end at line_start - 4 still has 1 strict hit(s)",
    ]


# --- checks 1 and 12: extra accepted kinds ---------------------------------------------------

RECORD = """\
| Item | Bug | Extra kinds | Reason |
| --- | --- | --- | --- |
| item_900 | b1 | `missing_none_check`, `unhandled_exception` | Either label fits |
"""


def extra_bug(kinds=("missing_none_check",)):
    return bug(
        kind="wrong_comparison",
        extra_accepted_kinds=list(kinds),
        extra_accepted_reason="Either label fits.",
    )


def test_check1_accepted_kinds_include_extras(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [extra_bug()])
    built, gt = build(tmp_path, path, r)
    assert gt.bugs[0].accepted_kinds == ["wrong_comparison", "missing_none_check"]
    assert check_accepted_kinds(r, gt) == []
    stale = gt.model_copy(
        update={"bugs": [gt.bugs[0].model_copy(update={"accepted_kinds": ["wrong_comparison"]})]}
    )
    assert check_accepted_kinds(r, stale) == [
        "b1: accepted_kinds ['wrong_comparison'] != expected "
        "['wrong_comparison', 'missing_none_check']"
    ]


def test_check1_extras_follow_global_group_without_duplicates(tmp_path, repo):
    path, commit = repo
    b = bug(
        kind="missing_none_check",
        extra_accepted_kinds=["unhandled_exception", "wrong_comparison"],
        extra_accepted_reason="Either label fits.",
    )
    r = recipe(commit, [b])
    built, gt = build(tmp_path, path, r)
    # default_accepted_kinds gives [missing_none_check, unhandled_exception] (D8a group).
    assert gt.bugs[0].accepted_kinds == [
        "missing_none_check",
        "unhandled_exception",
        "wrong_comparison",
    ]
    assert check_accepted_kinds(r, gt) == []


def test_check12_documented_extra_kinds(tmp_path, repo):
    _, commit = repo
    assert check_extra_kinds_documented(recipe(commit, [extra_bug()]), RECORD) == []
    both = recipe(commit, [extra_bug(("missing_none_check", "unhandled_exception"))])
    assert check_extra_kinds_documented(both, RECORD) == []


def test_check12_fails_for_undocumented_bug_or_kind(tmp_path, repo):
    _, commit = repo
    other_item = recipe(commit, [extra_bug()], item_id="item_901")
    assert check_extra_kinds_documented(other_item, RECORD) == [
        "b1: extra kinds not listed in docs/decisions/0005-per-bug-accepted-kinds.md"
    ]
    more = recipe(commit, [extra_bug(("missing_none_check", "off_by_one"))])
    assert check_extra_kinds_documented(more, RECORD) == [
        "b1: extra kinds ['off_by_one'] not listed in docs/decisions/0005-per-bug-accepted-kinds.md"
    ]
    assert check_extra_kinds_documented(more, None) == [
        "docs/decisions/0005-per-bug-accepted-kinds.md is missing"
    ]


def test_check12_ignores_bugs_without_extras(tmp_path, repo):
    _, commit = repo
    assert check_extra_kinds_documented(recipe(commit, [bug()]), None) == []


def test_check12_real_record_lists_item_001():
    r = load_recipe("item_001")
    assert r.bugs[0].extra_accepted_kinds == ["missing_none_check"]
    assert check_extra_kinds_documented(r, read_extra_kinds_record()) == []


# --- check 10: no give-aways -----------------------------------------------------------------


@pytest.mark.parametrize("word", ["bug", "Planted", "INJECT", "todo", "hack", "wrong", "xxx"])
def test_check10_flags_hint_words(word):
    r = recipe("a" * 40, [bug(replace=f"if value <= low_{word}:\n")])
    assert any("replace contains" in p for p in check_giveaways(r))


def test_check10_allows_word_already_in_find():
    r = recipe("a" * 40, [bug(find="if debug < low:\n", replace="if debug <= low:\n")])
    assert check_giveaways(r) == []


def test_check10_flags_added_comment_line_and_inline_comment():
    full = recipe("a" * 40, [bug(replace="# check boundary\nif value <= low:\n")])
    assert any("adds comment line" in p for p in check_giveaways(full))
    inline = recipe("a" * 40, [bug(replace="if value <= low:  # boundary\n")])
    assert any("inline comment" in p for p in check_giveaways(inline))


def test_check10_tokenizes_planted_file_for_new_comments(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug(replace="if value <= low:  # boundary\n")])
    built, _ = build(tmp_path, path, r)
    problems = check_giveaways(r, built.plant)
    assert any("planted file has new comments ['# boundary']" in p for p in problems)


def test_check10_clean_plant_passes(tmp_path, repo):
    path, commit = repo
    r = recipe(commit, [bug()])
    built, _ = build(tmp_path, path, r)
    assert check_giveaways(r, built.plant) == []


# --- check 11: secrets safe to push ----------------------------------------------------------


def _secret_bug(value: str):
    return bug(
        kind="hardcoded_secret",
        find="if value < low:\n",
        replace=f'if value < low and connect(password="{value}"):\n',
    )


@pytest.mark.parametrize(
    "token",
    [
        "AK" + "IA" + "ABCDEFGHIJKLMNOP",
        "gh" + "p_" + "a1" * 18,
        "github" + "_pat_" + "11ABCDEFG" * 3,
        "sk" + "_live_" + "abcdef123456",
        "rk" + "_" + "abcdef",
        "xo" + "xb-" + "1234-abcd",
        "AI" + "za" + "SyA1234567890abcdef",
        "sk" + "-proj-" + "a" * 30,
        "sk" + "-ant-" + "api03-xyz",
        "-----BEGIN RSA " + "PRIVATE KEY-----",
    ],
)
def test_check11_flags_provider_formats(token):
    problems = check_secrets(recipe("a" * 40, [_secret_bug(token)]))
    assert any("token format" in p for p in problems), problems


def test_check11_accepts_generic_random_secret():
    assert check_secrets(recipe("a" * 40, [_secret_bug(secrets.token_hex(16))])) == []


def test_check11_hardcoded_secret_must_add_a_literal():
    b = bug(kind="hardcoded_secret", find="if value < low:\n", replace="if value < LOW:\n")
    assert check_secrets(recipe("a" * 40, [b])) == [
        "b1: hardcoded_secret plant adds no string literal"
    ]


def test_check11_applies_to_every_kind():
    token = "gh" + "p_" + "Z9" * 18
    b = bug(replace=f'if value <= low or name == "{token}":\n')
    assert any("github_token" in p for p in check_secrets(recipe("a" * 40, [b])))
