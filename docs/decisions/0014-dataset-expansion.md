# 0014: Dataset expansion to 20 items

Date: 2026-11-22 · Status: accepted (session E1 done; E2 adds its section) · Implements: D5 (20
items: 14 planted, 6 clean; 10 large, 4 medium, 6 small), D8, D9. Replaces no locked decision.

Items item_006 to item_020 are added before stage 5's first run, in two sessions: **E1** builds
item_006 to item_013 and **E2** builds item_014 to item_020. Recipes, `build_dataset` and the 12
checks of `check_dataset` are those of session 0D (decisions 0004, 0005); no check was changed or
weakened, and item_001 to item_005 are untouched. No model was called. From stage 5's first run
onwards all 20 items are frozen (`eval/dataset/README.md`). The single-agent baseline of stage 5 runs
on the new items with its frozen prompt (`prompts/single.md`) and `limits.single`
(decision 0007).

## Session E1: plan and results

| Item | Repository | Commit (subject) | Bucket | `.py` files | Lines | Planned kinds | Cross-file | Multi-line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_006 | alembic | `326e8fb37aa6` (Treat empty string comments as no comment in autogenerate) | large | 9 | 231 | `wrong_comparison`, `ineffective_test` | | b2 |
| item_007 | alembic | `7ed6e478e849` (Add autogenerate support for CHECK constraint detection) | large | 12 | 1214 | `missing_none_check`, `unhandled_exception`, `missing_none_check` | b2 | |
| item_008 | alembic | `2e1f1599965c` (Use ForeignKey target accessors instead of splitting colspecs) | large | 8 | 226 | `off_by_one`, `wrong_comparison` | b2 | |
| item_009 | marshmallow | `68c68e3a5625` (Make Field a generic type; refactor inheritance hierarchies) | large | 9 | 707 | clean | | |
| item_010 | sqlite-utils | `2303b80aef69` (.transform() preserves check constraints) | medium | 4 | 449 | `sql_injection`, `off_by_one` | | |
| item_011 | sqlite-utils | `e4935e064407` (transform: coerce empty strings to NULL ...) | small | 2 | 62 | `sql_injection` | | |
| item_012 | click | `c53d3732b98c` (Fix `show_envvar` output for a sequence or an empty `envvar`) | small | 2 | 64 | clean | | |
| item_013 | sqlite-utils | `fd5b09f64b43` (Database as a context manager, fixed many pytest warnings) | large | 11 | 301 | `hardcoded_secret`, `hardcoded_secret` | | b1, b2 |

All eight items pass the 12 checks (`eval.check_dataset`); two runs of `build_dataset --all`
change nothing. E1 uses all four repositories (alembic 3, sqlite-utils 3, marshmallow 1, click 1;
with the 0D items: alembic 4, sqlite-utils 5, marshmallow 2, click 2, so E2 has room for at most
one more sqlite-utils item and two more alembic items).

### How the cross-file and multi-line plants work

- **item_007 b2** (`unhandled_exception`): `_compare_check_constraints` in
  `compare/check_constraints.py` catches `KeyError` instead of `NotImplementedError`, which the base
  `_InspectorConv.get_check_constraints` in `compare/util.py` raises.
- **item_008 b2** (`wrong_comparison`): `_fk_target_is_named` in `util/sqla_compat.py` returns the
  inverse (`target_column is not None`); its caller is `SchemaObjects._ensure_table_for_fk` in
  `operations/schemaobj.py`.
- Both are listed in `eval/dataset_notes.yaml`. Multi-line is `line_end > line_start` in
  `ground_truth.json`: item_006 b2 (2 lines), item_013 b1 and b2 (2 lines each).

### Interpretations

- **Hard-coded secrets.** Neither alembic nor sqlite-utils handles credentials, so no commit adds
  a natural one. Both item_013 plants add a random 24 character string as an SQLCipher-style key
  (`PRAGMA key`, a no-op on stock SQLite, so the tests are unaffected) in `Database.__enter__`
  and in `memory --save`. They are generic strings, not any provider token format (check 11).
- **Severity** is not stored; the plants were chosen so that 7 of the 12 are on a normal path
  (high) and 5 on an edge or test (medium).
- **item_006 b1** and **item_013** are not caught by the repository's own tests (the sqlite
  suite does not exercise comments, nor a key); that is allowed by check 9, which only needs
  0 collection or import errors.
- **D12.** Every item, including the small ones, would spawn sub-agents: the token estimate of
  the area (diff plus full file contents) is far above the 30K limit even for 2 files
  (`dataset_stats` shows it). This is the existing behaviour of `should_spawn_for`; it
  makes the D6 reuse rule reuse nothing on this dataset. Not changed here.

### Extra accepted kinds

None in E1 (0 of the 3 allowed). ADR 0005 is unchanged.

### Commits rejected

| Commit | Why |
| --- | --- |
| `2d3c6b9a1e50` (sqlite-utils, "Escape tokenize argument in enable_fts") | The commit is the fix of an SQL injection and its message says so, so the PR body gives the planted bug away |
| `8ca596d91194` (marshmallow, "Deprecation warnings for marshmallow 4 changes") | Only warning text, no logic in two files to plant in |
| `fcfccea8132e`, `b379a2a0c365`, `f5c63088e145`, `871038505bfa` (sqlite-utils large) | Considered for item_013 and item_009; no credential or SQL surface, or mostly mechanical edits; left in the pool |

### Statistics (E1 state, `uv run python -m eval.dataset_stats`)

```
items: 13 (9 planted, 4 clean)

bucket by type:
  large   clean    2
  large   planted  5
  medium  planted  2
  small   clean    2
  small   planted  2

bug kinds (overall / 0D / E1 / E2):
  hardcoded_secret      2   (0 / 2 / 0)
  ineffective_test      3   (2 / 1 / 0)
  missing_none_check    3   (1 / 2 / 0)
  off_by_one            2   (0 / 2 / 0)
  sql_injection         3   (1 / 2 / 0)
  unhandled_exception   2   (1 / 1 / 0)
  wrong_comparison      3   (1 / 2 / 0)
  total bugs           18

items per repository:
  alembic        4
  click          2
  marshmallow    2
  sqlite-utils   5

multi-line new bugs: 3  item_006/b2, item_013/b1, item_013/b2
cross-file new bugs: 2  item_007/b2, item_008/b2

D12 spawning (specialists that would spawn):
  item_001  py_files=2  tokens~52210  bug_reviewer, security_reviewer, test_reviewer
  item_002  py_files=2  tokens~30871  bug_reviewer, security_reviewer, test_reviewer
  item_003  py_files=4  tokens~156426 bug_reviewer, security_reviewer, test_reviewer
  item_004  py_files=8  tokens~76275  bug_reviewer, security_reviewer, test_reviewer
  item_005  py_files=8  tokens~147980 bug_reviewer, security_reviewer, test_reviewer
  item_006  py_files=9  tokens~108144 bug_reviewer, security_reviewer, test_reviewer
  item_007  py_files=12 tokens~107070 bug_reviewer, security_reviewer, test_reviewer
  item_008  py_files=8  tokens~87787  bug_reviewer, security_reviewer, test_reviewer
  item_009  py_files=9  tokens~133537 bug_reviewer, security_reviewer, test_reviewer
  item_010  py_files=4  tokens~100228 bug_reviewer, security_reviewer, test_reviewer
  item_011  py_files=2  tokens~95792  bug_reviewer, security_reviewer, test_reviewer
  item_012  py_files=2  tokens~93253  bug_reviewer, security_reviewer, test_reviewer
  item_013  py_files=11 tokens~139930 bug_reviewer, security_reviewer, test_reviewer

targets: all built items and completed sessions match
```

## Session E2

Not done yet.
