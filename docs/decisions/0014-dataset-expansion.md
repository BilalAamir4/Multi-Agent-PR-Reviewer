# 0014: Dataset expansion to 20 items

Date: 2026-11-22 · Status: accepted (sessions E1 and E2 done) · Implements: D5 (20
items: 14 planted, 6 clean; 10 large, 4 medium, 6 small), D8, D9. Replaces no locked decision.

> **Superseded (stage 4C, 2026-11-22):** the D12 spawn columns and notes below were computed with diff
> plus full file contents. The decision now counts the diff only; the current per-item spawn table is
> in decision 0012, "Stage 4C".

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

## Session E2: plan and results

| Item | Repository | Commit (subject) | Bucket | `.py` files | Lines | Planned kinds | Cross-file | Multi-line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| item_014 | sqlite-utils | `871038505bfa` (Type fixes now enforced by ty) | large | 11 | 161 | `sql_injection`, `hardcoded_secret`, `missing_none_check` | | b1 |
| item_015 | click | `7a0a3447f6dd` (Mark clearly functions private status. Deprecated Python 2 utilities.) | large | 10 | 177 | `off_by_one`, `unhandled_exception` | b1 | |
| item_016 | marshmallow | `8ca596d91194` (Deprecation warnings for marshmallow 4 changes) | large | 9 | 372 | clean | | |
| item_017 | alembic | `b0ffa3db5a16` (Don't double type-bound CHECK constraints in create_table) | medium | 4 | 274 | `wrong_comparison`, `ineffective_test` | | |
| item_018 | marshmallow | `9f751e1cef94` (Fix handling data_key in ValidationErrors raised in schema validators) | medium | 4 | 86 | clean | | |
| item_019 | click | `0d69b6ce8b25` (Adds support for editing multiple files) | small | 3 | 71 | `off_by_one`, `ineffective_test` | | b1, b2 |
| item_020 | click | `bec59289d8cf` (Fix `package_name` resolution when top-level module differs from distribution name) | small | 2 | 121 | `unhandled_exception` | | b1 |

All seven items pass the 12 checks and all 20 items pass them together (240 PASS lines); two runs of
`build_dataset --all` give a byte-identical `eval/dataset/` and change nothing in items 001 to 013.
E2 uses all four repositories (sqlite-utils 1, click 3, marshmallow 2, alembic 1; with E1 and 0D:
sqlite-utils 6, alembic 5, click 5, marshmallow 4, none above 6).

### How the plants work

- **item_014 b1** (`sql_injection`, `db.py`, 4 lines): `Database.cached_counts` formats the table names
  into the `where ... in (...)` text and then sets the parameter list to `None`. **b2**
  (`hardcoded_secret`): a random 24 character `connection_passphrase` class attribute of `Database`
  (a different spot and style from the `PRAGMA key` plants of item_013; check 11). **b3**
  (`missing_none_check`, `cli.py`): `_execute_query` raw mode indexes `row[0]` without the `if row` guard.
- **item_015 b1** (`off_by_one`, `core.py`, cross-file): `Command.get_short_help_str` passes
  `limit - 1` to `_make_default_short_help`, defined in `utils.py` (also changed by the commit;
  listed in `eval/dataset_notes.yaml`). **b2** (`unhandled_exception`, `types.py`): the file close
  callback is registered without the `_safecall` wrapper.
- **item_017 b1** (`wrong_comparison`, `schemaobj.py`): the `not` before `_is_type_bound(elem)` is
  removed, so explicit constraints are dropped. **b2** (`ineffective_test`, `tests/test_op.py`): the
  final statement assertion of `test_create_table_type_bound_check` becomes `assert_contains`.
- **item_019 b1** (`off_by_one`, `_termui_impl.py`, 3 lines): `list(filenames)[:1]` keeps only the first
  file; it does not affect the single-file `test_edit`, so b2's own test still passes with every bug
  planted. **b2** (`ineffective_test`, 2 lines): the read-back assertion of `test_edit` becomes
  `assert content`.
- **item_020 b1** (`unhandled_exception`, `decorators.py`, 3 lines): `packages_distributions()[package_name]`
  replaces `.get(package_name, [])`, so an unknown name raises `KeyError`.

### Interpretations

- **Hard-coded secret and SQL injection both in sqlite-utils.** Repository limits leave E2 one
  sqlite-utils item and the only alembic E2 commits with an SQL or credential surface were deletions
  or typing changes, so both plants are in item_014, the one E2 item from a repository named in
  `recommended_plants` with enough added lines for them.
- **Cross-file (item_015 b1).** The ground-truth line is the caller in `core.py`; the helper it calls is
  in `utils.py`, also changed by the commit.
- **Severity** is not stored; the plants were chosen so that 4 of the 10 are on a normal path (high:
  item_014 b1 and b2, item_017 b1, item_019 b1) and 6 are an edge case or a test (medium).
- **Mostly-unobserved by the repository's own tests.** Only some plants fail an existing test
  (item_014 b1, item_015 b1, item_017 b1, item_020 b1); the others pass the whole suite (check 9
  requires only 0 collection or import errors).
- **D12.** item_019 and item_020 are the first dataset items whose area stays under the 30K token limit
  (about 24.7K and 17.0K), so no specialist would spawn on them; the other 18 would spawn all three.
  This is the existing behaviour of `should_spawn_for`, not changed here. **Superseded by stage 4C:**
  the current per-item spawn table is in decision 0012, "Stage 4C" (diff-only estimate). Exactly six
  items do not spawn (item_001, 002, 011, 012, 019, 020); the other 14 do.
- **Clean items** item_016 and item_018 are marshmallow commits, the repository with the most room left.

### Extra accepted kinds

None in E2 (0 of the 3 allowed, 0 in all 22 new bugs). ADR 0005 is unchanged.

### Commits rejected

| Commit | Why |
| --- | --- |
| `d4f380b7244f` (alembic, "Drop support for SQLAlchemy 1.4") | Large, but almost entirely deletions: too few added lines to plant in |
| `91a51160fbcb` (alembic, "fix(typings): improve typing for server_default") | Annotations only; the two lines with logic are in `toimpl.py` and `tools/write_pyi.py`, too few for two plants in two files |
| `fcfccea8132e` (sqlite-utils, ANY column types) | No SQL-building or credential surface in the added lines |
| `43d5d3331f5b` (sqlite-utils, "Emit LIMIT -1 ...") | `limit`/`offset` are already formatted into the SQL by the existing code, so a plant would not be a change in the commit's own lines |
| `88efe3ea5e4e` (marshmallow, "Support Python 3.11-3.15") | Mechanical `dt.UTC` edits |
| `5b9630f50fde` (click large, localisation) | Left in the pool; no logic to plant in |
| `4f9086bf8fd9`, `1f9cd54f686e` (click small) | Considered for item_020; the exception handling in them is a `finally` restore or a close, which does not fit `unhandled_exception` as cleanly as item_020's |

First attempt at item_019 b1 (`[1:]`, which drops the first file) was changed to `[:1]` before the
item was accepted: with `[1:]` the single-file `test_edit` fails, so check 9 (the `ineffective_test`
bug's own test passes with every bug planted) could not pass. The check was not changed.

### Final statistics (`uv run python -m eval.dataset_stats`)

```
items: 20 (14 planted, 6 clean)

bucket by type:
  large   clean    3
  large   planted  7
  medium  clean    1
  medium  planted  3
  small   clean    2
  small   planted  4

bug kinds (overall / 0D / E1 / E2):
  hardcoded_secret      3   (0 / 2 / 1)
  ineffective_test      5   (2 / 1 / 2)
  missing_none_check    4   (1 / 2 / 1)
  off_by_one            4   (0 / 2 / 2)
  sql_injection         4   (1 / 2 / 1)
  unhandled_exception   4   (1 / 1 / 2)
  wrong_comparison      4   (1 / 2 / 1)
  total bugs           28

items per repository:
  alembic        5
  click          5
  marshmallow    4
  sqlite-utils   6

multi-line new bugs: 7  item_006/b2, item_013/b1, item_013/b2, item_014/b1, item_019/b1, item_019/b2, item_020/b1
cross-file new bugs: 3  item_007/b2, item_008/b2, item_015/b1

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
  item_014  py_files=11 tokens~121414 bug_reviewer, security_reviewer, test_reviewer
  item_015  py_files=10 tokens~108097 bug_reviewer, security_reviewer, test_reviewer
  item_016  py_files=9  tokens~140591 bug_reviewer, security_reviewer, test_reviewer
  item_017  py_files=4  tokens~79501  bug_reviewer, security_reviewer, test_reviewer
  item_018  py_files=4  tokens~33309  bug_reviewer, security_reviewer, test_reviewer
  item_019  py_files=3  tokens~24743  none
  item_020  py_files=2  tokens~17030  none

targets: all built items and completed sessions match
```

The single-agent baseline of stage 5 runs on all 20 items with its frozen prompt and `limits.single`
(decision 0007). Dataset complete: 20 items.
