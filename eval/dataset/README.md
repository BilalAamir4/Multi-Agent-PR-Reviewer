# Evaluation dataset

Each item is a real upstream commit turned into a pull request. `base_commit` is the commit's
parent. A planted item has 1 to 3 bugs inserted into lines the commit **added**, so its diff runs
from the parent to the commit plus the bugs. A clean item is the upstream commit unchanged.

| Item | Repository | Upstream commit | Bucket | `.py` files | Lines changed | Type | Bug kinds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| item_001 | marshmallow | [`2bf3ae6ac656`](https://github.com/marshmallow-code/marshmallow/commit/2bf3ae6ac656c977ec10808b5eb0e7480ced0311) | small | 2 | 71 | planted | `wrong_comparison` (also accepts `missing_none_check`) |
| item_002 | alembic | [`b42ebe1ff576`](https://github.com/sqlalchemy/alembic/commit/b42ebe1ff576e02b8bfc9ef1c3c3a5fd1ac41bf4) | small | 2 | 44 | clean | none |
| item_003 | sqlite-utils | [`7a52214624ae`](https://github.com/simonw/sqlite-utils/commit/7a52214624ae0e2c3fdf07215c1bcfc1393dbd93) | medium | 4 | 138 | planted | `sql_injection`, `ineffective_test` |
| item_004 | click | [`4ece17a107d7`](https://github.com/pallets/click/commit/4ece17a107d7b2e8bc3bd14356c2b53038fe41cf) | large | 8 | 310 | planted | `missing_none_check`, `unhandled_exception`, `ineffective_test` |
| item_005 | sqlite-utils | [`07b603e562af`](https://github.com/simonw/sqlite-utils/commit/07b603e562af19f80c3d00eb59b9cf29331127ce) | large | 8 | 132 | clean | none |
| item_006 | alembic | [`326e8fb37aa6`](https://github.com/sqlalchemy/alembic/commit/326e8fb37aa6bd85cfccad11919c3646e7677abe) | large | 9 | 231 | planted | `wrong_comparison`, `ineffective_test` |
| item_007 | alembic | [`7ed6e478e849`](https://github.com/sqlalchemy/alembic/commit/7ed6e478e84956958fe1c6ee5c76f694bfb3b808) | large | 12 | 1214 | planted | `missing_none_check` (2), `unhandled_exception` |
| item_008 | alembic | [`2e1f1599965c`](https://github.com/sqlalchemy/alembic/commit/2e1f1599965ccc841b97f097acc5c59c42f6e65c) | large | 8 | 226 | planted | `off_by_one`, `wrong_comparison` |
| item_009 | marshmallow | [`68c68e3a5625`](https://github.com/marshmallow-code/marshmallow/commit/68c68e3a562591a5b705907d948faac843e216c7) | large | 9 | 707 | clean | none |
| item_010 | sqlite-utils | [`2303b80aef69`](https://github.com/simonw/sqlite-utils/commit/2303b80aef69abe418636dc980c5350958d55411) | medium | 4 | 449 | planted | `sql_injection`, `off_by_one` |
| item_011 | sqlite-utils | [`e4935e064407`](https://github.com/simonw/sqlite-utils/commit/e4935e064407bc995f77795c025c33cef52d742e) | small | 2 | 62 | planted | `sql_injection` |
| item_012 | click | [`c53d3732b98c`](https://github.com/pallets/click/commit/c53d3732b98c764932e146c007146ad402acb229) | small | 2 | 64 | clean | none |
| item_013 | sqlite-utils | [`fd5b09f64b43`](https://github.com/simonw/sqlite-utils/commit/fd5b09f64b4353b0c85084eb04415e344d05832f) | large | 11 | 301 | planted | `hardcoded_secret` (2) |
| item_014 | sqlite-utils | [`871038505bfa`](https://github.com/simonw/sqlite-utils/commit/871038505bfad965177fd12eb45e62e7f4d25d65) | large | 11 | 161 | planted | `sql_injection`, `hardcoded_secret`, `missing_none_check` |
| item_015 | click | [`7a0a3447f6dd`](https://github.com/pallets/click/commit/7a0a3447f6ddd2c15438c5d098e289323f9f9556) | large | 10 | 177 | planted | `off_by_one`, `unhandled_exception` |
| item_016 | marshmallow | [`8ca596d91194`](https://github.com/marshmallow-code/marshmallow/commit/8ca596d91194a6c2d0d992a7b66490a0e909d719) | large | 9 | 372 | clean | none |
| item_017 | alembic | [`b0ffa3db5a16`](https://github.com/sqlalchemy/alembic/commit/b0ffa3db5a1654f365072de2ffcdb226dc28397b) | medium | 4 | 274 | planted | `wrong_comparison`, `ineffective_test` |
| item_018 | marshmallow | [`9f751e1cef94`](https://github.com/marshmallow-code/marshmallow/commit/9f751e1cef943e2ca1a77f93d3483b4a171d4303) | medium | 4 | 86 | clean | none |
| item_019 | click | [`0d69b6ce8b25`](https://github.com/pallets/click/commit/0d69b6ce8b25fe31ea580c608adb8e45df3805bb) | small | 3 | 71 | planted | `off_by_one`, `ineffective_test` |
| item_020 | click | [`bec59289d8cf`](https://github.com/pallets/click/commit/bec59289d8cf9b9b4010642b2fee483e5f8eeefc) | small | 2 | 121 | planted | `unhandled_exception` |

item_003 has exactly 4 changed `.py` files because stage 4's timing test uses it.

item_006 to item_013 are session E1 and item_014 to item_020 session E2 of the dataset expansion
(decision 0014); together they complete the 20 items of D5 (14 planted, 6 clean). `uv run python -m eval.dataset_stats` prints the
bucket, kind, repository, multi-line, cross-file and D12 spawning counts and compares them with the
targets; `eval/dataset_notes.yaml` lists the cross-file bugs.

## Files

- `eval/plants/<item_id>.yaml`: the **recipe**, the source of truth for an item. It names the
  repository and commit and, for each bug, its D8 kind, file, an exact `find` text (it must occur
  once in the file at the commit) and its `replace` text, plus a one-sentence description. This is
  the only file a person needs to read to audit an item.
- `eval/dataset/<item_id>/pr.diff`: `git diff --no-color --no-ext-diff --no-textconv -M <parent>`
  of the planted files.
- `eval/dataset/<item_id>/ground_truth.json`: the `GroundTruth` model (`eval/ground_truth.py`).
  Each bug's `line_start` and `line_end` are the first and last new-file lines where the planted
  file differs from the file at the commit (a line diff), and `accepted_kinds` is
  `default_accepted_kinds(kind)` plus the bug's `extra_accepted_kinds` from the recipe, if any
  (each with a reason and a row in `docs/decisions/0005-per-bug-accepted-kinds.md`).
- `eval/dataset/<item_id>/meta.json`: commit subject and body (`pr_title`, `pr_body`), size,
  `sweep`, the plant patch, sandboxed test results (original commit, planted item, each bug alone,
  and each `ineffective_test` bug's own test) and informational `static_hits` (ruff `E,F,B,S`
  without S101, and bandit without B101, on each bug's lines).

## Rebuild and check

```powershell
uv run python -m eval.build_dataset --all          # or: eval.build_dataset item_003
uv run python -m eval.check_dataset                # all items; exits 1 if any check fails
uv run python -m eval.check_dataset item_004 --no-tests   # skip the sandbox (reported as SKIP)
```

Building reads the commit's files from git objects, applies the recipe, writes the planted files
into a temporary worktree under `eval/workspaces/`, and runs the tests in the repository's sandbox
image (D2). Test durations vary between runs, so test results are reused from the existing
`meta.json` while their inputs (`tests_key`: image, test command, commit, `pr.diff` hash and the
bugs) are unchanged. That keeps a rebuild byte-identical; `--rerun-tests` forces new runs.

`check_dataset` runs twelve checks per item: schema, commit eligibility, bucket, diff integrity,
plants on added lines, the plant is real, separation, oracle score, sandboxed tests, no
give-aways, push-safe secrets, and documented extra kinds. Its docstring lists them in full.
Never weaken a check to make an item pass; change the item (another commit or another plant).

`eval.repos.materialize(item_id)` creates `eval/workspaces/<item_id>/`, a worktree at
`base_commit` with `pr.diff` applied, for the review tools; `eval.repos.item_files(item_id)`
returns the PR's files for sandbox runs.

## Frozen from the stage 1B baseline run onwards

The stage 1B baseline run is the first time a score is recorded. From that run onwards the recipes
in `eval/plants/` and the files in `eval/dataset/` (above all `ground_truth.json` and `pr.diff`)
are **frozen**, like `eval/score.py`. item_006 to item_020 were added before stage 5's first run;
from stage 5's first run onwards, all 20 items are frozen. A change after that needs a decision record in
`docs/decisions/`, and every configuration must be re-scored.
