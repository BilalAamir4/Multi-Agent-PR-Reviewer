# Evaluation dataset

Each item is a real upstream commit turned into a pull request. `base_commit` is the commit's
parent. A planted item has 1 to 3 bugs inserted into lines the commit **added**, so its diff runs
from the parent to the commit plus the bugs. A clean item is the upstream commit unchanged.

| Item | Repository | Upstream commit | Bucket | `.py` files | Lines changed | Type | Bug kinds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| item_001 | marshmallow | [`2bf3ae6ac656`](https://github.com/marshmallow-code/marshmallow/commit/2bf3ae6ac656c977ec10808b5eb0e7480ced0311) | small | 2 | 71 | planted | `wrong_comparison` |
| item_002 | alembic | [`b42ebe1ff576`](https://github.com/sqlalchemy/alembic/commit/b42ebe1ff576e02b8bfc9ef1c3c3a5fd1ac41bf4) | small | 2 | 44 | clean | none |
| item_003 | sqlite-utils | [`7a52214624ae`](https://github.com/simonw/sqlite-utils/commit/7a52214624ae0e2c3fdf07215c1bcfc1393dbd93) | medium | 4 | 138 | planted | `sql_injection`, `ineffective_test` |
| item_004 | click | [`4ece17a107d7`](https://github.com/pallets/click/commit/4ece17a107d7b2e8bc3bd14356c2b53038fe41cf) | large | 8 | 310 | planted | `missing_none_check`, `unhandled_exception`, `ineffective_test` |
| item_005 | sqlite-utils | [`07b603e562af`](https://github.com/simonw/sqlite-utils/commit/07b603e562af19f80c3d00eb59b9cf29331127ce) | large | 8 | 132 | clean | none |

item_003 has exactly 4 changed `.py` files because stage 4's timing test uses it.

## Files

- `eval/plants/<item_id>.yaml`: the **recipe**, the source of truth for an item. It names the
  repository and commit and, for each bug, its D8 kind, file, an exact `find` text (it must occur
  once in the file at the commit) and its `replace` text, plus a one-sentence description. This is
  the only file a person needs to read to audit an item.
- `eval/dataset/<item_id>/pr.diff`: `git diff --no-color --no-ext-diff --no-textconv -M <parent>`
  of the planted files.
- `eval/dataset/<item_id>/ground_truth.json`: the `GroundTruth` model (`eval/ground_truth.py`).
  Each bug's `line_start` and `line_end` are the first and last new-file lines where the planted
  file differs from the file at the commit (a line diff), and `accepted_kinds` comes from
  `default_accepted_kinds(kind)`.
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

`check_dataset` runs eleven checks per item: schema, commit eligibility, bucket, diff integrity,
plants on added lines, the plant is real, separation, oracle score, sandboxed tests, no
give-aways, and push-safe secrets. Its docstring lists them in full. Never weaken a check to make
an item pass; change the item (another commit or another plant).

`eval.repos.materialize(item_id)` creates `eval/workspaces/<item_id>/`, a worktree at
`base_commit` with `pr.diff` applied, for the review tools; `eval.repos.item_files(item_id)`
returns the PR's files for sandbox runs.

## Frozen after the first evaluation run

Once the first evaluation run happens, the recipes in `eval/plants/` and the files in
`eval/dataset/` (above all `ground_truth.json` and `pr.diff`) are **frozen**, like
`eval/score.py`. A change after that needs a decision record in `docs/decisions/`, and every
configuration must be re-scored.
