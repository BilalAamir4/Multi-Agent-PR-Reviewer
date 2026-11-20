# 0004: Dataset source repositories

Date: Nov 20, 2026 · Status: accepted · Implements: plan § Evaluation, D2, D3, D5 (stage 0C)

## Decision

The first-round dataset (D5) is built from commits of four repositories, approved by the owner
on Nov 20, 2026 from a measured shortlist of eleven.

| Repository | License | Why |
| --- | --- | --- |
| `simonw/sqlite-utils` | Apache-2.0 | Builds SQL strings over `sqlite3`: the natural home for `sql_injection`. Fast offline suite |
| `sqlalchemy/alembic` | MIT | Second SQL-flavoured codebase (DDL rendering, `op.execute` text), real logic in its large commits |
| `pallets/click` | BSD-3-Clause | Fast, offline, many medium and small commits with test changes |
| `marshmallow-code/marshmallow` | MIT | Small, fast, offline; validation code suits `missing_none_check` and `wrong_comparison` |

Configuration: `eval/repos.yaml` (loaded by `eval/repos.py`). Images: `sandbox_images/<name>/`.
Commit index: `eval/candidates/<name>.json`.

Rejected from the shortlist: `beetbox/beets` (package over 40K lines), `python-poetry/tomlkit`
(1 large commit), `simonw/datasette` (suite expected well over 60 s on one CPU; uses playwright
and a live server), `pallets/flask`, `dbcli/litecli`, `pallets/jinja`, `Textualize/rich` (5, 6, 3
and 3 large commits, mostly typing sweeps; litecli also has a path that is invalid on Windows).

## Selection criteria

Public GitHub repository with a permissive license (MIT, BSD-2/3-Clause, Apache-2.0, ISC, PSF);
pure Python with no compile step, Python 3.12 supported; pytest suite with no network, database
server, Docker or external service, under 60 s; about 2,000 to 40,000 `.py` lines in the package
excluding tests; at least 3 eligible commits per size bucket in the last 4 years; not bot-driven.
Across the set: one repository where `sql_injection` is natural, every D8 kind plantable, and
enough large commits for about 10 large items.

## Eligible-commit rule (`eval/find_candidates.py`)

- Non-merge commits **reachable from** `origin/<default branch>` (not only first-parent, so
  repositories that merge PR branches keep their commits), committer date in the last 4 calendar
  years, or on/after `eligible_since` when that is later.
- Author name or email does not contain `[bot]`, `dependabot`, `pre-commit-ci` or `renovate`.
- Subject does not match `format|lint|typo|bump|release|version|merge|revert|changelog|docs?:|
  style|pre-commit` (case-insensitive substring search, so "conversion" is also excluded).
- At least one non-test `.py` file **gains lines** (bugs are planted on added lines, D9a). A path
  is a test when a directory segment is exactly `tests`, `test` or `testing`, or the file is
  `test_*.py`, `*_test.py` or `conftest.py` (owner's rule). `click/testing.py` is not a test;
  `alembic/testing/*` is.
- At most 1,500 lines added plus deleted (all files); no binary change; no rename scored under 90%.
- File list from `git diff --numstat -z -M` and `--name-status -z -M`. Bucket from
  `eval.ground_truth.size_bucket` on `git diff --no-color --no-ext-diff --no-textconv -M
  <parent> <commit>` (fixed options so the user's git config cannot change the text).

### Sweep tagging (`eval/sweep.py`)

A commit gets `"sweep": true` when at least 70% of the added lines in its non-test `.py` files
are trivial: blank, import lines (including parenthesised import blocks), decorators, bare
annotations (`conn: Connection`), docstring text, or lines that equal a removed line in the same
change block after normalising away type-checker comments, string contents and quote style,
parameter/variable/return annotations, whitespace and trailing commas (or that are a piece of
those removed lines joined, for re-wrapped code). Sweep commits stay in the index with
`sweep_ratio`; session 0D does not use them for planted items. Comment-only edits are not
trivial (not in the owner's list), so pure spelling fixes in comments are not sweeps.

## Sandbox images

All images: `FROM python:3.12-slim@sha256:5e2bdb1b…4fa5b` (the 0B digest), user and group
`sandbox` 1000:1000, `RUN mkdir /work && chown 1000:1000 /work` (decision 0003),
`PYTHONDONTWRITEBYTECODE=1`, `HOME=/tmp` (the root filesystem is read-only), `PYTHONPATH` from
`pythonpath`, `USER 1000:1000`, `WORKDIR /work`. D2's runtime settings are unchanged.

Dependencies: `sandbox_images/<name>/bootstrap.Dockerfile` downloads the GitHub source archive at
`image_commit` (the base image has no git), pins `pip==25.1.1` (25.0.1 in the base image has no
`--group`), runs `pip install . --group <group>` (sqlite-utils `dev`, as its CI does; the others
`tests`), then `pip uninstall -y <package>`. `pip freeze --all` of that image is
`requirements.lock`; the real `Dockerfile` installs only the lock with `--no-deps`. Rebuilding
from the lock was confirmed to give an identical `pip freeze --all`. No image has
`pytest-timeout`, so no `test_command` uses `--timeout`.

Image additions beyond the lock (none of them installs package code; `find_spec` of the package
is `None` in each image, so tests import `/work`):

- **click: metadata-only dist-info** (`click-8.6.0.dev0.dist-info` with `METADATA` and `RECORD`
  only). `tests/test_deprecations.py` calls `importlib.metadata.version("click")` at import time,
  so without it the module fails to collect. Version from `pyproject.toml` at `image_commit`.
- **marshmallow: metadata-only dist-info** (`marshmallow-4.4.0.dist-info`), for the same reason:
  `tests/test_version_attributes.py`, present until 2025-01-04, reads the package version.
- **click: Debian `less=668-1`.** `tests/test_utils/test_echo_via_pager.py` runs the real pager
  (24 parametrised cases). This is a missing test tool, not something D2 blocks.

## Deselected tests and test-command options

click only, all in `test_command`:

| Option | Reason |
| --- | --- |
| `--deselect tests/test_termui.py::test_pipepager_less_detection_matches_stem[less]` | Writes a fake pager shell script to `tmp_path` and executes it; D2's `/tmp` tmpfs is mounted `noexec` (checked: `rw,nosuid,nodev,noexec`, exit 126) |
| `--deselect …::test_pipepager_less_detection_matches_stem[less.exe]` | Same |
| `--deselect tests/test_termui.py::test_pipepager_less_detection_case_insensitive_on_windows` | Same |
| `-W ignore::pytest.PytestRemovedIn10Warning` | Older tests pass an iterator to `parametrize`; pytest 9 warns, and click's `filterwarnings = error` turns the warning into a collection error. A test-tool deprecation, not a behaviour change |

The three deselected tests fail because of D2 settings rather than network, which is the
nearest case the stage 0C rule covers; the sandbox was not loosened.

## Narrowed date ranges (`eligible_since`)

One image per repository; older commits whose tests do not pass in it are excluded. Each boundary
was found with `uv run python -m eval.check_repos --narrow` (binary search over the indexed
commits, assuming old fail and new pass), then confirmed by the oldest-eligible range check and a
spot check of the oldest commit in every bucket plus two random commits per repository (20 runs,
all passing).

| Repository | `eligible_since` | Why older commits fail |
| --- | --- | --- |
| sqlite-utils | none (2022-11-20) | Oldest eligible commit (2022-11-29) passes |
| alembic | 2026-01-27 | Against SQLAlchemy 2.1.9 and black 26.5.1: 2 failures on 2026-01-22 (`test_stubs`), 25 on 2026-01-08 (`SADeprecationWarning` on constraint ops), 10 in 2025-03, and `dateutil` missing in 2022 |
| click | 2023-05-08 | 2023-01-19 and older: `tests/test_arguments.py::test_bytes_args` sets `encoding` on pytest 9's `DontReadFromInput`, which has no setter |
| marshmallow | 2024-10-09 | 2024-10-07 and older: `tests/base.py` imports `pytz`, not a dependency at `image_commit` |

## Baseline (3 runs at `image_commit`, sandbox wall time)

| Repository | Passed | Failed | Skipped | Median s |
| --- | --- | --- | --- | --- |
| sqlite-utils | 1497 | 0 | 16 | 9.0 |
| alembic | 1987 | 0 | 130 | 12.0 |
| click | 2625 (+1 xfailed) | 0 | 24 | 4.8 |
| marshmallow | 1192 | 0 | 0 | 2.0 |

The newest eligible large commit and the oldest eligible commit pass in every repository. Full
results are under `baseline` in `eval/candidates/<name>.json`.

## Plantable D8 kinds

- `sql_injection`, `hardcoded_secret`: recommended in sqlite-utils and alembic
  (`recommended_plants` in `eval/repos.yaml`).
- `off_by_one`, `missing_none_check`, `wrong_comparison`, `unhandled_exception`: all four.
- `ineffective_test`: all four; most eligible commits change tests (`test_files_changed`).
