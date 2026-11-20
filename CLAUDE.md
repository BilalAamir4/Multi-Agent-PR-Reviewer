# CLAUDE.md

Multi-Agent-PR-Reviewer: a LangChain/LangGraph system that reviews Python pull requests with
several agents, plus an evaluation of whether that finds more planted bugs than one agent.

## Source of truth

- `docs/plan.md` (plan, stages, Decisions D1 to D12) and `docs/model-routing.md` (model chains,
  quota router). Read both before changing design.
- D1 to D12 are **locked** (`docs/decisions/0001-locked-decisions.md`). A change needs a new
  numbered record in `docs/decisions/` naming the decision it replaces and why.
- Scoring interpretations: `docs/decisions/0002-scoring-details.md`.
- Dataset source repositories, images, eligible commits: `docs/decisions/0004-dataset-source-repos.md`.

## Environment

- Windows 11, **PowerShell** (not WSL), Python 3.12, `uv` for everything (never bare `pip`).
- All text files UTF-8 with LF endings (`.gitattributes`, `core.autocrlf false`).
- Paths from diffs and ground truth are POSIX strings handled with `PurePosixPath` (D3).
  Use `pathlib` for filesystem paths.

## Commands

```powershell
uv sync                     # install from uv.lock
uv run pytest -q            # tests
uv run ruff check .         # lint
uv run ruff format .        # format
uv run python -m eval.score --config <config_name>   # score eval/results/<config_name>/
uv run pytest -q -m "not integration"                # unit tests only (what CI runs)
uv run pytest -q -m integration     # needs keys and Docker; spends free-tier model calls
uv run python scripts/list_models.py      # writes docs/model-inventory.md (no quota used)
uv run python scripts/check_finding.py    # 1 call to the evaluation model, traced
uv run python scripts/check_tool_calls.py [model-id ...] [--model-calls 1]   # 2 calls per model
uv run python scripts/check_sandbox.py    # Docker probe of the D2 settings
uv run python scripts/check_langgraph.py  # langgraph dev + A2A on port 2024 (no model calls)
uv run python -m eval.find_candidates <name>  # index eligible commits -> eval/candidates/<name>.json
uv run python -m eval.check_repos [name ...]  # baseline sandboxed test runs (Docker, no model calls)
uv run python -m eval.check_repos --narrow <name>  # propose eligible_since for repos.yaml
docker build -t <image from eval/repos.yaml> sandbox_images/<name>  # build a repository's image
uv run python -m eval.build_dataset --all     # build eval/dataset/ from eval/plants/*.yaml (Docker)
uv run python -m eval.build_dataset <item_id> [--rerun-tests]   # one item; reruns cached tests
uv run python -m eval.check_dataset [item_id ...] [--no-tests]  # 11 checks per item (Docker)
```

Free-tier budgets are small (Gemini 4 Hydrogen and the Flash group: 20 requests a day). Do not
run the model checks in loops.

## Hard rules

- Do not use `langgraph-supervisor`.
- Dependency versions are pinned with `==`; no upgrades or loosening without approval.
- PR code never runs on the host, only in the Docker sandbox (D2). Only `pytest` runs there.
- Upstream repository code never runs on the host. On the host only `git` and read-only file
  inspection touch `eval/repos/`; tests, `setup.py` and build hooks run only in `run_in_sandbox`
  or inside `docker build`. Clones use `core.autocrlf false` and are never committed.
- Sandbox image contract (`docs/decisions/0003-sandbox-copy-into-volume.md`): every image passed
  to `run_in_sandbox`, including each Dockerfile in `sandbox_images/`, must contain
  `RUN mkdir /work && chown 1000:1000 /work`.
- Model IDs, providers, limits and chains live only in `config/models.yaml`.
- Tests that call a model API or Docker are marked `integration` and skip with a reason when
  keys or Docker are missing.
- Secrets come only from `.env` (see `.env.example`); never commit `.env` or put secrets in traces.
- `eval/score.py` must not change after the first evaluation run without a decision record.
  Neither may `eval/plants/*.yaml` or `eval/dataset/` (recipes and ground truth are frozen then).
- Never weaken a check in `eval/check_dataset.py` to make an item pass; change the item.
- Do not add placeholder modules for later stages.

## Layout

- `src/reviewer/schemas.py`: `Finding` schema. `src/reviewer/config.py`: `config/review.yaml`
  (D12 spawn limit) and `config/models.yaml` (`load_models_config`).
- `src/reviewer/sandbox.py`: `run_in_sandbox`, `get_client`, `SandboxUnavailable` (D2, D2b).
- `scripts/`: stage 0B environment checks; `scripts/langgraph_check/`: echo graph for `langgraph dev`.
- `eval/ground_truth.py`: dataset item format. `eval/equivalences.yaml`: D8a groups.
  `eval/score.py`: the matching rule and scorer CLI.
- `eval/dataset/<item_id>/{pr.diff, ground_truth.json, meta.json}` (see `eval/dataset/README.md`);
  `eval/results/<config>/`. `eval/plants/<item_id>.yaml`: plant recipes (source of truth).
  `eval/build_dataset.py` (recipes -> items), `eval/check_dataset.py` (item checks).
  `eval.repos.materialize(item_id)` -> `eval/workspaces/<item_id>/` (worktree at `base_commit`
  plus `pr.diff`, gitignored); `eval.repos.item_files(item_id)` for sandbox runs.
- `eval/repos.yaml` + `eval/repos.py`: source repositories, `ensure_clone`, `snapshot_files`,
  `run_repo_tests`. `eval/find_candidates.py` (commit filter, buckets), `eval/sweep.py` (sweep
  tagging), `eval/check_repos.py` (baseline). `eval/candidates/<name>.json`: commit index and
  baseline. `sandbox_images/<name>/`: `Dockerfile` (installs `requirements.lock`),
  `bootstrap.Dockerfile` (how the lock was made). `eval/repos/` (clones) is gitignored.

## Stage status

| Stage | Status |
| --- | --- |
| 0A Skeleton, deps, `Finding`, ground truth, scorer | done |
| 0B Environment checks (models, tracing, sandbox, `langgraph dev`) | done (see `docs/environment-check.md`) |
| 0C Dataset source repositories, sandbox images, commit index | done (see decision 0004) |
| 0D First 5 dataset items (pick commits, plant bugs) | done (see `eval/dataset/README.md`) |
| **Stage 0** | **complete** |
| 1 One agent | not started |
| 2 Calling specialists | not started |
| 3 A2A | not started |
| 4 Spawning workers | not started |
| 5 Evaluation | not started |
| 6 Approval and service | not started |
| 7 Hardening | not started |
