# CLAUDE.md

Multi-Agent-PR-Reviewer: a LangChain/LangGraph system that reviews Python pull requests with
several agents, plus an evaluation of whether that finds more planted bugs than one agent.

## Source of truth

- `docs/plan.md` (plan, stages, Decisions D1 to D12) and `docs/model-routing.md` (model chains,
  quota router). Read both before changing design.
- D1 to D12 are **locked** (`docs/decisions/0001-locked-decisions.md`). A change needs a new
  numbered record in `docs/decisions/` naming the decision it replaces and why.
- Scoring interpretations: `docs/decisions/0002-scoring-details.md`.
- Per-bug extra accepted kinds: `docs/decisions/0005-per-bug-accepted-kinds.md`.
- Router errors, retries, resets and wait accounting: `docs/decisions/0006-router-and-waits.md`.
- Dataset source repositories, images, eligible commits: `docs/decisions/0004-dataset-source-repos.md`.
- Post-processing (D11 for every configuration), call limits, the stage 1 baseline:
  `docs/decisions/0007-single-agent-baseline.md`.
- Specialists, shared agent machinery, composed prompts, `run_tests`, the `partial` status:
  `docs/decisions/0008-specialists.md`.
- Finding categories: `docs/categories.md`. All agent prompts must take their category
  definitions from this file.

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
uv run python -m eval.check_dataset [item_id ...] [--no-tests]  # 12 checks per item (Docker)
uv run reviewer review --workspace PATH --diff PATH [--title T] [--body-file P] [--out P] [--eval-mode]
uv run reviewer review --agent bug_reviewer|security_reviewer|test_reviewer --workspace PATH --diff PATH `
    [--focus TEXT] [--sandbox-repo NAME] ...   # default --agent single; exit 5 = partial (D2b)
uv run reviewer review --item item_001 [--out PATH] [--eval-mode]   # exit 0/3 quota/4 limit/1 error
docker build -t mapr-sandbox-dev_fixture:local sandbox_images/dev_fixture   # run_tests on dev fixtures
uv run python -m eval.run --config single [--items ...] [--force] [--reason TEXT]  # eval + score
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
- **No tuning on the dataset.** Develop and debug prompts only on `tests/fixtures/dev_pr/`
  (`tests/dev_pr.py`). Evaluation items run once per configuration; a technical failure may be
  re-run (`eval.run --force --reason ...`), but prompts, limits and post-processing never change
  after seeing an evaluation score or finding. Every re-run is recorded with its reason.
- Frozen since the stage 1B baseline (decision 0007): `eval/plants/`, `eval/dataset/`,
  `eval/score.py`, and for the "One agent" row `src/reviewer/prompts/single.md`
  (`aa94f1ae2cf6`) and `limits.single` in `config/review.yaml`.
- Prompts never contain ground truth, `eval/plants/`, `meta.json` fields other than
  `pr_title`/`pr_body`, or any mention of planted bugs, evaluation or scoring.
- **Specialists are not run on evaluation items before stage 5.** Develop them only on the dev
  fixtures in `tests/fixtures/dev_*` (`tests/dev_pr.py`); the CLI refuses `--item` with a
  specialist. `single.md` stays one frozen file; it never uses `prompts/common/`.
- Do not add placeholder modules for later stages.

## Layout

- `src/reviewer/schemas.py`: `Finding` schema. `src/reviewer/config.py`: `config/review.yaml`
  (D12 spawn limit) and `config/models.yaml` (`load_models_config`).
- `src/reviewer/sandbox.py`: `run_in_sandbox`, `get_client`, `SandboxUnavailable` (D2, D2b).
- `src/reviewer/models.py`: `QuotaRouter(config, evaluation=False).run(role, task, prompt_tokens)`
  calls `task(chat_model, model_id)` and returns `RouterResult(value, model)`; also `classify`,
  `estimate_tokens`, `review_waits()`, `QuotaTracker` (`quota_state.json`). Decision 0006.
- `src/reviewer/agents/output.py`: `FindingDraft`, `ReviewOutput`, `to_findings` (schema,
  not_in_diff, evidence, duplicate; every agent). `src/reviewer/agents/base.py`: `AgentSpec`,
  `run_agent(spec, ctx, user_message, router, *, item_id, evaluation) -> ReviewResult`
  (create_agent + ToolStrategy + call-limit middleware), `compose_prompt`, `ReviewResult`
  (`agent`, `notes`, status `partial`). `src/reviewer/agents/single.py`:
  `review(ctx, title, body, router)` on `base`. `src/reviewer/agents/specialists.py`:
  `specialist_spec`, `run_specialist(name, ctx, *, pr_title, pr_body, focus, router, ...)`,
  `run_security_review(request: dict) -> dict` (stage 3). `src/reviewer/prompts/`: `single.md`
  (frozen), `<specialist>.md` with `<!-- include: NAME -->` lines for `common/NAME.md`.
  `src/reviewer/cli.py`.
- `src/reviewer/tools/`: `ReviewContext.from_paths(workspace, diff_path, sandbox=None)`, then
  `make_tools(ctx)` gives `list_changed_files`, `read_diff`, `read_file`, `search_code`,
  `run_ruff`, `run_bandit` (read-only, strings, `ERROR: ...` on failure; needs `rg` on PATH);
  `make_tools(ctx, names, notes)` can add `run_tests` (`tools/testing.py`: pytest through
  `run_in_sandbox` with a `SandboxSpec`, at most `limits.<agent>.max_run_tests_calls` calls).
- `scripts/`: stage 0B environment checks; `scripts/langgraph_check/`: echo graph for `langgraph dev`.
- `eval/ground_truth.py`: dataset item format. `eval/equivalences.yaml`: D8a groups.
  `eval/score.py`: the matching rule and scorer CLI. `eval/run.py`: runs a configuration in
  evaluation mode, writes `eval/results/<config>/{item_*.json, runs/, score.json, summary.md}`.
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
  `dev_fixture` in `repos.yaml` is the local image for the dev fixtures (`url: null`, only names
  in `LOCAL_REPOS`); `eval.repos.sandbox_spec(name)`, `item_sandbox(item_id)`,
  `dataset_repos()`.
- `tests/fixtures/dev_{pr,security_pr,tests_pr,docs_pr,risky_pr}/`: synthetic dev fixtures
  (`workspace/`, `pr.diff`, informational `expected.json`).

## Stage status

| Stage | Status |
| --- | --- |
| 0A Skeleton, deps, `Finding`, ground truth, scorer | done |
| 0B Environment checks (models, tracing, sandbox, `langgraph dev`) | done (see `docs/environment-check.md`) |
| 0C Dataset source repositories, sandbox images, commit index | done (see decision 0004) |
| 0D First 5 dataset items (pick commits, plant bugs) | done (see `eval/dataset/README.md`) |
| **Stage 0** | **complete** |
| 1A Model router and review tools, dataset fixes | done (decisions 0005, 0006) |
| 1B One agent, CLI, evaluation runner, baseline | done (decision 0007) |
| **Stage 1** | **complete** |
| 2A Shared agent machinery, `run_tests`, three specialists | done (decision 0008) |
| 2B Orchestrator calling the specialists, routing test | not started |
| 3 A2A | not started |
| 4 Spawning workers | not started |
| 5 Evaluation | not started |
| 6 Approval and service | not started |
| 7 Hardening | not started |
