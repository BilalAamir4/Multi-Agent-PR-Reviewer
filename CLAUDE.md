# CLAUDE.md

Multi-Agent-PR-Reviewer: a LangChain/LangGraph system that reviews Python pull requests with
several agents, plus an evaluation of whether that finds more planted bugs than one agent.

## Source of truth

- `docs/plan.md` (plan, stages, Decisions D1 to D12) and `docs/model-routing.md` (model chains,
  quota router). Read both before changing design.
- D1 to D12 are **locked** (`docs/decisions/0001-locked-decisions.md`). A change needs a new
  numbered record in `docs/decisions/` naming the decision it replaces and why.
- Scoring interpretations: `docs/decisions/0002-scoring-details.md`.

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
```

## Hard rules

- Do not use `langgraph-supervisor`.
- Dependency versions are pinned with `==`; no upgrades or loosening without approval.
- PR code never runs on the host, only in the Docker sandbox (D2). Only `pytest` runs there.
- Secrets come only from `.env` (see `.env.example`); never commit `.env` or put secrets in traces.
- `eval/score.py` must not change after the first evaluation run without a decision record.
- Do not add placeholder modules for later stages.

## Layout

- `src/reviewer/schemas.py`: `Finding` schema. `src/reviewer/config.py`: `config/review.yaml`
  (D12 spawn limit).
- `eval/ground_truth.py`: dataset item format. `eval/equivalences.yaml`: D8a groups.
  `eval/score.py`: the matching rule and scorer CLI.
- `eval/dataset/<item_id>/{pr.diff, ground_truth.json}`; `eval/results/<config>/`.

## Stage status

| Stage | Status |
| --- | --- |
| 0A Skeleton, deps, `Finding`, ground truth, scorer | done |
| 0B Environment checks (models, tracing, sandbox, `langgraph dev`) | not started |
| 0C Dataset repositories and first 5 items | not started |
| 1 One agent | not started |
| 2 Calling specialists | not started |
| 3 A2A | not started |
| 4 Spawning workers | not started |
| 5 Evaluation | not started |
| 6 Approval and service | not started |
| 7 Hardening | not started |
