# Multi-Agent-PR-Reviewer

A LangChain/LangGraph system that reviews Python pull requests with several agents, and an
evaluation of whether that finds more planted bugs than a single agent.

Status: stage 1 (single-agent baseline) done. Python repositories only.

## Running the tests

```powershell
uv sync
uv run pytest -q
```

## Reviewing a pull request

```powershell
uv run reviewer review --workspace PATH --diff PATH [--title TEXT] [--body-file PATH] [--out PATH]
uv run reviewer review --item item_001 [--out PATH] [--eval-mode]
uv run python -m eval.run --config single     # review and score every dataset item
```

## Results

Every role on Gemini 3.1 Flash Lite (D4), one run per configuration. Recall and precision use
the strict matching rule, with location-only beside recall. Median time has rate-limit waiting
removed, with the waiting beside it (D7). Cost is the D5 price estimate.

| Configuration | Recall | Precision (strict / adjusted) | False alarms per clean PR | Cost per review | Median time |
| --- | --- | --- | --- | --- | --- |
| One agent (baseline, 5 items; the full 20-item numbers come in stage 5) | 0.17 (location-only 0.33) | 0.50 / - | 0.00 | $0.027 (107K input, 412 output tokens) | 15.1 s (+37.5 s waiting) |
| Orchestrator calling specialists |  |  |  |  |  |
| Plus spawned workers on large diffs |  |  |  |  |  |
| Plus verifier |  |  |  |  |  |

Details: `eval/results/single/summary.md` and `docs/decisions/0007-single-agent-baseline.md`.
