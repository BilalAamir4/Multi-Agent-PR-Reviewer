# 0010: Per-call retries, and the scoring policy for non-complete reviews

Date: Nov 22, 2026 · Status: accepted · Implements: stage 2C. Refines routing doc S1 and S9 and
decision 0006 (retry handling only; error classes, daily and too-big handling are unchanged).
Replaces no locked decision (D1 to D12).

## Why same-model retries moved to the model call

Before this record, `QuotaRouter.run` retried a `MINUTE` or `SERVER` error by re-running the
**whole agent task** on the same model. One 503 late in a 25-call orchestrator review threw away
most of those calls: the stage 2B `dev_risky_pr` runs used 52 and 61 model calls instead of about
25 and never reached `complete`.

The routing doc only requires a whole-task re-run when **switching models** (S4): tool-call IDs
and function-call history differ between providers. A retry on the **same** model can re-send the
failed call with the same message history. So:

- `MINUTE` and `SERVER` errors are retried **per model call**, inside the agent loop, on the same
  model, with the same request. Earlier calls are not repeated.
- A **fallback to another model** still re-runs the whole task from the start (S4). That is the
  only whole-task re-run left. Only after a call has used all its retries does the error reach the
  router, which then falls over (normal mode) or raises `RetriesExhausted` (evaluation mode).
- The specialist cache (decision 0009, S6) is unchanged: a fallback re-run of the orchestrator
  still gets finished specialists from the cache.

## How it is built

LangChain 1.5.1 has `ModelRetryMiddleware` (`langchain/agents/middleware/model_retry.py`). It was
**not used**: it cannot be configured to retry exactly the `MINUTE` and `SERVER` kinds of
`classify()` (its `retry_on` is a per-exception predicate, which could be made to call `classify`),
but its backoff is fixed (`initial_delay * factor ** n` with ±25% jitter, so no `retry_after` from
the provider and not our formula), it sleeps with `time.sleep` (no injectable clock or sleep, so no
deterministic tests), it logs no waits to the review's `WaitLog` (D7), it cannot tell the router how
many retries it made, and on exhaustion it returns an error `AIMessage` by default instead of
raising the original error. Our own code is about 40 lines:

- `reviewer.models.RetryPolicy` (`max_retries`, `base_s`, `cap_s`; delay =
  `min(retry_after or base_s * 2**n, cap_s)`) and `CallRetrier.call(fn)`. It uses `classify()`,
  `retry_after_seconds()`, the router's clock and sleep, and logs each wait as `retry_minute` or
  `retry_server` in the active `WaitLog`. When retries run out it raises the **original** error with
  the retry count attached (`retries_made(err)`); every other kind (`DAILY`, `TOO_BIG`, `FATAL`) is
  raised at once, unchanged.
- `QuotaRouter.call_retrier(model_id, wait_log)` builds one for each agent task from the router's
  policy for its mode.
- The retry lives **inside `_PerCallRateLimit`** (`agents/base.py`), the per-call rate-limiter
  middleware of stage 1B, which now takes an optional retrier. The rate limiter therefore runs
  **before every attempt, retries included**. Putting it in a second middleware class would have
  changed the single agent's middleware list, which `tests/test_single_identical.py` fingerprints;
  with the retry inside the existing class that test passes unchanged.
- `run_agent` and `orchestrate` create the retrier per task and pass it to the middleware.
  `ReviewResult.retries_by_reason` (new, default empty) counts the retries per reason, children's
  included for the orchestrator (the wait log nests, decision 0009); the CLI prints it
  (`retries: 2 retry_server`) and the runner stores it in `run_log`.

## Limits and backoff

| | Per call: retries | Backoff before retry n (n = 0, 1, ...) |
| --- | --- | --- |
| Normal mode | at most 2 | `min(retry_after or 10 × 2^n, 30)` s |
| Evaluation mode | at most **6** | `min(retry_after or 15 × 2^n, 60)` s |

Evaluation tolerates waiting because D7 removes waits from the reported time. (Stage 1A used 5
retries and a 10 s base with a 30 s cap in evaluation mode; both are replaced. The retry count is
now per call, not per whole task.)

## Behaviour table (after this change)

| Error kind | Per call (inside the agent) | Task level (router), normal mode | Task level, evaluation mode |
| --- | --- | --- | --- |
| `MINUTE` | wait, re-send to the same model; up to 2 retries (normal) or 6 (evaluation) | retries used up: skip the model, **re-run the whole task** on the next (S4) | retries used up: raise `RetriesExhausted`, no fallback (S9) |
| `SERVER` | as `MINUTE` | as `MINUTE` | as `MINUTE` |
| `DAILY` | not retried, raised at once | mark exhausted, next model, whole-task re-run (S2, S4) | mark exhausted, `QuotaPaused` |
| `TOO_BIG` | not retried, raised at once | skip the model (and smaller ones), whole-task re-run (S3) | `PromptTooLarge` |
| `FATAL` | not retried, raised at once | raised | raised |

The old same-model whole-task retry path in `QuotaRouter._attempt` was **removed**: every task
(`run_agent` for the single agent and the specialists, and the orchestrator's task) uses the
per-call retrier, so no task needs it. A task that makes a model call without the retrier gets no
retries at all (its `MINUTE` / `SERVER` error goes straight to fallback or `RetriesExhausted`).

## Every configuration, including the single agent

`run_agent` is shared infrastructure, so the single agent now retries per call too. This is a
change to its **retry behaviour only**: its prompt, tools, limits, middleware list and
post-processing are untouched (`tests/test_single_identical.py` passes unchanged).

**The stage 1 baseline was recorded with task-level retries.** Its run logs
(`eval/results/single/runs/item_001..005.json`, five items) show **zero retries**: every item ran in
one attempt with `attempts: 1`, outcome `complete`, and `waits_by_reason` holds only
`rate_limiter` (0 to 41 s); there is no `retry_minute` or `retry_server` wait in any of them. The
retry rule was never exercised, so the baseline numbers are unaffected by moving retries to the call
level.

## Scoring policy for `partial` (and every other) review (Task 2b)

Before this record `eval.run` scored only `complete` and `limit_reached`. A `partial` review would
have been silently dropped for multi-agent configurations while the single agent's `limit_reached`
reviews were scored as having no findings. The policy is now **the same for every configuration**
and is based on the **cause** (`eval.run.decide_scoring`):

- **Scored**: the findings that exist are written, and an agent that failed contributes nothing.
  - `limit_reached` (orchestrator: the specialists' findings are kept, decision 0009);
  - `partial` where every non-complete child ended `limit_reached` or `error` for a
    non-infrastructure reason (for example `NoStructuredResponse` or a `FATAL` model error);
  - `error` for a non-infrastructure reason, for the single agent and the orchestrator alike.
- **Not scored, must be re-run**: the runner records the cause and exits non-zero (exit 3 for
  `quota_paused`, which also stops the run, 1 otherwise), so the item is retried later.
  - any `quota_paused`, at the top or in a child;
  - `RetriesExhausted` or `AllExhausted` anywhere in the review;
  - a `partial` caused by `sandbox unavailable` (D2b): Docker being down says nothing about the
    architecture;
  - **(stage 3, decision 0011)** `RemoteUnavailable` or `RemoteError` in a child: the A2A security
    service was not running (`security service unavailable: <url>`), or reaching it failed (HTTP
    error, JSON-RPC error, failed task, timeout, invalid response). It is infrastructure like the
    sandbox: the item is re-run, not scored.
- The decision and its cause are stored in the item's runs file (`runs/<item>.json`:
  `scoring: {decision: scored | rerun_required, cause}` and the same in every `run_log` entry), and
  `summary.md` lists the counts (latest decision per item, and recorded runs versus
  `rerun_required` runs).
- `eval/score.py` is unchanged.

Interpretations: *(stage 3)* every remote call failure, not only "service not running", is
treated as infrastructure; a remote service that answered with an `error` response of its own
(for example `RetriesExhausted` or `PromptTooLarge` inside the service) follows the existing rules
for those error classes. `PromptTooLarge` (the PR does not fit the evaluation model, deterministic) is a
non-infrastructure error and is scored with the findings that exist; a `partial` with no child
explanation and no sandbox note, or a child `partial` without a sandbox note, has no recorded
cause and is treated as `rerun_required` rather than guessed. The policy applies to runs made from
now on; the five stage 1 baseline runs were all `complete` and are not affected.

## Stage 2 routing test (`dev_risky_pr`)

Run after this change, in evaluation mode: **2 of 2 runs ended `complete` with all three
specialists called**, one with 2 per-call retries (two 503s, each recovered on its first retry
with a 15 s wait, no task re-run), one with none. The stage 2 done criterion is met. The
table of both runs, and the call-accounting defect found while collecting them, are in decision
0009 (§ Stage 2C; the defect was fixed in stage 3, decision 0011, and the figures there are the
corrected ones: 25 model calls and 76,215 in / 2,262 out in total per run). `dev_docs_pr` was not re-run (passed 2/2 on the same prompt in 2B;
per-call retries do not change routing).

Tests: `tests/test_call_retries.py` (per-call behaviour, with fake models raising the real
exception classes and a fake clock) and `tests/test_scoring_policy.py` (every case above, through
`decide_scoring` and the runner, for `--config single` and `--config orchestrator`).
