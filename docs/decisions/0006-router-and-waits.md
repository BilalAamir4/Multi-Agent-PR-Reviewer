# 0006: Quota router, retries and wait accounting

Date: Nov 21, 2026 · Status: accepted · Implements: routing doc S1 to S9, D4, D7 in
`src/reviewer/models.py` (stage 1A). Replaces the sketch in `docs/model-routing.md` § Router.

## Error classes

`classify(err)` walks `err` and its `__cause__` chain (langchain-google-genai keeps only a message
on the outer exception; the structured `google.genai` error is its cause) and returns:

| Kind | Detected from | Normal mode | Evaluation mode |
| --- | --- | --- | --- |
| `TOO_BIG` | HTTP 413; a `ContextOverflowError` (Google and Groq wrappers); Groq 400 `code: context_length_exceeded`; Groq 429 "Request too large"; Gemini 400 token-limit message | Skip to the next model; also skip later models whose TPM is no larger than the failed one's. Nothing is marked exhausted (S3) | Raise `PromptTooLarge` |
| `DAILY` | Gemini `QuotaFailure.violations[].quotaId` containing `PerDay`; Groq 429 message naming `per day` / `(RPD)` / `(TPD)` | Mark exhausted in `QuotaTracker`, go to the next model at once (S2) | Mark exhausted, raise `QuotaPaused(model, reset_at)`; the caller saves progress and resumes after the reset |
| `MINUTE` | Any other 429 / `RESOURCE_EXHAUSTED` | Wait, retry the same model at most 2 times, then next model; nothing marked (S1) | Same wait, up to 5 retries, then `RetriesExhausted` |
| `SERVER` | 500, 502, 503 (including "The model is overloaded"), 504; httpx/httpx2 timeout, network and protocol errors; `groq.APIConnectionError`/`APITimeoutError` | As `MINUTE` | As `MINUTE` |
| `FATAL` | Everything else: 400, 401, 403, 404, schema and parsing errors | Raise at once, never retried | Raise at once |

Message matching is a commented fallback only where no field exists (Groq's window, Gemini's
token-limit 400, a 429 without a `QuotaFailure` detail).

Retry wait: `min(retry_after or 10 × 2^attempt, 30)` seconds, where `retry_after` comes from
Gemini's `RetryInfo.retryDelay` or the `retry-after-ms` / `retry-after` header. A `retry_after`
of 0 falls back to the backoff. Evaluation mode never falls back to another model (D4, S9).

## The router owns retries

Chat models are built with `init_chat_model(id, model_provider=..., temperature=0,
max_retries=0)`. Stage 0B saw Gemini's client retry 6 times on its own; with `max_retries=0` the
google-genai client makes one attempt (`attempts=0` is coerced to 1) and the groq client none.

## A fallback re-runs the whole task (S4)

`QuotaRouter.run(role, task, prompt_tokens)` calls `task(chat_model, model_id)`, which runs one
whole agent task. A failed attempt is never continued on another model: the next pick re-runs
the task from the start, because tool-call IDs and function-call history differ between
providers. The tools only read, so a re-run is safe. The result carries the model that produced
it (S7).

## Token estimate

`estimate_tokens(text) = ceil(len(text) / 3)`, a deliberate overestimate (text averages nearer
4 characters per token), so a prompt the router accepts never passes a TPM limit by accident.
`pick` takes the first non-exhausted model whose TPM is null or `>= prompt_tokens / 0.9`.

## Rate limits and reset rules

- One `RateLimiter` per model: at most `floor(rpm × 0.9)` requests (minimum 1) in any rolling 60
  seconds; `rpm: null` gives 1 request per 10 seconds.
- `QuotaTracker` is one per process (S5), saved atomically to `quota_state.json` (temporary file,
  then rename) on every change, reloaded on start-up with expired entries dropped.
- Gemini models reset at the next midnight in `America/Los_Angeles` (via `zoneinfo`; `tzdata` is
  in the lock through python-dateutil). Groq models reset after the error's retry time when it
  gives one, otherwise 24 hours after detection.

## Waiting (D7)

Every wait is logged as `(model, reason, seconds)` with reason `rate_limiter`, `retry_minute` or
`retry_server`, in a `WaitLog` scoped to one review through a `ContextVar` (`review_waits()`),
so parallel reviews do not mix. **"Rate-limit waiting" in reports is the sum of all three
reasons** (`total_wait_seconds()`), and review time is reported with that sum removed.
