# 0011: The security specialist as an A2A service (stage 3)

Status: accepted, Nov 22, 2026. Replaces no D-decision. Changes the stage 3 wording of the plan
(parity data, see "Parity").

## Why A2A

The orchestrator reaches the security specialist over the A2A protocol served by the LangGraph
Agent Server (`langgraph dev`, langgraph-api 0.18.3), so the specialists can later run as
independent services. No `langgraph-supervisor`. The local specialist stays, behind
`a2a.security_reviewer.mode: local|remote` in `config/review.yaml` (default `local`) and
`--security-mode` on the CLI. `call_security_reviewer` keeps its name, signature, once-per-review
rule and cache; the orchestrator prompt (`4eda79561a8c`) and tool docstring are unchanged.
Specialist prompt versions are unchanged: bug `0a0cf383d913`, security `6f396484c1bb` (equal to
decision 0008), test `c7c71887b22e`; single `aa94f1ae2cf6`.

## How langgraph-api's A2A works (differences from the prompt's assumptions)

`POST /a2a/{assistant_id}`, JSON-RPC 2.0. The legacy method `message/send` is aliased to
`SendMessage`; its result is the task itself (`state: "completed"`, `artifacts[].parts[]` with
`kind: "text"`); the v1 name wraps it as `result.task` with upper-case states. The client accepts
both. Each text part becomes one human message `{"role": "human", "content": text}` in
`messages`; a `data` part is merged into top-level input keys, not messages. The graph's last AI
message is returned as the artifact. `configuration.historyLength: 0` suppresses the history echo.
The in-memory server keeps request content in its thread state, and runs 1 background worker by
default, so the service handles one review at a time.

## Request and response

Request: one JSON text part, `SecurityReviewRequest` (`src/reviewer/a2a_schema.py`, strict,
`extra="forbid"`): `protocol_version "1"`, `pr_title`, `pr_body`, `focus`, `diff_text`,
`workspace_archive` (base64 tar.gz), `workspace_sha256`, `evaluation`, optional `trace_parent`
(`langsmith-trace`, `baggage` only). Response: the JSON text of the AI message,
`SecurityReviewResponse`: the `run_security_review` response plus `retries_by_reason`,
`waits_by_reason`, `wait_intervals` (objects `{start, end, reason}`, Unix seconds) and
`service {name, pid, version}`. An invalid request, bad archive or internal failure answers with
`status: "error"` and `error_class` `InvalidRequest`, `ArchiveError` or `InternalError`; the
service never raises.

## Snapshot transport and safety

The code travels in the request (a deterministic tar.gz of the files collected as `run_tests`
collects them: no symlinks, no file over 1 MB). The service unpacks it into a temp directory,
builds a `ReviewContext` without a sandbox, and always deletes the directory. It never reads the
caller's filesystem. `snapshot.py` checks the SHA-256 and unpacks member by member (never
`extractall`), rejecting absolute or non-canonical names, `..`, backslashes, drive letters,
Windows reserved names, symlinks, hard links, devices and duplicates. Limits: 50 MB compressed,
200 MB unpacked, 20,000 entries. The security specialist has no `run_tests`, so no sandbox is
needed in the service.

## Parity

Plan wording changed (user decision): parity uses the dev fixtures `dev_security_pr`,
`dev_risky_pr`, `dev_pr`, not dataset items (which may not be run before stage 5).

1. Deterministic: the same scripted fake model runs locally and through the real HTTP path
   (`tests/test_integration_a2a.py`, `langgraph dev` on port 2026). Responses are identical after
   removing `service`, timings, waits and PIDs. The fake model is read only when
   `REVIEWER_TEST_MODE=1` and `REVIEWER_TEST_FAKE_MODEL` is set; the variable without the mode is
   rejected (`TestHookRejected`, the service fails to start; tested). Archive safety (tampered SHA,
   traversal, symlink, oversized, invalid request) is tested in `tests/test_a2a_service.py`.
2. Live (`scripts/check_a2a_parity.py`, `docs/a2a-parity.json`): per fixture, local twice and
   remote once with a fixed focus, on `gemini-3.1-flash-lite`, evaluation mode. Findings are
   compared as sets of `(file, category, overlapping line ranges)`; a difference counts findings
   without a partner (maximum matching). **Pass rule:** remote matches at least one local run
   exactly, or its difference from the nearest local run is no larger than the difference between
   the two local runs.

Result: all three fixtures passed with exact matches (difference 0 everywhere, all runs
`complete`, equal model calls: 6, 5, 7 per run).

## Quota state, rate limits, waits

The service has its own `QuotaRouter`, `RateLimiter` and `QuotaTracker` (`quota_state.a2a.json`,
gitignored). **Limitation:** the orchestrator process and the service do not share limiter or
quota state, so their request rates add up and the sum can exceed a model's RPM/RPD limit; the
per-call retries (decision 0010) absorb 429s on each side.

D7 across processes: the service returns its wait intervals as Unix timestamps. The orchestrator
maps them onto its own clock (`offset = clock() - time.time()`, **assumption: both processes run
on the same machine and clock**) and records them as nested waits, so they join the interval union
(`wait_s_wall`) and the sum (`wait_s_sum`). The remote child's retries, calls and tokens are
counted once, as a child.

## Failures

`RemoteUnavailable("security service is not running at <url>")` makes the security child `error`
with the note `security service unavailable: <url>`; `RemoteError` (HTTP status, JSON-RPC error,
failed task, invalid response, timeout) is an error child too. Both give `rerun_required`
(decision 0010, updated). Default timeout 900 s in evaluation mode, 300 s otherwise.

## Security of the service

127.0.0.1 only, **no authentication** (a limitation: any local process can call it, and request
content, including the code, is held in the server's in-memory threads). Nothing under
`a2a_security/` or in the client reads or logs secrets: the service gets keys through the `env`
file named in `langgraph.json` (`.env`), and traces carry no keys.

## Tracing

The client sends `RunTree.to_headers()` as `trace_parent`; the service opens the review under
`tracing_context(parent=...)` and also records `orchestrator_trace_id` and `a2a_pid` as run
metadata. Live result: the service's review is **linked, not nested**: it is its own trace
(`01a1ffbd-e241-70af-8517-35ac3c81c15a`) with `orchestrator_trace_id` =
`a260a738-51e3-4e08-ad9d-d22e736c7bd5` (the orchestrator's trace) on its runs. Orchestrator trace:
https://smith.langchain.com/o/dc790a75-743f-467d-a940-18bb085de125/projects/p/494eeee1-bc47-4880-971b-55ac8176fdee/r/a260a738-51e3-4e08-ad9d-d22e736c7bd5

## End to end on `dev_risky_pr` (`--sandbox-repo dev_fixture`)

| | remote | local |
| --- | --- | --- |
| status | complete | complete |
| specialists | bug, security, test | bug, security, test |
| security transport | a2a (pid 24252, 4,443 request bytes, round trip 57.7 s) | local |
| model / tool calls | 27 / 25 (orchestrator 6 / 7) | 27 / 25 (6 / 7) |
| tokens in / out | 81,433 / 2,341 | 81,433 / 2,341 |
| `wait_s_wall` / `wait_s_sum` | 88.4 s / 181.3 s | 102.4 s / 206.4 s |
| net / total time | 31.0 s / 119.4 s | 19.8 s / 122.2 s |

Stage 3 is done: deterministic parity passed, live parity passed on all three fixtures, and the
remote end-to-end run ended `complete`.

## Related fix: usage counting (task 0)

See decision 0009 (§ Stage 2C): the orchestrator's `own_*` counters included the children's model
calls when LangSmith tracing was on. Fixed by attributing events by inherited agent-name tags;
regression test `tests/test_usage_accounting.py`. Figures in 0009 and 0010 are corrected.
