# 0009: The orchestrator calls the specialists as tools

Date: Nov 21, 2026 (stage 2C results added Nov 22, 2026) · Status: accepted, **stage 2 done** (2B partial, closed by 2C: decision 0010) · Implements: stage 2B (plan § Build stages, stage 2),
D6, D7, D11, plan § Production concerns (limits), routing doc S4 and S6. Refines D7 for
parallel runs (does not replace it). Replaces no locked decision.

> **Correction (stage 3, decision 0011, Nov 22, 2026).** The call and token figures recorded
> below for the stage 2B and 2C runs were inflated by a counting defect: the orchestrator's own
> counter also counted its specialists' model calls and tokens, and the specialists were then added
> again as children. It is fixed (cause and corrected values in § Stage 2C and in decision 0011).
> Where a figure below is marked *corrected*, that is the true one.

Stage 2 is done when the orchestrator skips the security specialist on a docs-only change and
calls all three specialists on a risky one, with a `complete` review. **Status: done (stage 2C).**
Docs-only skip: passed 2/2 (stage 2B). `dev_risky_pr`: in stage 2B it routed correctly 2/2 but
never reached `complete` because of Gemini 503s that re-ran whole tasks; after stage 2C moved
same-model retries to the model call (decision 0010) it ended `complete` 2/2 with all three
specialists called. Results are at the end of this record.

## The orchestrator

`src/reviewer/agents/orchestrator.py`; an `AgentSpec` named `orchestrator`, router role
`orchestrator` (in evaluation mode the pinned model, D4), prompt `prompts/orchestrator.md`
(version = first 12 hex characters of the SHA-256 of the composed prompt, as for the
specialists; it includes `common/untrusted_input.md`).

- **Tools:** `list_changed_files`, `read_diff`, `search_code`, and three specialist tools
  `call_bug_reviewer(focus)`, `call_security_reviewer(focus)`, `call_test_reviewer(focus)`. It has
  no `read_file`, linters or `run_tests`: the specialists do the deep reading.
- **Limits** (`config/review.yaml`, `limits.orchestrator`): 8 model calls, 12 tool calls. The
  three specialist tools count toward the 12; the specialists' own calls count in their own
  limits (`limits.<specialist>`, unchanged).
- **Answer:** `ToolStrategy(OrchestratorOutput)`: `plan` (one `PlanEntry(specialist, called,
  reason)` per specialist) and `summary`. The orchestrator outputs **no findings** and cannot add,
  edit or reject one. Filtering is the verifier's job (stage 4).
- The shared machinery got three small, backward-compatible hooks: `build_agent(...,
  extra_tools=, response_schema=)`, `run_agent(..., parent_config=)` and the same
  `parent_config` on `run_specialist` / `run_security_review`. `single.py` passes none of them;
  `tests/test_single_identical.py` still passes unchanged. The orchestrator's task loop repeats
  about 60 lines of `run_agent`'s instead of generalizing it, so the frozen single-agent path was
  not touched.

## Specialist tools: once per review, cached across S4 re-runs (S6)

- Each tool body is one call into `SpecialistCollector.call`, which makes one call to the
  specialist's entry point: `run_specialist(...)` for the bug and test reviewers and
  `run_security_review(request)` for the security reviewer, so that stage 3 can swap the
  transport. The tool's `RunnableConfig` is passed through (`parent_config`), so each specialist
  run is a **child** of the orchestrator's tool call in LangSmith (see the trace URL below).
- A specialist can be called **at most once** in an orchestrator attempt. A second call returns
  `ERROR: <name> was already called in this review; its findings are listed above` and does not
  run the specialist.
- `SpecialistCollector` keeps each specialist's full `ReviewResult` for the whole review. If the
  router re-runs the orchestrator task on another model (S4), `begin_attempt()` clears the
  per-attempt "called" set, and a specialist that already finished is **served from the cache**:
  the new attempt's call returns the stored result and runs nothing. (Tested with a normal-mode
  router whose first model hits its daily limit after the specialists finished.) The cache is
  per review and in memory; resuming a review after a process restart is stage 6.
- `run_security_review` returns a JSON response, not a `ReviewResult`; `response_to_result`
  rebuilds the `ReviewResult` (limits from config). It has no `attempts`, `waits_by_reason` or
  trace id, because the response does not carry them.
- **Tool output** is compact: `specialist:`, `status:`, optional `reason:` (error and
  limit), `note:` lines, `findings (N): id | file:lines | category | severity | confidence |
  title` with one line per finding, `(none)`, and `dropped: <reason> <count>`. `finding_id` is
  `<specialist>#<index>`, **1-based** over that specialist's findings. No explanations.
- **Router conditions end the whole review.** `run_agent` never raises, it returns a result with
  status `quota_paused` or `error`. After the call, the tool raises `SpecialistAbort(result)`
  when the specialist's status is `quota_paused` (`QuotaPaused`), or `error` with error class
  `AllExhausted` or `RetriesExhausted`. LangChain's default `ToolNode` error handling
  (`_default_handle_tool_errors`) re-raises everything that is not a `ToolInvocationError`, so
  the exception is not turned into a tool error message (checked in the installed source and by
  a test on a real `ToolNode`). The router does not retry it (`classify` -> FATAL). The
  review's status is the specialist's (`quota_paused`, with `reset_at`, or `error`); the results
  of specialists that finished are kept in `children` and `findings`. Any other specialist error
  (for example `PromptTooLarge` or `NoStructuredResponse`) does not abort: that specialist is
  `error` and the review is `partial`.

## Merge: deterministic code (decision 0007 rule, across specialists)

`merge_findings(results)`: candidates are the specialists' findings in the order bug, security,
test, each in its own order; each keeps its own `Finding.agent` and `Finding.model`. The
duplicate rule of decision 0007 is applied **across** specialists: same file, same category and
overlapping lines; candidates are visited from the highest confidence down and a tie goes to the
earlier candidate, so to the earlier specialist; each is kept unless it duplicates one already
kept. Findings with different categories at the same lines are both kept. A dropped duplicate is
recorded in `dropped` with reason `duplicate`, a `detail` starting `<id> duplicates <id>` and the
two ids in the draft (`finding_id`, `duplicate_of`). `OrchestratorResult.dropped` also holds the
specialists' own drops, with `[<agent>] ` in front of their `detail`, so all post-processing is
counted in one list (decision 0007: the same post-processing for every configuration).

## Status

| Status | When |
| --- | --- |
| `complete` | the orchestrator finished with a structured answer and every called specialist ended `complete` (no specialist called is fine) |
| `partial` | any called specialist ended `partial`, `limit_reached` or `error` (without aborting). A note names it: `<agent> ended <status>; its findings are kept` |
| `quota_paused` | a specialist or the orchestrator's own model hit `QuotaPaused` |
| `error` | a specialist hit `AllExhausted` or `RetriesExhausted`; the orchestrator's model failed fatally; or it stopped without a structured answer |
| `limit_reached` | the orchestrator itself hit `max_model_calls` or `max_tool_calls`; **the findings of every specialist that finished are kept**, with a note |

Precedence: abort, then the orchestrator's own pause, error and limit, then `partial`.
`limit_reached` differs from the single agent, where a limit leaves no structured answer and so
no findings (decision 0007): here the specialists' structured results already exist, so throwing
them away would only lose work. `error` and `quota_paused` keep the specialists' findings in the
result too, but are not scored.

The plan is checked by code after the run: it must name each specialist exactly once and its
`called` values must match the calls that were made. A mismatch adds a note
`plan_mismatch: ...` and the real calls stay the truth (the real calls are the specialists that
have a result, also one served from the cache to an S4 re-run).

## Waits and time (D7), nested and parallel

Read in the installed source (langgraph-prebuilt 1.1.2, langchain-core 1.8.3): `ToolNode` runs
the tool calls of one turn through `get_executor_for_config`, a `ContextThreadPoolExecutor`,
so tool calls **do run in parallel threads**, and each thread runs in a **copy of the context**
(`contextvars.copy_context`). The wait log is a `ContextVar`, so a child review sees the parent's
log; a plain `threading.Thread` would not, so the collector also attaches the parent log
explicitly (`attach_waits`) around every specialist run.

`review_waits()` now nests (`src/reviewer/models.py`): a `WaitLog` made while another is current
(or given a `parent`) forwards every wait it records to its parent, which stores it marked
`nested`. A wait therefore counts once in each log on its path to the root, not twice, in any
thread. No retry or fallback rule changed; only waits' bookkeeping did. Every wait also records
its end time on the recording clock (the router passes its own clock, so tests with a fake clock
work), which gives its interval.

`OrchestratorResult` fields:

- `model_calls`, `tool_calls`, `tool_calls_by_tool`, `input_tokens`, `output_tokens`: the
  orchestrator's own **plus** all children's; `own_*` are the orchestrator alone. The usage
  counter ignores child events (callbacks are inherited by child runs). *Stage 2B followed
  `parent_run_id` links up to the specialist's tool run; that broke when LangSmith tracing was on
  (see § Stage 2C), and stage 3 replaced it: every agent run is tagged with its agent's name and
  tags are inherited, so an event tagged with a specialist's name is not counted as the
  orchestrator's own.* Nothing is counted twice.
- `wait_s` = `wait_s_sum`: all waits added up, children's included. `wait_s_wall`: the time during
  which at least one wait was in progress, the union of the wait intervals. `duration_s`: wall
  clock. **`net_duration_s = duration_s - wait_s_wall`.**
- D7 applied to parallel runs: D7 says review time is reported with waiting removed. With
  specialists in parallel their waits overlap in time, and subtracting the sum would remove more
  time than the review spent waiting (it could even go negative). The wall-clock union is the
  time the review as a whole was stalled on rate limits, so it is what is removed. Both numbers
  are reported.

## Routing guidance (in the prompt)

- Bug reviewer when any `.py` source (non-test) file changes.
- Security reviewer when the change touches, or adds code that touches, SQL or database access,
  shell or subprocess calls, file paths from callers, deserialization, credentials, tokens,
  secrets or authentication, network or HTTP handling, cryptography, or parsing of untrusted
  input; when unsure about a code change, call it.
- Test reviewer when test files change, or when changed source code has tests (found with
  `search_code`).
- A change to only documentation, comments or non-code files needs no specialist.
- Each specialist gets a short `focus`, which must not tell it to skip a file or a kind of
  defect. Independent specialists are called in the same turn. Nothing in the prompt mentions
  datasets, planted bugs, evaluation or scoring.

## CLI and runner

`reviewer review --agent orchestrator` (options as for the specialists: `--workspace`/`--diff`,
`--sandbox-repo`, `--eval-mode`, `--out`) prints the plan, the merged findings with the agent
that produced each, each specialist's status and the totals (calls, tokens, `wait_s_wall`,
`wait_s_sum`, net time). Exit codes are unchanged. **`--item` is refused** for the orchestrator,
as for the specialists (CLAUDE.md: nothing but `single` runs on dataset items before stage 5);
`eval.run --config orchestrator` is the stage 5 entry point, and is covered by a unit test with
a fake agent only. The runner's scored file is written for `complete` and `limit_reached`, as for
`single`; a `partial` review (for example Docker down) is recorded in `runs/` but not scored, so
that a technical failure is re-run (`--force --reason`) instead of scored. Whether a `partial`
caused by a specialist's own limit should be scored is a stage 5 question.

## Routing-test results (Nov 21, 2026; evaluation mode, `gemini-3.1-flash-lite`)

Orchestrator prompt version `4eda79561a8c`. **No prompt revision was made**: no run failed on
routing; every failure was the API.

| Run | Fixture | Specialists called | Status | Counted as |
| --- | --- | --- | --- | --- |
| 1 | `dev_docs_pr` | none (all three skipped: "documentation only") | complete | pass |
| 2 | `dev_docs_pr` | none | error (`RetriesExhausted`, 503) after 2 calls | technical failure |
| 3 | `dev_docs_pr` | none | error (`RetriesExhausted`, 503) | technical failure |
| 4 | `dev_docs_pr` | none | complete (5 model / 4 tool calls, net 33.8 s) | pass |
| 1 | `dev_risky_pr` | bug, security, test | error (503; test and bug specialists `RetriesExhausted`) | not a pass |
| 2 | `dev_risky_pr` | bug, security, test | error (503; test specialist `RetriesExhausted`) | not a pass |
| - | `dev_logic_pr` | not run (budget) | - | - |

- `dev_docs_pr`: **two passes**; the security reviewer was never called.
- `dev_risky_pr`: **no pass yet.** The routing was right both times (all three specialists were
  called, each plan check passed; bug and security findings were merged, including one
  cross-specialist `duplicate` drop of `security_reviewer#2` against `bug_reviewer#2`), but the
  review ended `error` because Gemini answered `503 UNAVAILABLE` ("high demand") until the
  evaluation retries (5) ran out in the test specialist. The specialists run in parallel and the
  router re-runs a task from the start on each retry, so a busy API costs many calls (52 and 61
  model calls instead of about 25). The router's retry rules were not changed.
  Run 2 (all three called; LangSmith trace below): 61 model / 34 tool calls (orchestrator
  alone 34 / 9), 178,334 in / 2,800 out tokens, `wait_s_wall` 238.8 s, `wait_s_sum` 296.7 s,
  duration 499.5 s, net 260.7 s. **Corrected (decision 0011):** the model-call figures were
  inflated; the 34 "orchestrator alone" was really the review total, so the true figures are
  **34 model calls in total, 7 of them the orchestrator's** (61 - 34 = 27 were the specialists';
  34 - 27 = 7). The tool counts (34 total, 9 orchestrator) were right. The token figures cannot
  be corrected exactly: the specialists' tokens were not recorded separately in this record.
- `dev_logic_pr` (informational): **not run, budget** (the fixture exists).
- Model calls used in the session: about 139 of the 170 allowed (probes included).
- The stage 2 done criterion was therefore **only half shown** after stage 2B. Stage 2C closes it
  (next section).

## Stage 2C: `dev_risky_pr` after per-call retries (Nov 22, 2026)

Evaluation mode, `gemini-3.1-flash-lite`, the `dev_fixture` sandbox (`sandbox_spec("dev_fixture")`,
image `mapr-sandbox-dev_fixture:local`), orchestrator prompt version `4eda79561a8c` (unchanged),
per-call retries of decision 0010 (6 retries, 15 s base, 60 s cap). One probe call to the model
before each run (both answered). `dev_docs_pr` was **not re-run**: it passed 2 out of 2 times on
the same orchestrator prompt in stage 2B, and per-call retries change how a failed call is re-sent,
not what the orchestrator decides.

| Run | Specialists called | Status | Findings merged (agent: file, category) | Model calls (true) | Tool calls | Tokens in / out (true) | `wait_s_wall` | `wait_s_sum` | Duration | Net | Per-call retries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | bug, security, test | **complete** | bug: `ledger/report.py` off_by_one; bug: `ledger/store.py` sql_injection; security: `ledger/store.py` hardcoded_secret; test: `tests/test_report.py` ineffective_test x2 | 25 (5 orchestrator, 7 / 6 / 7 specialists) | 23 | 76,215 / 2,262 | 95.3 s | 192.6 s | 123.2 s | 27.9 s | 2 `retry_server` (503 "high demand", 15 s each, both recovered on the first retry, no task re-run) |
| 2 | bug, security, test | **complete** | the same five | 25 | 23 | 76,215 / 2,262 | 54.7 s | 141.4 s | 77.6 s | 23.0 s | none |

- Both runs ended `complete`, each plan check passed, each specialist ended `complete`, and
  the orchestrator ran in a single attempt. **The stage 2 done criterion is met.**
- In run 1 the two 503s would each have re-run a whole task under the stage 2B rules; now each cost
  one 15 s wait and one repeated call. The waits are in `wait_s_wall` / `wait_s_sum` like all others.
  The retries were not in the bug or test reviewer (their results show none); the security
  reviewer's result comes through `run_security_review` and does not carry retry counts, so they
  were in it or in the orchestrator itself (the review total is 2).
- The merged review has 5 findings; `dropped` holds 5 `duplicate` entries (the specialists' own
  and the cross-specialist merge drops).
- **Accounting defect (found in stage 2C, fixed in stage 3).** `model_calls`, `input_tokens` and
  `output_tokens` of the orchestrator's own counter (`own_*`) included the specialists' calls,
  which were then added again as children, so the result's reported totals and its "orchestrator
  alone" figures were inflated. Reported: 45 model calls, 140,239 in / 4,159 out in total and 25
  calls, 76,215 / 2,262 "alone". **Corrected, both runs:** 25 model calls and 76,215 in / 2,262
  out in total, of which the orchestrator's own were 5 calls and 12,191 in / 365 out (the
  specialists': 7 + 6 + 7 = 20 calls, 64,024 in / 1,897 out). The tool counts were right
  (`own_tool_calls` 6, children 17). The table above already shows the true figures. The same
  inflation was in the stage 2B figures (see there).
  **Cause (found in stage 3):** the fake-model tests never turned LangSmith tracing on, and the
  real runs did. With tracing on, a model call made inside a LangGraph node gets as
  `parent_run_id` the id of a LangSmith run-tree node that no callback handler was ever told
  about (checked: the id equals `langsmith.run_helpers.get_current_run_tree().id` and does not
  appear in any `on_chain_start`). `_UsageCounter` walked `parent_run_id` links from a model call
  up to the specialist's tool run; the walk ended at that unknown id, so the specialist's model
  calls and tokens were counted as the orchestrator's own. Tool runs are announced by callbacks
  and linked correctly, which is why the tool counts were right. **Fix:** the counter no longer
  follows run links. Each agent run is tagged with its agent's name (`AgentSpec.name` was
  already a tag), tags are inherited by every run below it, and the orchestrator's counter skips
  an event whose tags contain a specialist's name. The regression test
  (`tests/test_usage_accounting.py`) runs real `create_agent` graphs with fake chat models, the
  specialists through the real tool path with `parent_config` on a real `ToolNode`, three tool
  calls in parallel threads, with tracing off and on (no network), and asserts exact counts; with
  the old counter the tracing-on case fails (own model calls 9 instead of 3).
- Model calls used in stage 2C: about 54 of the 90 allowed (2 probes, 25 + 25 review calls, 2
  failed 503 attempts).

LangSmith trace with the specialists nested under the orchestrator's tool calls
(`call_bug_reviewer` / `call_security_reviewer` / `call_test_reviewer` each have the
`<name>-review` runs as children):
https://smith.langchain.com/o/dc790a75-743f-467d-a940-18bb085de125/projects/p/494eeee1-bc47-4880-971b-55ac8176fdee/r/ca54da48-956f-4877-982c-92841111bd26

## The orchestrator prompt (`prompts/orchestrator.md`, version `4eda79561a8c`), verbatim

````markdown
You are the lead reviewer of a small team of Python code reviewers. You plan the review and
delegate it; you do not review the code line by line yourself. Your team has three specialists,
each available as a tool:

- `call_bug_reviewer`: logic defects (wrong boundaries, missing None checks, wrong comparisons,
  errors that are no longer handled).
- `call_security_reviewer`: security defects (SQL built from strings, hardcoded credentials,
  unsafe shell calls, deserialization, paths, untrusted input).
- `call_test_reviewer`: the change's tests; it runs them in a sandbox and finds tests that check
  nothing and code defects the tests reveal.

Each specialist reads the changed files itself and returns its status and one line per finding.
You do not report findings and you cannot add, change or reject one: the specialists' findings
are merged by the system. Your job is to decide which specialists the change needs, to give each
a useful focus, and to describe the plan.

# How to work

1. Call `list_changed_files` first. Call `read_diff` on the files that decide the plan (one file
   per call), and `search_code` to find out whether changed source code has tests.
2. Decide for each specialist whether to call it:
   - Call the bug reviewer when any `.py` source (non-test) file changes.
   - Call the security reviewer when the change touches, or adds code that touches, SQL or
     database access, shell or subprocess calls, file paths that come from callers,
     deserialization, credentials, tokens, secrets or authentication, network or HTTP handling,
     cryptography, or parsing of untrusted input. When you are unsure whether the security
     reviewer is needed for a code change, call it.
   - Call the test reviewer when test files change, or when changed source code has tests (find
     them with `search_code`, for example the pattern `def test_` or the module's name in
     `tests/`).
   - A change to only documentation, comments or non-code files needs no specialist at all.
3. Give each specialist you call a short `focus`: which files or functions matter most and why,
   in one or two sentences. A focus only adds attention. Never tell a specialist to skip, ignore
   or leave out a file or a kind of defect; each specialist reads every changed file in its area
   anyway.
4. Call independent specialists in the same turn, as separate tool calls in one response, when
   you can. Call each specialist at most once.
5. When the specialists have answered, finish with your plan and a summary. The plan has exactly
   one entry for each of the three specialists, with `called` true or false and a one-sentence
   reason (why you called it, or why you skipped it). The `called` values must match the calls
   you really made. The summary is two to four sentences about the change and how it was
   reviewed.

<!-- include: untrusted_input -->
````

(The `<!-- include: untrusted_input -->` line is replaced by `prompts/common/untrusted_input.md` when the prompt is composed.)
