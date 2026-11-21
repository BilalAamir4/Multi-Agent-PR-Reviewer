# 0012: Specialists spawn their own sub-agents (stage 4A)

Status: accepted, 2026-11-22. **Replaces D6 and D12** of `0001-locked-decisions.md`. D1 to D5,
D7 to D11 are unchanged.

## What changes and why

- **Old D6** (the orchestrator spawns extra workers per file) and **old D12** (its spawn limit)
  put the spawning above the specialists, where the orchestrator knows only the plan, not how much
  each specialist has to read. A large share is a problem of one specialist's prompt, so the
  specialist is the one that should split it. Sub-agents report to their specialist, which keeps
  their findings and adds its own cross-file findings.
- **New D6.** A specialist `S` spawns sub-agents over batches of its own files when its share is
  large. The "plus spawned sub-agents" configuration (row 3 of the results table) is the
  orchestrator with spawning on. **Reuse rule:** for an item where no specialist the orchestrator
  called would spawn, row 3 is the same review as row 2, so row 2's scored result and run file are
  copied (`reused_from: "orchestrator"`); otherwise a full review runs with spawning on.
- **New D12.** Spawn limits live in `config/review.yaml` (`spawn:` and `limits.subagent`) and are
  tested in `tests/test_config.py`. Row 2 runs with spawning **off**, row 3 with it on.
- Stage 4 is split: **4A** (this record) is specialist-level spawning, the switch, the reuse rule
  and the timing criterion; **4B** (later) is the StateGraph around the review and the verifier.

## When a specialist spawns (`should_spawn_for`)

The specialist's **area** is the Python files of the diff it reads: bug and security take every
changed `.py` file that is not deleted, the test reviewer takes test and source files alike (one
area function, `specialist_area`; test paths use `eval.find_candidates.is_test_path`). It spawns
when **either** trigger holds: the area has at least `min_py_files` (4) files (`files`), or the
estimated prompt for the whole area is above `max_specialist_prompt_tokens` (30000) (`tokens`).
Both can fire. `spawn.enabled: false` (or `--no-spawn`, or `spawn_enabled: false` in an A2A
request) turns it off for the whole review; behaviour is then identical to stage 3 (the
`run_agent` call is the same, `tests/test_subagents.py::test_spawn_off_*`).

## Batches (`make_batches`)

1. Pair a test file with a source file by stem, only when exactly one source file matches.
2. Group the rest by parent directory.
3. Pack groups in `(directory, first path)` order into batches of at most `max_files_per_subagent`
   (3) files and `batch_token_budget` (20000) estimated tokens; an oversized group is split, a
   single oversized file stands alone.
4. At most `max_subagents` (5) batches: when over, raise the file limit by 1 until it fits and
   record `file_limit_used` and `budget_exceeded`. If the file limit already covers all files and
   the token budget alone binds, the budget is ignored (`token_budget_ignored`) so the loop ends.

Deterministic and independent of input order. `dev_multi_pr` gives batch 1
`inventory/pricing.py + tests/test_pricing.py` and batch 2 `inventory/reports.py +
inventory/stock.py + tests/test_integration.py`.

## The sub-agent `S/sub<k>`

- Router role `worker`, limits `limits.subagent` (10 model calls, 15 tool calls, 1 `run_tests`),
  the **same tools as `S`**. Prompt: `S`'s composed prompt unchanged **plus**
  `prompts/common/subagent_assignment.md` (names the assigned files; takes precedence over "read
  every changed file"). No existing prompt file changed (hashes in `tests/test_subagents.py`).
- Findings outside the assigned files are dropped in code with reason `not_assigned`.
- `QuotaPaused`, `AllExhausted` and `RetriesExhausted` in any sub-agent end the whole review (the
  orchestrator's `SpecialistCollector` sees the specialist's status and stops). Any other
  sub-agent failure stays as that child's status; the review goes on with the others.

## Fan-out and the specialist's own pass

- Fan-out is a LangGraph `StateGraph` with `Send` (one task per batch), invoked with
  `max_concurrency` and results gathered through an `operator.add` list. A `RunnableLambda`
  wrapper named `<S>-review` parents the fan-out and the cross-file pass so LangSmith nesting
  works with the specialist's own `parent_config`.
- Verified in the installed langgraph 1.2.22 / langchain-core 1.8.3: sync `Send` tasks run in a
  thread pool sized by `max_concurrency` (`langgraph/pregel/_executor.py`
  `BackgroundExecutor.__init__` enters `get_executor_for_config`, and
  `langchain_core/runnables/config.py` `get_executor_for_config` builds
  `ContextThreadPoolExecutor(max_workers=config["max_concurrency"])`; the async executor uses a
  semaphore), and each task runs in `copy_context().run`, so contextvars (the wait log, the
  usage counter) reach the node threads. The node strips `max_concurrency` from its child config.
- After the fan-out, `S`'s own agent runs with its prompt unchanged **plus**
  `prompts/common/coordination.md` (cross-file defects, no repeats, no need to re-read all
  files). Its user message adds the sub-agents' compact finding lines and summaries.
- **Merge** in code with the existing duplicate rule (`reports.merge_candidates`): higher
  confidence wins; ties go to `S/sub1`, `S/sub2`, ... then `S`. Every finding keeps its `agent`.
- Result: `children`, `spawn` (`SpawnInfo`: decision, batches, per-sub-agent start and end, fan-out
  wall, wait wall and sum, net, cross-file pass), `own_*` counters (the cross-file pass alone),
  totals counted once, `prompt_version` of the composed prompt used.
- Counting: sub-agent runs inherit the specialist's name tag from the wrapper run, so the
  orchestrator's tag-based counter excludes them; the specialist adds them from the children's
  results. Regression-tested through the real callback flow with parallel `Send`, tracing on and
  off.
- **Waits (D7).** Each sub-agent's waits are in its own log and joined into the fan-out log and
  the specialist's; `wait_s_sum` adds them, `wait_s_wall` is the union of intervals. Net time is
  total minus the wall-clock union, so parallel waits are removed once.
- **Status.** `complete` only if the specialist's cross-file pass and every sub-agent are
  complete. Otherwise `partial` (a sub-agent failed, findings kept), or the cross-file pass's own
  `limit_reached` or `error`. On an abort the result takes the aborting child's status.

## Scoring (extends decision 0010)

`decide_scoring` looks through the whole tree. A `partial` specialist is **scored** when every
non-complete sub-agent ended `limit_reached`, a non-infrastructure `error`, or a `partial` that
is itself explained that way; it is `rerun_required` when any is a `partial` with no recorded
cause, or has `sandbox unavailable`, `quota_paused`, `RetriesExhausted`, `AllExhausted` or a
remote failure anywhere below.

## A2A (extends decision 0011)

`protocol_version` is `"2"`. The request gains `spawn_enabled`; the service rejects `"1"` with
"upgraded together (decision 0012)", applies `spawn_enabled` together with its own config
(`spawn.enabled` there is ANDed in: a false request turns spawning off), uses **its own**
`max_concurrency` and quota state, and returns `children`, `spawn`, `own` and the wait intervals
(sub-agents' included, each once). Parity test with sub-agents: scripted model, one sub-agent at
a time, `serve` versus `run_security_review`, equal after stripping timing keys at any depth.

## Evaluation configurations

- `eval.run --config orchestrator` forces spawning off and records it (run file
  `spawn_enabled: false`, summary line).
- `eval.run --config subagents` applies the reuse rule per item; missing orchestrator results end
  the run with exit 2 and "run --config orchestrator first". `summary.md` lists reused and run
  items; the run log of a reused item has `reused_from`.
- No evaluation item has been run for either configuration yet (stage 5).

## Timing criterion (live, dev fixture `dev_multi_pr`, `bug_reviewer` alone, evaluation mode)

| Run | Fan-out wall | Wait (wall = sum) | Fan-out net | Cross-file net | Whole review net | Model calls |
| --- | --- | --- | --- | --- | --- | --- |
| concurrency 3 | 62.2 s | 50.2 s | **12.1 s** | 9.5 s | 21.6 s | 21 |
| concurrency 1 | 64.9 s | 39.2 s | **25.7 s** | 8.3 s | 34.0 s | 21 |

Both ended `complete`; parallel net is about half of sequential, far outside the 10% band, so the
criterion holds and no repeat was needed. Sub-agent intervals (parallel): sub1 0 to 6.3 s, sub2 0
to 62.2 s (its wait); sequential: sub1 0 to 7.0 s, sub2 7.0 to 64.9 s. Wall clock hardly differs
because the free-tier limiter dominates; net time (D7) is the comparison.

## End to end (orchestrator, `--security-mode remote --sandbox-repo dev_fixture`)

The run ended **`partial`**, not `complete` (the requirement). Cause: the orchestrator planned
`security_reviewer: skipped` for a diff with no security-sensitive change (the plan is the
model's choice and the prompt is frozen), and `test_reviewer/sub2` hit `max_model_calls`
(`limit_reached`; its findings kept). `bug_reviewer` and `test_reviewer` both spawned 2
sub-agents; 44 model calls, 38 tool calls. The security sub-agents were therefore exercised
separately, `--agent security_reviewer --security-mode remote` on the same fixture: `complete`,
2 sub-agents **inside the service** returned as `children` through the wire (14 model calls).
Not re-run to avoid spending the rest of the 150-call budget. Run ids are in the run output;
LangSmith could not be read back (the key returned 401 on read), so the UI nesting was not
inspected, only the recorded run ids per agent.

Defects found on `dev_multi_pr` (informational): wrong_comparison at `pricing.py:23` found (labelled
`off_by_one`; bug and security runs); cross-file `missing_none_check` found by both
`bug_reviewer` alone runs; `stock.py:25` `off_by_one` found only by
the security sub-agent run. No prompt was changed after seeing these results.

## Limitations and interpretations

- `own_*` counters are 0 on a result that did not spawn; `spawn` is `None` when spawning is off
  and `spawned: false` when on but not triggered.
- Over the service the request carries only `spawn_enabled`; concurrency is the service's.
- Plan text outside Appendix A (intro "spawns extra workers", skills table "one per file", stack
  row "Send ... one worker per file") was left as it is.
- Tokens are estimates (`estimate_tokens`); batches are by files and tokens only, not by
  language-aware analysis.
