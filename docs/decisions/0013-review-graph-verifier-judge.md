# 0013: The review graph, the verifier and the orchestrator as judge (stage 4B)

Date: 2026-11-22 · Status: accepted · Implements: plan stage 4 (second half), D6 (row 4 "plus
verifier" runs on the saved findings of row 3), D11 (the evidence check runs before any judge),
routing doc S6 and S8 for the judge step. Replaces no locked decision. Adds the optional row 4b.

Stage 4 is done when the Task 0 end-to-end run (decision 0012, "End to end again") and the full
graph run below both ended `complete`. **Status: done** (stage 4C, 2026-11-22): Task 0.4 ended `complete` and the full graph run on `dev_multi_pr` ended `complete` (3 of 4 expected defects kept; the cross-file one was not found).

## The review graph (`src/reviewer/graph.py`)

`START -> review -> judge -> present -> END`, a LangGraph `StateGraph` over `ReviewState`
(`schemas.py`, next to `Finding`, which is unchanged), compiled by `build_graph(checkpointer=None,
runtime=None)`.

| Node | Does | Writes (only these keys) |
| --- | --- | --- |
| `review` | `orchestrate(...)` exactly as `--agent orchestrator` does: plan, specialists as tools, spawning (D12), D11, merge | `review_result`, `findings`, `status`, `notes` |
| `judge` | dispatches on `state["judge"]`: `verifier`, `orchestrator` (the judge) or `none` (pass through) | `judge_result`, `final_findings`, `status`, `notes` |
| `present` | code only: sorts by severity (high, medium, low), then confidence (descending), then file and line; builds `approval_payload` | `final_findings`, `approval_payload` |

- **State** is JSON-serializable: findings and both results are stored as `model_dump(mode="json")`
  dicts. Inputs: `review_key`, `pr_title`, `pr_body`, `workspace`, `diff_path`, `sandbox_repo`,
  `security_mode`, `spawn_enabled`, `judge`, `evaluation`. Nothing that cannot be serialized (the
  router, the agent functions, the clock) is in the state: those are `GraphRuntime`, a plain
  dataclass the nodes close over. A node rebuilds its `ReviewContext` from `workspace`,
  `diff_path` and `sandbox_repo`.
- **Idempotent nodes.** Each node reads the state, returns only its own keys and gives the same
  update when run again on the same state. `judge` rebuilds its notes from `review_result`, never
  from the accumulated `notes`, so a resumed run does not add them twice; `present` on its own
  output changes nothing (both asserted in `tests/test_graph.py`).
- **Stage 6.** `build_graph` has a comment where the approval goes: one node after `present`
  (`present -> approve -> post -> END`) whose body calls `interrupt(state["approval_payload"])`,
  and the checkpointer goes to `compile(checkpointer=...)` of this outermost graph only. Nothing
  above changes. The graph already compiles with a checkpointer and `thread_id =
  review_key` (tested with `InMemorySaver`: one checkpoint per node, state readable afterwards).
- **`review_key`** is `<repo or "local">#<pr or item_id>@<head sha or diff sha256[:12]>`
  (`make_review_key`); stage 6 uses it as the `thread_id`. The CLI, with no repository, builds
  `local#adhoc@<12 hex of the diff>`.
- **Status** carries through: `review`'s status is the graph's unless the judge step changes it.
  The judge is skipped when the review ended `quota_paused` or `error` (nothing to judge, note
  added). A judge step that ends `quota_paused`, or `error` with `AllExhausted` or
  `RetriesExhausted`, **aborts** the review with that status (decision 0010: re-run), the findings
  pass through unfiltered. Any other non-`complete` step (a check that hit its limit or failed, a
  judge that hit its limit) makes a `complete` review `partial` with a note and keeps the findings
  the step could not judge.
- **LangSmith.** `run_graph` names the root run `review-graph` (tags `review-graph`,
  `judge:<kind>`, metadata `review_key`). Every agent receives the node's `RunnableConfig` as
  `parent_config` (`orchestrate` got a new optional `parent_config`; nothing else about it
  changed), so the orchestrator, its specialists, the verifier's fan-out and the judge are nested
  in that one run. The regression test enables tracing with a client that makes no network call
  and checks that exactly one run has no parent, that it is `review-graph` and that every other
  run shares its `trace_id`.
- **Approval payload**, one entry per final finding, in order: `id` (a stable id, `F` + the first
  8 hex of a SHA-256 over file, lines, category, agent and title; a repeat gets `-2`), `file`,
  `line_start`, `line_end`, `category`, `severity`, `confidence`, `title`, `explanation`,
  `evidence`, `agent`, `verdict` (`unchecked` when the judge is `none`) and `verdict_reason`.

## Verdicts and the keep-or-drop policy (both judges)

Neither judge edits a finding, adds one or re-categorizes one. They give each finding one verdict:

- `confirmed`: the code shows the defect as described, or clearly the same defect.
- `rejected`: the code handles the case, the claim misreads the code, or the described behaviour
  cannot happen.
- `uncertain`: it cannot tell within its budget.
- It judges whether the defect is real, not whether the category label is perfect.

The definitions are the same text in `prompts/verifier.md` and `prompts/common/judge.md` (a test
compares them). **Policy:** keep `confirmed` and `uncertain`; drop `rejected` (`Dropped` with the
reason `verifier_rejected` or `judge_rejected` and the verdict's reason as `detail`; both are new
`DropReason` values). A finding keeps its `agent` and `model`. A check that ends `limit_reached` or
in a non-infrastructure `error` counts as `uncertain` (kept, with a note, `source` says why);
`QuotaPaused`, `AllExhausted` and `RetriesExhausted` end the step. D11 runs before: a judge never
sees a finding whose evidence check failed.

## The verifier (`agents/verifier.py`)

- **One check per finding**, in parallel: a small `StateGraph` sends one `Send("verify", ...)` per
  finding, at most `verifier.max_concurrency` (3) at a time (the sub-agent pattern of decision
  0012). Each check is a `create_agent` with the tools `read_diff`, `read_file`, `search_code`,
  `limits.verifier_per_finding` (6 model calls, 8 tool calls) and `ToolStrategy(Verdict)`
  (`verdict`, `reason`). Router role `verifier` (normal chain: `openai/gpt-oss-120b`,
  `qwen/qwen3.8-27b`, `gemini-3.6-flash`; the router's TPM check keeps a prompt that does not fit
  Groq's 8K off it, with 4,000 tokens of headroom added to the estimate for tool output); in
  evaluation mode the pinned model (D4).
- **Prompt** `prompts/verifier.md`, version `4fb90e13fd02`: an independent checker; the finding is
  a claim to check, not a fact; read the code at the stated lines and around them, look for guards
  elsewhere (callers, callees, validation) and decide whether the failure can happen; the verdict
  definitions above; the shared untrusted-input fragment; nothing about datasets, planted bugs,
  evaluation or scoring (a test checks the words). The user message holds the finding's file,
  lines, category, severity, title, explanation and evidence, marked as an untrusted claim, and the
  PR title (also untrusted). It does **not** say which agent produced the finding and carries no
  finding id.
- Records `verifications: [{finding_ref, verdict, reason, source, model_id, calls, tokens}]`.

## The orchestrator as judge (`agents/judge.py`, optional row 4b)

The lead reviewer re-judging its team. It **is** the orchestrator: router role `orchestrator`,
`orchestrator.md` unchanged plus the new fragment `prompts/common/judge.md` at the end (composed
version `ec54742e1b4c`), its own tools `list_changed_files`, `read_diff`, `search_code` (**not**
`read_file`) and no specialist tools. One agent run covers all findings (`limits.judge`: 8 model
calls, 12 tool calls), ending in `ToolStrategy(JudgeOutput)`: `verdicts: [{finding_id, verdict,
reason}]`. The fragment says the section takes precedence over the planning instructions above and
that no specialist is called now. The user message holds the review's saved plan and summary and
every merged finding with an id and **the agent that found it**.

How it differs from the verifier, on purpose, because that is the comparison: one run for all
findings against one small run each; the orchestrator's read-only tools (no `read_file`) against
the verifier's; it knows the plan and which specialist found what; the orchestrator's model chain
(`gemini-3.8-flash` first) against the verifier's. Code checks the answer: a finding with no valid
verdict (absent, or its id given twice, in which case **both** verdicts are ignored) counts as
`uncertain`, kept, with a note; an id that is not in the list is noted and ignored. The step stays
`complete` in these cases (a `partial` with no failed child would be `rerun_required` under
decision 0010, which would punish the model for an omission, not an infrastructure problem).

## Evaluation (`eval/run.py`)

- `--config verifier` (row 4) and `--config judge` (row 4b) read the saved `eval/results/subagents/`
  results: the scored findings and, from the run file, the plan and summary. For each item they
  materialize the workspace, run **only** the judge step on those findings and write the kept
  findings in the scorer's format to `eval/results/<config>/`. An item with no findings makes no
  model call. Missing results: exit 2, "run --config subagents first". `eval/score.py` is
  untouched.
- The run file records the step's calls, tokens and time (`result`) **and** `review_cost` (the saved
  row 3 review) and `cumulative` (the sum: calls, tokens, duration, summed and wall-clock waits and
  net time; the two run one after the other, so every figure is a plain sum). `summary.md` reports
  the judge step alone ("Cost and time") and the cumulative row, the verdict counts and the number
  dropped.
- Scoring policy (decision 0010) applies to the judge step: `RetriesExhausted`, `AllExhausted` or
  `quota_paused` means `rerun_required` (no scored file, non-zero exit); a check that hit a limit or
  failed otherwise is scored (the step is `partial` with the failed child).
- Stage 5 run order: `single` (done) -> `orchestrator` -> `subagents` -> `verifier` -> optionally
  `judge`.

## Results of the dev runs

- **Task 0.4 end to end** (orchestrator, remote security, spawning on): `complete`; see decision 0012,
  "End to end again" (66 model calls, hierarchy table, run id `0658f63f-e965-46e8-adb1-8fe0ff5941b6`).
- **Judge mix, verifier half** (`scripts/check_judges.py`, `tests/fixtures/dev_pr/judge_mix.json`,
  evaluation mode, status `complete`, 15 model calls / 11 tool calls, 31,211 in / 548 out tokens,
  10.5 s net + 52.8 s waiting, prompt `4fb90e13fd02`, informational):

  | Finding | Really | Verdict | Reason |
  | --- | --- | --- | --- |
  | `catalog.py:36` off_by_one | real | confirmed | the slice end `start + size - 1` yields `size - 1` items |
  | `cart.py:22-23` missing_none_check | real | confirmed | `coupon.percent` read without a None check, contradicting the docstring |
  | `cart.py:18` missing_none_check | wrong | rejected | line 16 already raises `KeyError` when `find(sku)` is None |
  | `test_cart.py:29-30` ineffective_test | wrong | rejected | the test asserts the total is 360, which verifies the discount |

  Kept 2, dropped 2; no prompt revision was made.
- **Judge mix, orchestrator-as-judge half** (stage 4C, 2026-11-22, evaluation mode, status `complete`,
  5 model calls / 4 tool calls, 14,725 in / 403 out tokens, 12.0 s net + 0.0 s waiting, prompt
  `ec54742e1b4c`, trace `930ee272-c7b4-4273-8142-040773bf5380`). Same four findings, next to the
  verifier's verdicts above:

  | Finding | Really | Verifier | Judge (orchestrator) |
  | --- | --- | --- | --- |
  | `catalog.py:36` off_by_one | real | confirmed: slice end `start + size - 1` yields `size - 1` items | confirmed: the slice excludes the last element of the page |
  | `cart.py:22-23` missing_none_check | real | confirmed: `coupon.percent` read without a None check | confirmed: `find(code)` can return `None`, then `.percent` raises `AttributeError` |
  | `cart.py:18` missing_none_check | wrong | rejected: line 16 already raises `KeyError` | rejected: `add` already checks `self.catalog.find(sku) is None` |
  | `test_cart.py:29-30` ineffective_test | wrong | rejected: the test asserts 360, verifying the discount | rejected: the test asserts the total is 360, so it is effective |

  Both judges kept 2 and dropped 2, the same four verdicts. Verifier: 15 model calls, 31,211 in / 548 out
  tokens (4 checks). Judge: 5 model calls, 14,725 in / 403 out tokens (one run). No prompt revision.
- **Full graph on `dev_multi_pr`** (stage 4C; `--judge verifier`, remote security, spawning on,
  `--sandbox-repo dev_fixture`, evaluation mode): status **`complete`**, root run `review-graph`
  `25f148be-1a90-475f-bd8a-d9c0d83978e9`. See "Full graph run" below.
- **Accounting regression** (`tests/test_graph.py`, both judges, tracing off and on): passes. Review stage
  9 model / 7 tool calls; verifier step 4 / 2 (two checks of 2 calls); judge step 2 / 1; totals are the plain
  sum; the step's summed wait covers its children; one root run `review-graph` and one `trace_id`.

## Full graph run (stage 4C)

Hierarchy (model calls / tool calls, tokens in / out, findings):

| Agent | Status | Calls | Tokens | Findings |
| --- | --- | --- | --- | --- |
| orchestrator (review stage total) | complete | 59 / 50 | 270,204 / 3,823 | 3 after merge |
| &nbsp;&nbsp;bug_reviewer | complete | 17 / 14 | 75,195 / 1,515 | 3 |
| &nbsp;&nbsp;&nbsp;&nbsp;bug_reviewer/sub1 | complete | 5 / 4 | 20,121 / 397 | 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;bug_reviewer/sub2 | complete | 8 / 7 | 39,955 / 504 | 1 |
| &nbsp;&nbsp;security_reviewer (remote A2A) | complete | 13 / 10 | 50,516 / 718 | 1 |
| &nbsp;&nbsp;&nbsp;&nbsp;security_reviewer/sub1 | complete | 5 / 4 | 20,631 / 166 | 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;security_reviewer/sub2 | complete | 5 / 4 | 19,770 / 310 | 1 |
| &nbsp;&nbsp;test_reviewer | complete | 23 / 19 | 121,849 / 1,196 | 2 |
| &nbsp;&nbsp;&nbsp;&nbsp;test_reviewer/sub1 | complete | 6 / 5 | 27,415 / 328 | 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;test_reviewer/sub2 | complete | 10 / 8 | 57,353 / 531 | 1 |
| verifier (3 checks: 3/2, 2/1, 3/2) | complete | 8 / 5 | 18,269 / 381 | 3 checked |

All three specialists spawned two sub-agents each (batches as expected in `expected.json`).

Stage totals: review 59 model / 50 tool, 270,204 in / 3,823 out, 54.3 s net, waits 207.2 s wall-clock
(733.2 s summed), 261.5 s total; judge (verifier step) 8 / 5, 18,269 in / 381 out, 8.4 s net, waits
3.5 s wall (3.9 s summed), 11.9 s total; **total 67 model / 55 tool calls, 288,473 in / 4,204 out,
62.7 s net, waits 210.7 s wall-clock (737.1 s summed), 273.5 s**.

Verdicts: all 3 findings `confirmed`; **dropped by the verifier: 0**. Before the verifier, the
orchestrator's merge and evidence checks dropped (code, not judgement): evidence mismatches in
`bug_reviewer/sub1` and `test_reviewer/sub1`, one `not_assigned` (`bug_reviewer/sub2` reported on
`pricing.py`, not its file), and 5 duplicates (security `sql_injection` at `stock.py:79`, the test
reviewer's two findings, the bug reviewer's own `stock.py:25`, and the security sub-agent's own
duplicate).

Expected defects (`tests/fixtures/dev_multi_pr/expected.json`, informational):

| Defect | Found | By |
| --- | --- | --- |
| `pricing.py:23` bulk threshold (`wrong_comparison`; reported as `off_by_one`) | yes | `bug_reviewer` (own cross-file pass; also `test_reviewer`) |
| `stock.py:25` `last_movements` slice | yes | `bug_reviewer/sub2` (also `test_reviewer/sub2`) |
| `stock.py:79` SQL injection | yes | `bug_reviewer`, also `security_reviewer/sub2` |
| `reports.py:15-17` cross-file `None` from `unit_price` | **no** | nobody |

The kept payload therefore has 3 of 4 expected defects; the cross-file one was not found.

## Limitations and interpretations

- `uncertain` keeps the finding, so a verifier that cannot tell never lowers recall; the cost is
  precision. That is the policy that was asked for, not a tuned one.
- The verifier's per-check prompt estimate leaves out the diff (the checker reads files through
  tools); a `read_file` of a large file can still exceed Groq's TPM, in which case the router's
  TOO_BIG handling falls over to the next model in normal mode.
- Ids are content hashes: two runs that find the same defect with a different title get different
  ids. They identify a finding inside one review, not across reviews.
- The CLI's graph run has no repository or PR number (`local#adhoc@...`); stage 6 fills those in.
