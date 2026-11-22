# 0015: Evaluation protocol (stage 5)

Date: 2026-11-22 · Status: accepted · Implements: plan § Evaluation, D4 to D7, D9 to D9b, D12.
Replaces no locked decision. Written and committed **before the first stage 5 run**; nothing in it
may change because of a result.

## What is evaluated

Five configurations on the 20 items of `eval/dataset/` (14 planted, 6 clean; 28 planted bugs), one
run each (D5):

| # | Config (`eval.run --config`) | What runs | Spawning |
| --- | --- | --- | --- |
| 1 | `single` | the one agent, `prompts/single.md`, `limits.single` | n/a |
| 2 | `orchestrator` | orchestrator calling the three specialists | **off** (forced, decision 0012) |
| 3 | `subagents` | the same, specialists spawn sub-agents (D12) | **on** (forced) |
| 4 | `verifier` | the verifier on the saved `subagents` findings (decision 0013) | as saved |
| 4b | `judge` | optional: the orchestrator as judge on the same saved findings | as saved |

### Run order and D6 reuse

`single` (the 15 new items) → `orchestrator` (20 items) → `subagents` → `verifier` → `judge` (only if
quota allows). Items run one at a time, in item-id order.

- **`subagents` reuse (D6, decision 0012).** For each item, if none of the specialists the saved
  `orchestrator` run called would spawn (`should_spawn_for`, D12 as fixed by stage 4C), the saved
  `orchestrator` result is copied, marked `reused_from: "orchestrator"`. Otherwise a full review runs
  with spawning on. By decision 0012 "Stage 4C", six items do not spawn (item_001, 002, 011, 012,
  019, 020), so up to 6 items are reused and 14 are run. If the orchestrator did not call a
  specialist on an item, that specialist is not counted (so more items could be reused; whatever
  `eval.run` decides is what stands, and the reused list is reported).
- **`verifier` and `judge`** run only the judge step on the saved `subagents` findings (including
  the reused items); they never re-review. Their cost in the table is cumulative (the `subagents`
  review plus the step).
- **Quota.** About 1,900 model calls in all at about 500 a day: stage 5 is split into sessions
  (5A, 5-continue, 5Z). Runs stop cleanly on the budget guard or `QuotaPaused` and resume the next
  day; items with a scored result are skipped on resume.

## Fixed settings

- **Model (D4, S9).** Every role uses `gemini-3.1-flash-lite` (`evaluation.model` in
  `config/models.yaml`), `QuotaRouter(evaluation=True)`: no cross-model fallback. On `QuotaPaused`
  the runner saves its progress and stops; it is not worked around.
- **Security specialist: `security_mode: local`** (`a2a.security_reviewer.mode` in
  `config/review.yaml`). Decision of the project owner. Stage 3 (decision 0011, `docs/a2a-parity.json`)
  showed parity between the local and the remote specialist, so A2A is a transport and not a review
  design being compared, and `local` avoids reviews having to be re-run because the service was
  down. Every runs file records `security_mode`, `security_transport` and
  `security_transport_detail`.
- **Docker Desktop must be running** (the test specialist runs `pytest` in the sandbox, D2). A
  review that ends `partial` because the sandbox was unavailable is `rerun_required` (decision 0010).

## What is frozen from this record on

Frozen (a change needs a new decision record, a re-run of every affected item in every affected
configuration, and the project owner's agreement):

- every prompt and fragment (`src/reviewer/prompts/**`), limits, tool sets, spawn, verifier and judge
  settings (`config/review.yaml`, `config/models.yaml`), and all of `src/`;
- `eval/score.py`, `eval/ground_truth.py`, `eval/equivalences.yaml`, `eval/dataset/`, `eval/plants/`;
- the router rules (decisions 0006, 0010) and every D-decision.

Tooling may be added (run planner, call-budget guard in `eval/run.py`, LangSmith experiments, the
labelling CLI, the report generator) but must not change review or scoring behaviour.

**Verify with git.** The code the runs start from is commit `cb12ec8` (stage 4C). The frozen paths
have these tree or blob hashes at that commit, and
`git diff cb12ec8 -- src config eval/score.py eval/ground_truth.py eval/equivalences.yaml eval/dataset eval/plants`
must stay empty for the whole evaluation:

| Path | Git object at `cb12ec8` |
| --- | --- |
| `src/` | `af22eb9485245af20de29e2cd2ea58f521c6896a` |
| `src/reviewer/prompts/` | `fae2ad4dae4e8a03b542f5fea8a568bb744c13dd` |
| `config/` | `f2570a668a2bb18e93d815675f12bcff47a0db17` |
| `eval/dataset/` | `ff09146600dcde828dc5d520ac06ffa986fce616` |
| `eval/plants/` | `b926afbf0005f38b9c51b6f10f5fd46cba926cb9` |
| `eval/score.py` | `3e0e80a62ecc16e67bb7f36960db881b2409a3fb` |
| `eval/ground_truth.py` | `3f9e4f2e26b4d813388b52925168e01fbca0276c` |
| `eval/equivalences.yaml` | `f0d17c5d4270d888a800814b512ced7301913905` |

### Prompt hashes

Composed system prompt, first 12 hex of SHA-256 (`AgentSpec.prompt_version()`; these are the values
recorded in every result):

| Agent | Hash |
| --- | --- |
| `single` (`single.md`, file bytes) | `aa94f1ae2cf6` |
| `bug_reviewer` | `0a0cf383d913` |
| `security_reviewer` | `6f396484c1bb` |
| `test_reviewer` | `c7c71887b22e` |
| `orchestrator` | `4eda79561a8c` |
| `verifier` | `4fb90e13fd02` |
| `judge` (orchestrator + `common/judge.md`) | `ec54742e1b4c` |
| `bug_reviewer` cross-file pass (`+ coordination.md`) | `5271f6cf460a` |
| `security_reviewer` cross-file pass | `d7612cacc2eb` |
| `test_reviewer` cross-file pass | `2c7eaba3d288` |

Sub-agent prompts are the specialist's prompt plus `common/subagent_assignment.md` (with the assigned
files filled in). File hashes: `bug_reviewer.md` `9ab92d337cb3`, `security_reviewer.md`
`21a722c725fe`, `test_reviewer.md` `42107586a765`, `orchestrator.md` `8d4b479a17f0`, `verifier.md`
`29984203474f`, `common/categories.md` `2b7519455983`, `common/coordination.md` `e1f7cb746235`,
`common/judge.md` `d125dbe2bed2`, `common/output_rules.md` `2bd3107cf735`,
`common/severity_confidence.md` `5aa193db92ce`, `common/subagent_assignment.md` `c061433e8e60`,
`common/untrusted_input.md` `21728de451d8`.

## No tuning

Nothing in the frozen list changes because of a result, a finding or a score. A technical failure
may be re-run only under the policy below. If a run shows a **systematic defect** (for example a crash
that hits every item of a configuration, or wrong accounting), work stops and the project owner is
told; a fix then needs its own decision record and a re-run of every affected item in every affected
configuration.

## Scoring and re-run policy (decision 0010)

`decide_scoring` is the same for every configuration: `scored` for `complete`, `limit_reached`, a
non-infrastructure `error`, and a `partial` whose non-complete children all ended that way;
`rerun_required` for `quota_paused`, `RetriesExhausted` or `AllExhausted` anywhere, `RemoteUnavailable`
or `RemoteError`, a sandbox-unavailable `partial`, and a `partial` with no recorded cause. A
`rerun_required` item is not scored and is retried with `eval.run --config <c> --items <id> --force
--reason "<why>"`; every attempt, with its reason and decision, stays in the item's `run_log`.
Re-runs happen **only** for `rerun_required` items. A scored result is never re-run to get a better
one.

### The `single` row mixes two retry regimes

`single` results for item_001 to item_005 come from **stage 1** (Nov 20), recorded under task-level
retries; their run logs show **zero retries** (decision 0010), so the retry regime did not affect
them. The 15 new items (item_006 to item_020) run under the current **per-call retries** with the
same frozen prompt (`aa94f1ae2cf6`) and `limits.single` (25 model / 40 tool calls). Both sets are
reported as one row; the report lists which items came from which stage.

## Waiting (D7)

Every wait is logged as `rate_limiter`, `retry_minute` or `retry_server`. For `single`,
`net_duration_s = duration_s - wait_s`. For orchestrated runs, parallel waits overlap, so the removed
wait is the wall-clock union `wait_s_wall` (decision 0009): `net_duration_s = duration_s -
wait_s_wall`; `wait_s` (the sum) is reported too. Reported per configuration: **median net time**
and **median wait** (`wait_s_wall` for orchestrated rows, `wait_s` for `single`). For `verifier` and
`judge` the time and cost are cumulative with the `subagents` review.

## Blind labelling (D9b, `eval/label.py`)

Adjusted precision needs a person to judge unmatched findings, with the configuration hidden.

1. Collect every **strict-unmatched, in-diff** finding (the scorer's `unmatched_findings` of strict
   mode; findings discarded by D9a are out) from every configuration with complete results.
2. **Deduplicate claims across configurations**: two findings are the same claim when they are in
   the same item, the same file, the same category, and their line ranges overlap
   (`a.start <= b.end and a.end >= b.start`, no tolerance); claims are grouped transitively. Each
   claim is labelled once and the label is applied to every finding in it, in every configuration.
   (Reused `subagents` items carry the same findings as `orchestrator`, so they collapse into one
   claim.)
3. Claims are shuffled with a fixed seed. The labeller never sees the configuration, the agent, the
   model, the run, the confidence or how many configurations made the claim.
4. For each claim the labeller sees: the item's PR title, file, lines, category, title, explanation,
   evidence and the code around it (±15 lines) from the materialized workspace at the PR's target
   state (`eval.repos.materialize`).
5. Answers: `r` (real bug), `f` (false alarm), `s` (skip for now), `q` (save and quit). Progress is
   saved in `eval/labels/claims.json`; a restart resumes. Skipped and unanswered claims stay
   unlabelled and are reported as `unlabelled_unmatched` by the scorer.
6. When labelling is done, one `manual_labels.json` per configuration is written next to its
   findings (`eval/results/<config>/manual_labels.json`, the scorer's format: finding id →
   `real_bug` | `false_alarm`). Adjusted precision = (tp + labelled real bugs) / (tp + fp), as
   `eval/score.py` defines it. Labelling happens in 5Z after every run is finished; labels never
   feed back into any prompt, limit or rule.

## Metrics

Per configuration (all from files in `eval/results/<config>/`; each number in the report traces to
`score.json`, `runs/*.json` or `manual_labels.json`):

- strict **recall**; strict and **adjusted precision**; **false alarms per clean PR** (6 clean items);
  location-only recall and precision beside the strict ones;
- **mean tokens** (input / output) and **estimated cost per review** at the D5 prices in
  `config/models.yaml` (cumulative for `verifier` and `judge`);
- **median net time** and **median wait**;
- the same table for the small, medium and large buckets;
- **per-kind recall** (strict, by the planted kind of each bug);
- **recall on the 3 cross-file bugs** (item_007/b2, item_008/b2, item_015/b1,
  `eval/dataset_notes.yaml`);
- **hits found only by sub-agents**: strict hits whose matching finding's `agent` ends in `/sub<k>`
  (a sub-agent of a specialist), and among them the bugs that the `orchestrator` row missed;
- **calls and tokens by agent** for each configuration (orchestrator, each specialist, each
  sub-agent, verifier or judge), from the runs files;
- for `verifier` and `judge`: findings in, kept, dropped; how many dropped findings were **hits**
  (recall lost) and how many were **false alarms** (precision gained);
- the lists of items reused (`subagents`), re-run (any configuration, with cause), and taken from
  stage 1 (`single`).

## Call budget

Free tier, 500 requests a day for the evaluation model. `eval.run --max-calls N` (tooling added
under this record) stops cleanly before an item whose estimate would take the invocation past `N`
(exit 3, "budget reached; resume tomorrow"), never interrupts a running item, and also exits 3 on
`QuotaPaused`. Day 1 uses 430 as the total for the day, leaving headroom for retries and probes.
`eval/plan_runs.py` shows the status matrix and the estimates; its estimates use the observed mean
model calls per item for a configuration and bucket once 3 or more items exist, otherwise the
defaults (small 10, medium 30, large 70 for `orchestrator` and `subagents`; 10 for `single`; 3 per
finding for `verifier`). Model calls counted are the `model_calls` of each result (retries are not
counted there, so the real daily count can be slightly higher).
