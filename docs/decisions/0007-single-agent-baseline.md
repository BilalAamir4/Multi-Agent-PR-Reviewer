# 0007: Single-agent baseline, shared post-processing and call limits

Date: Nov 21, 2026 · Status: accepted · Implements: D11, plan § Production concerns (limits),
stage 1B. Refines D11 (does not replace it).

## Post-processing applies to every configuration

D11 places the evidence check "before the verifier runs". It is applied to **every**
configuration, not only "plus verifier": the one agent, the orchestrator with specialists, the
spawned workers and the verifier all pass their findings through the same deterministic code,
`to_findings` in `src/reviewer/agents/output.py`. So every architecture gets identical
post-processing, and "plus verifier" is credited only for what the verifier agent adds.

The agent returns `ReviewOutput` (`summary`, `findings: list[FindingDraft]`). `FindingDraft` is
`Finding` without `agent` and `model`, with only simple constraints (`line_start`/`line_end` at
least 1, `confidence` from 0 to 1), so the structured-output schema stays plain for Gemini.
`to_findings(drafts, agent, model, ctx)` runs four steps, in this order, and records each removed
draft as `Dropped(draft, reason, detail)`:

1. **`schema`**: build the `Finding`, with `agent` and `model` filled in by code. A
   `ValidationError` (for example `line_end < line_start`, a blank title, a backslash path)
   drops the draft with the error text.
2. **`not_in_diff`**: the file must be a changed file of the diff that is not deleted (renamed
   files count under their new path). The D9a ±3 line filter is **not** applied here; the scorer
   applies it, so discarded findings stay visible in the scores.
3. **`evidence`** (D11): see the rule below.
4. **`duplicate`**: see the rule below.

### Evidence-matching rule

Take `evidence`, split it on newlines, drop leading and trailing blank lines, and strip each
line's leading and trailing whitespace. Strip each line of the file after the PR the same way
(lines split on `\n`, as git numbers them). The evidence lines must equal a **contiguous run**
of file lines `s..e` (1-based) such that:

- at least one matched line is inside `[line_start, line_end]` (`s <= line_end` and
  `e >= line_start`), and
- every matched line is inside `[line_start - 3, line_end + 3]`.

If the snippet occurs several times, any one qualifying run is enough. Otherwise, or if the file
cannot be read, the draft is dropped. Indentation differences are accepted; skipped lines,
line-number prefixes and wrong lines are not. `Finding.evidence` keeps the text unstripped.

### Duplicate rule

Two findings are duplicates when they have the same file, the same category and overlapping line
ranges (`a.line_start <= b.line_end` and `a.line_end >= b.line_start`). Findings that survived
steps 1 to 3 are visited from the highest confidence down (the earlier finding first on a tie);
each is kept unless it duplicates one already kept, in which case it is dropped with the reason
`duplicate`. Kept findings stay in their original order. A draft dropped at an earlier step can
never win this step.

## Call limits come from config

`config/review.yaml` has `limits.single: {max_model_calls: 25, max_tool_calls: 40}`
(`ReviewConfig.limits`). The agent uses LangChain's call-limit middleware with
`exit_behavior="end"`, so reaching a limit ends the run cleanly (no exception inside the graph).
A run that ends that way before a structured response exists is `limit_reached` with no findings;
nothing is salvaged from free text. Later agents get their own `limits.<role>` block.

Implementation notes (installed `langchain` 1.5.1):

- `ToolStrategy` implements the structured answer as a tool call named `ReviewOutput`, and
  `ToolCallLimitMiddleware(tool_name=None)` counts every tool call on an AI message, including
  that one. The agent subclasses the middleware so only the six review tools are counted and the
  final answer is never blocked.
- The model call limit ends the run before call 26; the tool call limit ends it when a batch of
  calls would take the count past 40 (that batch does not run).
- The router waits on the model's rate limiter once per task. The agent adds a middleware that
  waits on the same limiter before every later model call in the loop, recorded in the review's
  `WaitLog` as `rate_limiter` (D7), so a long tool loop stays under the RPM instead of hitting
  429s that would re-run the whole task (S4). The router's behaviour is unchanged.
- Token counts are summed from `usage_metadata` of every completed model call in every attempt;
  model and tool call counts likewise.

## The single-agent prompt

`src/reviewer/prompts/single.md`. Its `prompt_version` is the first 12 hex characters of the
SHA-256 of the file's bytes, recorded in every result. Category definitions come from
`docs/categories.md`, phrased as what to look for in new or changed code. The prompt mentions no
dataset, planted bugs, evaluation or scoring, and gives no hint about how many defects to expect.
The user message gives only the PR title and description (from `meta.json` `pr_title` and
`pr_body`), marked as untrusted data.

It was developed only against the synthetic dev fixture `tests/fixtures/dev_pr/` (19 model calls
in three runs). Two changes came from that fixture: the evidence rule now says to leave out the
`NN | ` prefix `read_file` adds (with an example), and the `evidence` field's schema description
says the same. The model had copied the prefixes, so every evidence check failed.

## Results: stage 1 baseline

Run once on Nov 20, 2026 at 20:29 UTC (Nov 21, 01:29 Pakistan time) with
`uv run python -m eval.run --config single`. Model `gemini-3.1-flash-lite` (evaluation mode, D4),
prompt version `aa94f1ae2cf6`, limits 25 model calls and 40 tool calls. All 5 items completed on
the first attempt. **No item was re-run.**

| Mode | Bucket | Items | Recall | Precision | tp | fp | fn | Discarded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict | all | 5 | 0.17 | 0.50 | 1 | 1 | 5 | 0 |
| strict | small | 2 | 0.00 | - | 0 | 0 | 1 | 0 |
| strict | medium | 1 | 0.00 | 0.00 | 0 | 1 | 2 | 0 |
| strict | large | 2 | 0.33 | 1.00 | 1 | 0 | 2 | 0 |
| location_only | all | 5 | 0.33 | 1.00 | 2 | 0 | 4 | 0 |
| location_only | small | 2 | 0.00 | - | 0 | 0 | 1 | 0 |
| location_only | medium | 1 | 0.50 | 1.00 | 1 | 0 | 1 | 0 |
| location_only | large | 2 | 0.33 | 1.00 | 1 | 0 | 2 | 0 |

- False alarms per clean PR: 0.00 (strict and location-only). Adjusted precision (D9b): not
  computed; no manual labels.
- Median review time with rate-limit waiting removed (D7): 15.1 s, median wait 37.5 s.
- Mean tokens per review: 107,027 input, 412 output; estimated $0.02738 per review at the D5
  price estimate ($0.25 / $1.50 per million tokens; run on the free tier).
- Dropped by post-processing: none.
- Disclosure (added in stage 2A): before the baseline run, `eval/dataset/item_001/meta.json`
  (which includes the plant patch) was opened while the prompt was being written; nothing from
  it entered the prompt, and the prompt's category definitions are the pre-written ones from
  `docs/categories.md`.

Full tables: `eval/results/single/summary.md`; per-item results and run logs:
`eval/results/single/runs/`.

### Frozen from this commit on

- The dataset recipes in `eval/plants/` and everything in `eval/dataset/` (above all
  `ground_truth.json` and `pr.diff`) are frozen, as `eval/dataset/README.md` says; so is
  `eval/score.py` (decision 0002).
- For the "One agent" row, `src/reviewer/prompts/single.md` (version `aa94f1ae2cf6`) and
  `limits.single` in `config/review.yaml` are frozen. A change to either needs a new decision
  record and a new run of every item, reported as a new row.
