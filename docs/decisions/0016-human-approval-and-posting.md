# 0016: Human approval, checkpointed resume and GitHub posting (stage 6)

Date: 2026-11-24 · Status: accepted (code and unit tests done; live acceptance run **pending**,
see "Task 6 results") · Implements: plan § stage 6. Replaces no locked decision.

## Where the code lives

`approval/` at the repository root (not under `src/`, which decision 0015 freezes during stage 5):
`github.py` (client), `decisions.py`, `posting.py`, `graph.py`, `store.py`, `workspace.py`,
`cli.py`, `__main__.py`. It imports the node functions of `reviewer.graph` and builds its own
`StateGraph`; `reviewer.graph.build_graph` is untouched. Run it with
`uv run python -m approval review-pr|approve|resume|status`.

**Follow-up after stage 5.** Once the freeze lifts, move `approval/` into `src/reviewer/` as a
behaviour-identical refactor: `github.py`, an `approval` flag (default off) on `build_graph`, and
`reviewer` CLI subcommands.

## The graph

`START -> review -> judge -> present -> approve -> post -> END`. State is `ReviewState` plus `repo`,
`pr_number`, `head_sha`, `base_sha`, `pr_url`, `decisions`, `approved_findings`, `posted`; all
JSON-serializable; no token.

- **`approve`** calls `interrupt(approval_payload)` (the list `present` builds: `id`, `file`,
  `line_start`, `line_end`, `category`, `severity`, `confidence`, `title`, `explanation`, `evidence`,
  `agent`, `verdict`, `verdict_reason`). Resume value: a list of
  `{finding_id, action: "approve"|"edit"|"reject", edited: {title?, explanation?, severity?}}`.
  With no findings, or a review that ended `error` or `quota_paused`, there is no interrupt.
- **Decision rules** (`decisions.py`): every finding id exactly once, no unknown id, `edited` only
  for `edit` and only `title`, `explanation`, `severity` (never file, lines or category), titles
  and explanations not blank, severity high/medium/low. Invalid decisions raise `DecisionError`
  before anything is posted.
- **`post`**: marker `<!-- mapr-review:{review_key} -->`. It lists the PR's reviews; if one body
  contains the marker it records `{review_id, url, existed: true}` and posts nothing. Otherwise it
  creates one `COMMENT` review (inline comments for the approved findings, summary body with the
  marker) and records `{review_id, url, existed: false, inline, in_summary}`. No approved finding:
  `{skipped: "no approved findings"}` and no GitHub call.

## Checkpoints and resume

`SqliteSaver` at `.reviewer/checkpoints.sqlite` (gitignored), `thread_id` = `review_key`. A checkpoint
is written after each node, so a killed process resumes (`resume`, `graph.invoke(None, config)`) from
the last completed node. A node that was running when the process died re-runs from its start; that
is safe because the nodes are idempotent (decision 0013) and `post` checks its marker first.
`approve` on a finished review re-runs `post` from its checkpoint (it finds the marker).
Decision progress is saved in `.reviewer/decisions/<key>.json`.

## Mapping findings onto diff lines

`eval.score.changed_lines` of the PR diff. A finding becomes an inline comment (`side: RIGHT`) only
if `line_end` (and `line_start` for a range) are changed lines of its file. A range uses
`start_line`/`start_side` only if both ends are in one hunk; otherwise it is a single-line comment on
`line_end`. Other findings are listed in the review's summary body. Comment text: title and severity,
explanation, evidence in a code fence, footer `found by <agent> · verified: <verdict>` plus
` · edited by reviewer` when edited.

## Token and secrets

`GITHUB_TOKEN` only from `.env`; fine-grained, one repository, Pull requests read/write, Contents
read-only, Metadata read-only, 30 days. It is sent only in the `Authorization` header; `scrub()`
removes it (and `Bearer ...` / `github_pat_...` shapes) from every error text, and `main` scrubs
anything it prints. The clone URL has no credentials. `create_review` is not retried on 5xx (the POST
may have been processed); 429, secondary rate limits (`retry-after`, `x-ratelimit-reset`) and, for
reads, 5xx are retried with backoff.

## Cut

The webhook (first on the plan's cut list) and Postgres are cut; the CLI posts and SQLite checkpoints.

## Isolation from the evaluation

No file under `src/`, `config/`, `eval/score.py`, `eval/ground_truth.py`, `eval/equivalences.yaml`,
`eval/dataset/` or `eval/plants/` changed (`git diff cb12ec8` on them prints nothing); `eval.run` does
not import `approval`. Test: `build_graph()` has the same nodes, edges and final state keys.

## Interpretations

The workspace is `<folder>/repo` with `pr.diff` beside it; `--key-suffix S` appends `+S` to the key;
the same key is never reviewed twice (`review-pr` says it exists); `skip` leaves a finding undecided
and `approve` stops until all are decided.

## Task 6 results

**Pending (model 503).** The probe (`scripts/check_finding.py`, 1 model call) returned
`503 UNAVAILABLE` on 2026-11-24. Done without model calls: test repository
`BilalAamir4/mapr-playground`, PR https://github.com/BilalAamir4/mapr-playground/pull/1 (head
`9623bcd`), and a read-only check of the real client (`get_pr`; `get_pr_diff` identical to the
fixture's `pr.diff`; `list_reviews` empty). Not done: the live review, the crash/resume demo, the
approve/reject run and the double-post check.
