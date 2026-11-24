# 0016: Human approval, checkpointed resume and GitHub posting (stage 6)

Date: 2026-11-24 (live acceptance 2026-11-25) · Status: accepted (code, unit tests and live
acceptance run done, see "Task 6 results") · Implements: plan § stage 6. Replaces no locked decision.

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

The 2026-11-24 attempt hit `503 UNAVAILABLE`; the acceptance run was done on 2026-11-25 (probe
`scripts/check_finding.py` passed first). Test repository `BilalAamir4/mapr-playground`, PR
https://github.com/BilalAamir4/mapr-playground/pull/1 (head `9623bcd`).
Command: `review-pr --judge verifier --security-mode local --sandbox-repo dev_fixture --eval-mode`.

- **Review to the interrupt.** Key `BilalAamir4/mapr-playground#1@9623bcde6d1fa6d5618bc1e09e617081adfb47dd`.
  Review status `partial` (two verifier checks ended `limit_reached`; those findings were kept as
  uncertain, per the 0013 policy). 5 findings waited for approval.
- **Crash and resume.** Same command with `--key-suffix demo`. `status` showed `last completed: review,
  next node: judge`; the process was killed while the judge ran; `resume` printed
  `resuming after review; running judge` and stopped at the same interrupt with 5 findings. The review
  node was not re-run.
- **Human decisions (the project owner, live).** All 5 findings approved, no edits, no rejects. The
  owner checked each against the diff: `ledger/report.py:11-12` off_by_one (verifier: confirmed),
  `ledger/store.py:5-6` hardcoded_secret (confirmed), `ledger/store.py:24` sql_injection (confirmed;
  found by bug_reviewer), `tests/test_report.py:5` and `:9` ineffective_test (verifier: uncertain,
  `limit_reached`; both real).
- **Posting.** One `COMMENTED` review,
  https://github.com/BilalAamir4/mapr-playground/pull/1#pullrequestreview-6047172731, carrying the
  marker. Checked afterwards through the public GitHub API: exactly one review on the PR and 5 inline
  comments, on the files and lines above (ranges 11-12 and 5-6 as `start_line`/`line`).
- **Double-post check.** `approve` run again on the same key printed `already posted (marker found),
  not posted again` with the same URL and asked nothing. Screenshot: `docs/media/double-post-check.png`.
- **Not exercised live.** The `edit` and `reject` paths and `--dry-run` were covered only by unit tests,
  not by the live run. The `+demo` review is still waiting at its interrupt, untouched; nothing was
  posted for it. The screen recording of the approval step (`docs/media/approval-demo.*`) is not in the
  repository.
- **Limitation.** `plan_runs` counts only evaluation runs, so the stage 6 model calls (two reviews) are
  not in its daily total.
