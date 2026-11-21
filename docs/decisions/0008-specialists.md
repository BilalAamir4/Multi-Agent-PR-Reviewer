# 0008: Specialists, shared agent machinery and the `run_tests` sandbox tool

Date: Nov 21, 2026 · Status: accepted · Implements: stage 2A (plan § Build stages, stage 2),
D2, D2a, D2b, D11, plan § Production concerns (limits). Refines D2b (does not replace it).
Replaces no locked decision.

Stage 2 is split in two. **2A** (this record): the shared agent machinery, the `run_tests` tool
and three specialists, each runnable on its own. **2B** (later): the orchestrator that calls
them as tools, merges their findings and passes the routing test.

## Shared machinery (`src/reviewer/agents/base.py`)

- `AgentSpec(name, role, prompt_path, tool_names, limits, extra_tools, focus_categories)`.
  `name` is also `Finding.agent`; `role` is the router chain; `tool_names` is a subset of the six
  stage 1A tools and `extra_tools` adds `run_tests`; `focus_categories` is used only in the
  prompt. The run name is `<name>-review`.
- `run_agent(spec, ctx, user_message, router, *, item_id=None, evaluation=False)` does what the
  stage 1 `review()` did: `create_agent` with `ToolStrategy(ReviewOutput)`, the call-limit
  middleware (counting only review tools), the per-call rate-limit middleware, `router.run`
  inside `review_waits()`, `to_findings` (decision 0007) and token, call and time accounting
  with LangSmith tags `[name, item_id or "adhoc"]` and metadata (`prompt_version`, `model_id`,
  `evaluation`). `evaluation` must equal `router.evaluation`, or it raises `ValueError`.
- `ReviewResult` gains `agent` (default `"single"`, so stage 1 JSON still loads) and
  `notes: list[str]`, and the status `partial` (below). Every other field is unchanged.

`single.py` now builds its `AgentSpec` and calls `run_agent`. Its prompt (`aa94f1ae2cf6`),
user message, `limits.single`, tool set and middleware are unchanged:
`tests/test_single_identical.py` captures everything `review()` passes to `create_agent` and to
`invoke` (system prompt hash, tools with their descriptions and argument schemas, response
format and schema, middleware and limits, agent name, first message, run name, tags, metadata
keys, recursion limit) and compares its SHA-256 with the digest recorded from commit 1a869be
before the refactor. The stage 1 unit tests pass unchanged.

## Composed prompts; `single.md` stays separate

`src/reviewer/prompts/common/` holds four fragments copied **verbatim** from `single.md` as it was
at 1a869be: `categories.md`, `severity_confidence.md`, `output_rules.md`, `untrusted_input.md`
(a unit test checks each is a substring of `single.md`). A specialist prompt is its own intro,
focus and "How to work" sections plus the lines `<!-- include: NAME -->`, which are replaced by
the fragments; `<!-- focus_categories -->` becomes "Your focus categories: ...". A leftover
include marker is an error. `single.md` is not changed and does not use the fragments; it has no
markers, so it composes to itself byte for byte. A specialist's `prompt_version` is the first 12
hex characters of the SHA-256 of its composed prompt, so a fragment change changes it.

## The specialists (`src/reviewer/agents/specialists.py`)

| name | Focus | Tools | Limits (model / tool calls) |
| --- | --- | --- | --- |
| `bug_reviewer` | `off_by_one`, `missing_none_check`, `wrong_comparison`, `unhandled_exception`, `other` for logic defects | `list_changed_files`, `read_diff`, `read_file`, `search_code`, `run_ruff` | 20 / 30 |
| `security_reviewer` | `sql_injection`, `hardcoded_secret`, `other` for security defects (unsafe deserialization, shell injection, path traversal, ...) | `list_changed_files`, `read_diff`, `read_file`, `search_code`, `run_bandit` | 15 / 25 |
| `test_reviewer` | `ineffective_test`, and defects in the code under test that the tests reveal | `list_changed_files`, `read_diff`, `read_file`, `search_code`, `run_tests` | 20 / 30, `run_tests` at most 3 |

- All use the `specialist` router chain. Limits are in `config/review.yaml` under
  `limits.<name>`; `AgentLimits` gains `max_run_tests_calls` (null for agents without
  `run_tests`).
- Each prompt says a defect outside the focus may still be reported with its correct category
  if it is clearly real.
- The user message is the PR title and description marked as untrusted (as in stage 1), plus
  either `<focus>...</focus>` with the text the orchestrator supplies in 2B, or "No particular
  focus was requested: review the whole change in your area."
- `run_specialist(name, ctx, *, pr_title, pr_body, focus=None, router, item_id=None,
  evaluation=False)`.

### "Read every changed file in your area"

Each specialist prompt tells the agent to call `read_diff` on every changed file in its area
before it answers: every changed `.py` file for the bug and security reviewers; every changed
test file, plus the changed code those tests exercise, for the test reviewer. **This is a
deliberate design difference from the baseline.** The single agent's prompt (frozen since the
stage 1B run) only says to read the diff "one file at a time if the diff is long", and in the
baseline it sometimes skipped changed files. The rule belongs to the multi-agent design being
evaluated, so it is not back-ported to `single.md`; the "One agent" row stays as measured.

### Stage 3 entry point

`run_security_review(request: dict) -> dict` is the security specialist's whole job as one
call with JSON-serializable input and output, for the A2A service. Request keys (all required,
no others): `workspace` (path), `diff_text`, `pr_title`, `pr_body`, `focus` (string or null),
`evaluation` (bool). Response: `agent`, `status`, `findings` (`Finding` dicts), `dropped`,
`notes`, `summary`, `error_class`, `error`, `limit`, `reset_at`, `model_id`, `prompt_version`,
`evaluation` and `counts` (model and tool calls, calls by tool, tokens, duration, wait, net
duration). A unit test checks that the response equals `result_to_response` of
`run_specialist`'s result for the same scripted run, and survives `json.dumps`/`json.loads`.

## `run_tests` (`src/reviewer/tools/testing.py`; D2, D2a, D2b)

- `SandboxSpec(image, test_command, pythonpath, timeout_s)` is a frozen dataclass;
  `ReviewContext.sandbox` is optional. `eval.repos.sandbox_spec(name)` builds it from
  `eval/repos.yaml`; `eval.repos.item_sandbox(item_id)` from an item's source repository (the
  CLI's `--item` uses it); `--sandbox-repo NAME` picks an entry for an ad-hoc review. With no
  spec the tool returns `ERROR: no test sandbox is configured for this review` and the result
  gets the note `run_tests unavailable: ...` (status unchanged: no spec is a configuration
  choice, not a failure). `pythonpath` is recorded only; the image already sets `PYTHONPATH`,
  and `run_in_sandbox` is unchanged.
- **Command:** the spec's `test_command` with every `-x` and `--exitfirst` removed, then
  `--tb=short`, `-rfE` and `-q`, each only if not already present (the dataset commands already
  have `-q`, and a second one would make pytest quieter), then the test IDs.
- **Test IDs:** at most 50 per call. Each is refused if it is empty, starts with `-`, contains
  whitespace, `..` or a backslash, or if its path part (before the first `::`) fails the
  `read_file` path rules (absolute, drive, `:`, `.git`) or is not an existing file or directory
  in the workspace; `tests/x.py::` (nothing after `::`) is refused too. A leading `./` is
  dropped. No IDs (or an empty list) runs the whole suite.
- **Files:** never mounted (D2). When the workspace is the root of a git work tree (an
  `eval/workspaces/<item>` worktree with `pr.diff` applied), the files are its tracked files
  (`git ls-files -s`, symlinks and submodules left out) plus the files the diff writes, read
  from disk, which is the PR version; deleted files are gone from disk and left out. Files over
  1 MB are skipped, as `snapshot_files` does. Any other workspace (the synthetic dev fixtures)
  sends every regular file except `.git`, `__pycache__` and tool caches.
- It runs through `run_in_sandbox(spec.image, files, command, spec.timeout_s)` with the D2
  settings unchanged.
- **Output:** `pytest exit code N: <summary line>`, `Failed: <ids>`, `Errors: <ids>` (from the
  `-rfE` short summary), then the FAILURES and ERRORS sections (`--tb=short`). The raw output
  tail (4,000 characters, stdout plus stderr) is added when there is no summary line or the exit
  code is not 0 or 1. Everything is cut to 12,000 characters with the usual
  `[truncated: ...]` note. On a timeout: `ERROR: tests timed out after N s` plus whatever output
  exists.
- **Limit:** at most `limits.test_reviewer.max_run_tests_calls` (3) calls per agent run, counted
  by a second `ToolCallLimitMiddleware` (`tool_name="run_tests"`, `exit_behavior="continue"`).
  A fourth call is refused with an error tool message and the review goes on. `run_tests` calls
  also count towards the agent's overall tool-call limit.

### D2b: `partial`

If Docker cannot be reached, `run_tests` returns `ERROR: Docker Desktop is not running`, the
result gets the note `sandbox unavailable: Docker Desktop is not running`, and a review that
finishes with a structured answer ends with status **`partial`**: it finished, but without a tool
it needed. Its findings are kept. A missing image (`ImageNotFound`) is handled the same way
(`sandbox unavailable: sandbox image <image> is not built`). The tool checks the image with
`client.images.get` before `run_in_sandbox`, because `run_in_sandbox` maps the docker client's
HTTP errors (which are `OSError`s) to `SandboxUnavailable` and would report a missing image as
Docker not running. The CLI exits with code 5 for `partial`; the other exit codes are unchanged.

## Dev fixtures and the `dev_fixture` image

The specialists are developed only on synthetic fixtures in `tests/fixtures/` (`tests/dev_pr.py`).
Each has `workspace/`, `pr.diff` (a real `git diff` of a before and an after tree) and an
informational `expected.json`:

- `dev_pr` (stage 1B): gains `tests/test_catalog.py`, whose `test_page_sizes` fails on the
  PR's `page()` off-by-one. Its `pr.diff` also had a stray `pr.diff.out` entry (a nested copy of
  an older diff, not a file in the workspace); it was removed. The four real file diffs are
  byte-identical.
- `dev_security_pr`: `search_users()` formats `term` and `order_by` into SQL; `admin.py` has a
  password literal.
- `dev_tests_pr`: `test_mean_empty` now swallows the `ValueError`; a new `_limit()` helper
  compares with `low` instead of `high`, which the existing `test_clamp` catches.
- `dev_docs_pr`: `README.md` and `docs/usage.md` only (for 2B's routing test; not run here).
- `dev_risky_pr`: SQL injection, a sync password, an off-by-one and a weakened test in three
  `.py` files (under the D12 spawn limit; for 2B; only built here).

`sandbox_images/dev_fixture/` follows decisions 0003 and 0004 (same base digest, uid/gid 1000,
`RUN mkdir /work && chown 1000:1000 /work`, `PYTHONPATH=/work`) and installs `pytest==9.2.0`
and its dependencies at this project's lock versions. `eval/repos.yaml` lists it as
`dev_fixture` with `url: null`, `license: n/a`, `default_branch: null`, `image_commit: null` and
`image: mapr-sandbox-dev_fixture:local`. The loader allows these values only for the names in
`eval.repos.LOCAL_REPOS` and only all together; `dataset_repos()` (used by `eval.check_repos`)
and `repo_name_for_slug` skip it, and `ensure_clone` refuses it.

## Not run on evaluation items before stage 5

No specialist runs on `eval/dataset/` before the stage 5 evaluation. The CLI refuses
`--item` with any `--agent` other than `single`.

## Dev runs and prompt changes

One run per specialist on its fixture, evaluation mode (`gemini-3.1-flash-lite`), Nov 21, 2026:

| Agent | Fixture | Prompt | Status | Expected defects found | Model / tool calls | Tokens in / out | Net / wait (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bug_reviewer` | `dev_pr` | `0a0cf383d913` | complete | 2 of 2 | 7 / 6 | 24,230 / 488 | 14.5 / 0.0 |
| `security_reviewer` | `dev_security_pr` | `6f396484c1bb` | complete | 2 of 2 | 7 / 6 | 21,390 / 448 | 16.4 / 32.5 |
| `test_reviewer` | `dev_tests_pr` | `c7c71887b22e` | complete | 2 of 2 | 7 / 6 (1 `run_tests`) | 24,894 / 491 | 15.5 / 0.3 |

**Prompt changes made during dev runs: none.** Every specialist found both of its fixture's
defects on the first run, and nothing was dropped by post-processing, so the prompts are as
first written.
