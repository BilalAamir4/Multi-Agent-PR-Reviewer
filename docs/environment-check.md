# Environment check (stage 0B)

Date: Nov 19, 2026 · Windows 11, PowerShell, Python 3.12.9, uv 0.11.16

Session 0B was interrupted by a power cut after the first `check_finding.py` call; it was resumed
the same day. The interrupted run left a LangSmith root run stuck in `pending` with no output
(`01a1f451-c12a-735a-b004-3b9aa606ea68`); it was counted as a used call and the check was re-run.

## Versions

| Component | Version |
| --- | --- |
| Docker Desktop engine | client 29.8.0 / server 29.8.0 (linux/amd64, WSL 2) |
| `langgraph-api` | 0.18.3 (with `langgraph-cli` 0.4.43, `langgraph-runtime-inmem` 0.38.3) |
| `langchain` / `langchain-core` | 1.5.1 / 1.8.3 |
| `langchain-google-genai` / `google-genai` | 4.5.0 / 3.5.0 |
| `langchain-groq` / `groq` | 1.2.1 / 1.7.0 |
| `langgraph` | 1.2.22 |
| `langsmith` | 0.15.0 |
| `docker` (Python SDK) | 7.2.0 |
| `httpx`, `pydantic`, `python-dotenv` | 0.28.1, 2.14.3, 1.2.4 |
| `pytest`, `ruff` | 9.2.0, 0.17.5 |

ripgrep: 14.1.1 (installed after 0B; confirmed Nov 20, 2026).

## Model IDs

Full listing: `docs/model-inventory.md`. Chains and limits: `config/models.yaml`.

| Routing doc name | Exact ID | Provider |
| --- | --- | --- |
| Gemini 3.8 Flash | `gemini-3.8-flash` | `google_genai` |
| Gemini 3.6 Flash | `gemini-3.6-flash` | `google_genai` |
| Gemini 3 Flash | `gemini-3-flash-preview` (owner's choice; only gemini-3-flash ID listed) | `google_genai` |
| Gemini 2.5 Flash | `gemini-2.5-flash` | `google_genai` |
| Gemini 3.5 Flash Lite | `gemini-3.5-flash-lite` | `google_genai` |
| Gemini 3.1 Flash Lite (evaluation, D4) | `gemini-3.1-flash-lite` (owner's choice over `-preview`) | `google_genai` |
| Gemma 4 31B | `gemma-4-31b-it` | `google_genai` |
| Gemma 4 26B | `gemma-4-26b-a4b-it` | `google_genai` |
| Gemini 4 Hydrogen | `gemini-4-hydrogen` | `google_genai` |
| gpt-oss-120b | `openai/gpt-oss-120b` | `groq` |
| qwen3.8-27b | `qwen/qwen3.8-27b` | `groq` |
| gpt-oss-20b | `openai/gpt-oss-20b` | `groq` |
| llama-prompt-guard-2-86m | `meta-llama/llama-prompt-guard-2-86m` | `groq` |

## Results

| Task | Check | Result | Note |
| --- | --- | --- | --- |
| 4 | Traced call returns a valid `Finding` (`scripts/check_finding.py`) | **pass** | `gemini-3.1-flash-lite`: `off_by_one`, lines 12–13 (planted 12–13), evidence verbatim, no key material in the trace |
| 4 | Same check as an integration test | fail (503) | `503 UNAVAILABLE: model is currently experiencing high demand`; the call budget (5) was then used up, so not retried |
| 5 | Tool calling, `gemma-4-31b-it` | **pass** | called `get_line(n=7)`, answer contains the line |
| 5 | Tool calling, `openai/gpt-oss-120b` | **pass** | called `get_line(n=7)`, answer contains the line |
| 5 | Tool calling, `gemma-4-26b-a4b-it` | not tested | 503 UNAVAILABLE on the run and on the one retry |
| 5 | Tool calling, `gemini-4-hydrogen` | not tested | 503 UNAVAILABLE on the run and on the one retry (capped at 1 request); 2/2 daily calls used |
| 5 | Tool calling, `gemini-3.1-flash-lite` | not tested | 503 UNAVAILABLE on the run and on the one retry |
| 6 | Probe prints `hello from the sandbox` | **pass** | exit code 0 |
| 6 | Network blocked | **pass** | `OSError: [Errno 101] Network is unreachable` connecting to 1.1.1.1:53 |
| 6 | Root filesystem read-only | **pass** | `errno 30 Read-only file system` writing `/probe_test` |
| 6 | `/tmp` writable | **pass** | tmpfs |
| 6 | `/work` writable | **pass** | after the image contract in decision 0003 (`/work` owned by 1000:1000) |
| 6 | Not root | **pass** | uid=1000 gid=1000 |
| 6 | Timeout | **pass** | `timed_out=True` after 3 s, no sandbox container or new dangling volume left |
| 7 | `langgraph dev`: `GET /ok` | **pass** | 200 |
| 7 | `POST /assistants/search` lists `echo` | **pass** | `assistant_id=046b5dc7-2108-582c-9c6c-31d556d138f0` |
| 7 | A2A `message/send` to `/a2a/{assistant_id}` | **pass** | result contains `echo: ping from check_langgraph` |
| 7 | Server stopped, port 2024 free | **pass** | process tree killed with `taskkill /T /F` |

`tool_calling` in `config/models.yaml` is `true` for the two models that passed; the three that
got 503s stay `unknown`, because overload says nothing about tool calling.

### Follow-up re-run (Nov 20, 2026, after the 1:00 PM PKT quota reset)

`uv run python scripts/check_tool_calls.py gemma-4-26b-a4b-it gemini-4-hydrogen
gemini-3.1-flash-lite` (at most 2 model calls per run, no extra retries), then the `Finding`
integration test once.

| Check | Result | Note |
| --- | --- | --- |
| Tool calling, `gemma-4-26b-a4b-it` | not tested | 503 UNAVAILABLE again; `tool_calling` stays `unknown` |
| Tool calling, `gemini-4-hydrogen` | **pass** | called `get_line(n=7)`, answer is the exact line; `tool_calling: true` |
| Tool calling, `gemini-3.1-flash-lite` | **pass** | called `get_line(n=7)`, answer is the exact line; `tool_calling: true` |
| `Finding` integration test (`tests/test_integration_finding.py`) | **pass** | 1 call to `gemini-3.1-flash-lite`, traced, no key in the trace |

LangSmith trace of the `Finding` call:
https://smith.langchain.com/o/dc790a75-743f-467d-a940-18bb085de125/projects/p/494eeee1-bc47-4880-971b-55ac8176fdee/r/01a1f495-c4c3-71ad-a9dc-64703bceb6aa?poll=true

Sandbox images:

- base `python:3.12-slim` digest
  `python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b`
- check image `mapr-sandbox-base:check`
  (`sha256:59d67e7d748648b31d73d5548992efa86644be7d0d70648644061be571e3ef39`), built from the base
  pinned by digest plus `RUN mkdir /work && chown 1000:1000 /work`.

## Windows notes

- `langgraph dev` resolves graph paths in `langgraph.json` against the working directory, not
  the config file's folder. `scripts/check_langgraph.py` therefore runs it from
  `scripts/langgraph_check/` with `--config langgraph.json`. When a graph fails to load, the
  reloader keeps the process alive, so the script also stops waiting on "Application startup
  failed".
- Stopping `langgraph dev` needs the whole process tree killed (`uv` → `langgraph` → `uvicorn`).
- The Gemini client retries up to 6 times by default; the check scripts pass `max_retries=0` so
  the free-tier daily request quotas are not spent silently.
- `langsmith` 0.15.0 warns that `list_runs()` and `get_run_url()` are deprecated (removal after
  Jan 31, 2027) in favour of `client.runs.query()` / `client.runs.get_url()`.
