# PR review agent: model routing and quotas

Nov 17, 2026 · @Bilal · Companion to `docs/plan.md` · Updated Nov 19, 2026 to match the plan's locked decisions (D4 to D7)

The project runs on free tiers only: the Gemini API (Google AI Studio) and Groq. Each agent role has its own chain of models, and a small router moves to the next model when one runs out of quota. Quotas below were read from both consoles on Nov 17, 2026; check them again before a long run.

## What the free quotas allow

| Group | Models | Limits per model | What runs out first |
| --- | --- | --- | --- |
| A. Gemini Flash | 3.8 Flash, 3.6 Flash, 3 Flash, 2.5 Flash | 5 RPM, 250K TPM, 20 RPD | Requests per day: about 80 calls a day across all four |
| B. Gemini Flash Lite | 3.5 Flash Lite, 3.1 Flash Lite | 15 RPM, 250K TPM, 500 RPD | Requests per minute when many agents run at once |
| C. Groq | gpt-oss-120b, qwen3.8-27b, gpt-oss-20b | 30 RPM, 8K TPM, 1K RPD, 200K TPD | Tokens: one call over 8K is rejected, and 200K a day is about 40 to 50 agent calls |
| D. Gemma | Gemma 4 31B, Gemma 4 26B | 30 RPM, 16K TPM, 14.4K RPD | Untested: check that tool calling works through the Gemini API before relying on it (stage 0) |

Each Gemini model has its own quota bucket, so a chain of Flash models adds up rather than sharing one limit. Pro models have no free quota. Gemini 4 Hydrogen has 20 requests a day but is an unknown model: try it once before trusting it (stage 0). Groq's free `llama-prompt-guard-2-86m` (14.4K requests a day) is a ready injection classifier for the safety tests in stage 7.

## Chains by role

These chains are for normal runs. During evaluation every role uses one pinned model instead (see S9).

| Role | Calls per review | Chain, left to right | Why |
| --- | --- | --- | --- |
| Orchestrator | 4 to 6, in sequence | 3.8 Flash, 3.6 Flash, 3 Flash, 2.5 Flash, 3.5 Flash Lite | Planning and routing need the best reasoning; few calls, so 5 RPM is fine |
| Specialists (bugs, security, tests) | 4 to 6 each, whole files | 3.5 Flash Lite, 3.1 Flash Lite, Gemma 4 31B, gpt-oss-120b (only if the prompt is under about 7K tokens) | Need large context, which Groq's 8K TPM usually cannot hold |
| Per-file workers | 3 to 4 each, in parallel | 3.1 Flash Lite, gpt-oss-120b, qwen3.8-27b, Gemma 4 26B | One file per prompt, so Groq can take overflow |
| Verifier | 1 to 3, small prompts | gpt-oss-120b, qwen3.8-27b, 3.6 Flash | A different model family checks Gemini's findings; prompts fit in 8K |

Rough capacity: a small PR uses about 25 calls, so about 15 reviews a day before the orchestrator chain runs out. A 12-file PR adds about 45 worker calls, so about 10 large reviews a day.

## When quota runs out

| # | Scenario | How to detect it | What the router does |
| --- | --- | --- | --- |
| S1 | Per-minute limit during a burst (most common) | 429; Gemini quota ID contains `PerMinute`; Groq sends `retry-after` | Wait the retry delay (cap about 30 s), retry the same model up to twice, then fall over. Prevent with a rate limiter per model set just under its RPM, and a concurrency cap of about 4 on the `Send` fan-out. Log every wait so review time can be reported with waiting removed (D7) |
| S2 | Daily quota gone mid-run | Gemini quota ID contains `PerDay`; Groq message names requests or tokens per day | Mark the model exhausted until reset and go to the next one at once. Gemini resets at midnight Pacific, 1:00 PM in Pakistan. Save the exhausted list to a JSON file so a restarted run skips dead models |
| S3 | One request is larger than the model's TPM (over 8K on Groq, 16K on Gemma) | 413 or "request too large" | Not a quota problem; retrying never works. Count tokens first and skip any model whose TPM is below the prompt size; if none fit, trim the context |
| S4 | Quota runs out inside an agent's tool loop | Any S2 or S3 error after the agent has made tool calls | Re-run that agent's whole task on the next model. Do not pass half a message history to another provider: tool-call ID formats differ, and Gemini 3-series models can reject function-call history they did not produce. The tools only read, so a re-run is safe |
| S5 | Parallel workers all fall over together and drain the next model at once | Many 429s in the same second | One shared quota tracker for the whole process chooses the model for each call; workers ask it instead of deciding alone |
| S6 | Orchestrator runs out at the final merge | S2 error in the merge node | The checkpoint keeps the specialist results; the merge node retries on the next model. Merging is easy enough for Flash Lite |
| S7 | A fallback model returns broken tool calls or an invalid `Finding` | Schema validation fails | Use the one repair retry. If the same model fails twice in a run, skip it for that run. Tag every finding with the model that produced it |
| S8 | Every model in a chain is exhausted | Router finds no usable model | Stop cleanly: save the checkpoint, return a partial review naming the files not reviewed, resume the same `thread_id` after the reset |
| S9 | Quota runs out during evaluation | Any of the above in an evaluation run | Turn cross-model fallback off. A fallback would compare models, not architectures. **Every role uses one pinned model, Gemini 3.1 Flash Lite (plan D4).** Keep only the S1 wait-and-retry, and on S2 pause until the next day |
| S10 | Another project uses the same Gemini quota | Gemini Embedding 1 was already at 146/100 RPM and 977/1K RPD on Nov 17 | Create a separate AI Studio project for this agent (stage 0). If it needs embeddings, use Gemini Embedding 2, which has its own 1K a day |

The plan's first round runs 20 PRs once per configuration on the free tier (D5) and reuses results between configurations (D6). That is roughly 1,200 calls, about 3 days of 3.1 Flash Lite's 500 requests a day (a rough estimate).

The stage 4 timing test can be hidden by rate limits: 15 RPM on Flash Lite slows a parallel fan-out. Run it on a 4 to 5 file diff, and report how much of the time was waiting on limits.

## Router

LangChain's `.with_fallbacks()` is not enough on its own: it switches on any error at once and does not remember which models are exhausted for the day. Write a small router in `src/reviewer/models.py` with the chains in `config/models.yaml`, using the exact model IDs from each console. The evaluation model is a separate key in the same file, so switching evaluation to a paid key is a config change only.

```python
def classify(err) -> str:
    msg = str(err)
    if "Request too large" in msg or getattr(err, "status_code", None) == 413:
        return "TOO_BIG"  # S3: skip models with smaller TPM
    if "PerDay" in msg or "per day" in msg.lower():
        return "DAILY"  # S2: mark exhausted, next model
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "MINUTE"  # S1: wait retry-after, retry same model
    return "OTHER"  # 5xx etc: retry with backoff


class QuotaRouter:
    def __init__(self, chains, tpm, state_file="quota_state.json"):
        self.chains, self.tpm = chains, tpm
        self.exhausted = load(state_file)  # {model: reset_time}

    def pick(self, role, prompt_tokens):
        for m in self.chains[role]:
            if m in self.exhausted:
                continue
            if prompt_tokens > self.tpm[m] * 0.9:
                continue
            return m
        raise AllExhausted(role)  # S8: checkpoint + partial result
```

Each agent node calls `router.pick(role, count_tokens(prompt))`, builds the model with `init_chat_model`, and on a `DAILY` or `TOO_BIG` error re-runs the whole agent task with the next pick (S4). Free-tier Gemini requests may be used to improve Google's products, so test only on public repositories.
