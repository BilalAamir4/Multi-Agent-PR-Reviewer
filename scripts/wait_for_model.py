"""Wait until the evaluation model answers again (tooling only; nothing frozen changes).

Every 20 minutes one probe call to the evaluation model (a one-line prompt, ``max_retries=0``,
as in ``scripts/check_finding.py``). Exit 0 after 2 probes in a row succeed; exit 1 after 8 hours
without that. Each probe is appended to ``logs/wait_for_model.log`` (time, result) and counts as
one model call toward the day's budget.

    uv run python scripts/wait_for_model.py [--interval-min 20] [--max-hours 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "wait_for_model.log"


def log(line: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def probe() -> tuple[bool, str]:
    from langchain.chat_models import init_chat_model

    from reviewer.config import load_models_config

    model_id = load_models_config().evaluation.model
    llm = init_chat_model(model_id, model_provider="google_genai", temperature=0, max_retries=0)
    try:
        llm.invoke("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001 - report whatever came back
        return False, f"{type(e).__qualname__}: {str(e)[:160]}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-min", type=float, default=20)
    ap.add_argument("--max-hours", type=float, default=8)
    args = ap.parse_args()
    load_dotenv(REPO / ".env")
    start = time.monotonic()
    streak = probes = 0
    while True:
        ok, detail = probe()
        probes += 1
        streak = streak + 1 if ok else 0
        stamp = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
        log(f"{stamp} probe {probes} {'OK' if ok else 'FAIL'} streak={streak} {detail}")
        if streak >= 2:
            log(f"model available after {probes} probes (counted toward today's budget)")
            return 0
        if time.monotonic() - start + args.interval_min * 60 > args.max_hours * 3600:
            log(f"gave up after {probes} probes, no 2 successes in a row")
            return 1
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.exit(main())
