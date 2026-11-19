"""List the models available to the Gemini and Groq keys and write docs/model-inventory.md.

Keys are read from .env and sent only in headers; they are never printed or written.
Listing models does not count against the generation quotas.

    uv run python scripts/list_models.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "model-inventory.md"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_URL = "https://api.groq.com/openai/v1/models"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} is missing from .env")
    return value


def list_gemini(client: httpx.Client, key: str) -> list[dict]:
    models: list[dict] = []
    params: dict[str, str] = {"pageSize": "1000"}
    while True:
        r = client.get(GEMINI_URL, headers={"x-goog-api-key": key}, params=params)
        r.raise_for_status()
        body = r.json()
        for m in body.get("models", []):
            models.append(
                {
                    "name": m.get("name", ""),
                    "displayName": m.get("displayName", ""),
                    "inputTokenLimit": m.get("inputTokenLimit"),
                    "outputTokenLimit": m.get("outputTokenLimit"),
                    "supportedGenerationMethods": m.get("supportedGenerationMethods", []),
                }
            )
        token = body.get("nextPageToken")
        if not token:
            break
        params = {"pageSize": "1000", "pageToken": token}
    return sorted(models, key=lambda m: m["name"])


def list_groq(client: httpx.Client, key: str) -> list[dict]:
    r = client.get(GROQ_URL, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    models = [
        {
            "id": m.get("id", ""),
            "owned_by": m.get("owned_by", ""),
            "context_window": m.get("context_window"),
        }
        for m in r.json().get("data", [])
    ]
    return sorted(models, key=lambda m: m["id"])


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value).replace("|", "\\|")


def render(gemini: list[dict], groq: list[dict], today: dt.date) -> str:
    lines = [
        "# Model inventory",
        "",
        f"Listed on {today.isoformat()} with `scripts/list_models.py` from the keys in `.env`.",
        "",
        f"## Gemini API ({len(gemini)} models)",
        "",
        "| name | displayName | inputTokenLimit | outputTokenLimit | supportedGenerationMethods |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in gemini:
        lines.append(
            "| "
            + " | ".join(
                _cell(m[k])
                for k in (
                    "name",
                    "displayName",
                    "inputTokenLimit",
                    "outputTokenLimit",
                    "supportedGenerationMethods",
                )
            )
            + " |"
        )
    lines += [
        "",
        f"## Groq ({len(groq)} models)",
        "",
        "| id | owned_by | context_window |",
        "| --- | --- | --- |",
    ]
    for m in groq:
        lines.append(
            "| " + " | ".join(_cell(m[k]) for k in ("id", "owned_by", "context_window")) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv(REPO / ".env")
    google_key = _require("GOOGLE_API_KEY")
    groq_key = _require("GROQ_API_KEY")
    with httpx.Client(timeout=30) as client:
        try:
            gemini = list_gemini(client, google_key)
            groq = list_groq(client, groq_key)
        except httpx.HTTPStatusError as e:
            # The URL carries no key (keys go in headers), so it is safe to show.
            sys.exit(f"HTTP {e.response.status_code} from {e.request.url}: {e.response.text[:500]}")
    OUT.write_bytes(render(gemini, groq, dt.date.today()).encode("utf-8"))
    print(f"Gemini: {len(gemini)} models, Groq: {len(groq)} models -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
