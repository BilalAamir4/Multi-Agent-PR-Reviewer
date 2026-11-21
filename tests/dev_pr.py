"""The synthetic dev fixtures: the only things prompts are developed against (stage 1B, 2A).

``tests/fixtures/<name>/workspace/`` is a tiny project after the PR, ``pr.diff`` the PR, and
``expected.json`` its deliberate defects (informational only: no test asserts they are found).
None of them is part of the evaluation.

- ``dev_pr``: coupons and catalog pages (logic defects; ``tests/test_catalog.py`` fails on it).
- ``dev_security_pr``: a SQL query built with string formatting and a hardcoded password.
- ``dev_tests_pr``: a weakened test and a new helper with a bug that ``test_clamp`` catches.
- ``dev_docs_pr``: ``README.md`` and ``docs/usage.md`` only (for the stage 2B routing test).
- ``dev_risky_pr``: SQL, a credential, logic and tests at once (for the stage 2B routing test).
- ``dev_logic_pr``: a logic-only change with no security surface: a wrong perimeter helper and a
  weak test (stage 2B, informational: does the orchestrator skip the security reviewer?).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reviewer.tools import ReviewContext, SandboxSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DEV_PR = FIXTURES / "dev_pr"
TITLE = "Add coupons and catalog pages"
BODY = "Adds percentage coupons to the cart and a page() helper to the catalog."

PR_TEXT: dict[str, tuple[str, str]] = {
    "dev_pr": (TITLE, BODY),
    "dev_security_pr": (
        "Add user search and an admin login",
        "Adds a search endpoint over the user directory and a login check for the admin console.",
    ),
    "dev_tests_pr": (
        "Add median() and tidy the clamp code",
        "Adds median(), moves clamp's limiting into a helper and simplifies a test.",
    ),
    "dev_docs_pr": ("Improve the README and usage docs", "Documentation only."),
    "dev_logic_pr": (
        "Add rectangle_perimeter() and average_side()",
        "Adds two geometry helpers and tests for them.",
    ),
    "dev_risky_pr": (
        "Add per-account entries, running balances and a nightly sync URL",
        "Adds entries_for(), running_balances() and sync_url(), and updates the report tests.",
    ),
}


def pr_text(name: str) -> tuple[str, str]:
    return PR_TEXT[name]


def expected_defects(name: str = "dev_pr") -> list[dict]:
    return json.loads((FIXTURES / name / "expected.json").read_bytes())["defects"]


def make_fixture_workspace(
    name: str, tmp: Path, sandbox: SandboxSpec | None = None
) -> ReviewContext:
    """Copy ``tests/fixtures/<name>/workspace`` into ``tmp/workspace`` and return its
    ``ReviewContext`` (with ``sandbox`` for ``run_tests``)."""
    workspace = Path(tmp) / "workspace"
    shutil.copytree(FIXTURES / name / "workspace", workspace)
    return ReviewContext.from_paths(workspace, FIXTURES / name / "pr.diff", sandbox)


def make_dev_workspace(tmp: Path) -> ReviewContext:
    """Copy the ``dev_pr`` workspace into ``tmp/workspace`` and return its ``ReviewContext``."""
    return make_fixture_workspace("dev_pr", tmp)
