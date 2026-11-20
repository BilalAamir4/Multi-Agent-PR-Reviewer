"""The synthetic dev fixture (stage 1B, Task 5): the only thing prompts are developed against.

``tests/fixtures/dev_pr/workspace/`` is a tiny project after the PR, ``pr.diff`` the PR, and
``expected.json`` its two deliberate defects. It is not part of the evaluation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from reviewer.tools import ReviewContext

DEV_PR = Path(__file__).resolve().parent / "fixtures" / "dev_pr"
TITLE = "Add coupons and catalog pages"
BODY = "Adds percentage coupons to the cart and a page() helper to the catalog."


def expected_defects() -> list[dict]:
    return json.loads((DEV_PR / "expected.json").read_bytes())["defects"]


def make_dev_workspace(tmp: Path) -> ReviewContext:
    """Copy the fixture's workspace into ``tmp/workspace`` and return its ``ReviewContext``."""
    workspace = Path(tmp) / "workspace"
    shutil.copytree(DEV_PR / "workspace", workspace)
    return ReviewContext.from_paths(workspace, DEV_PR / "pr.diff")
