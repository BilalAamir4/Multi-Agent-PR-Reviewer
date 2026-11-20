"""src/reviewer/agents/output.py: FindingDraft, ReviewOutput and the four post-processing steps
(decision 0007), on the synthetic dev fixture. No network, no model calls."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from reviewer.agents.output import (
    FindingDraft,
    ReviewOutput,
    evidence_lines,
    to_findings,
)
from reviewer.config import repo_root
from reviewer.schemas import ALL_CATEGORIES
from reviewer.tools import ReviewContext
from tests.dev_pr import make_dev_workspace

# shop/catalog.py after the PR, lines 30 to 36 (new-file numbers):
#   30     def page(self, number, size):
#   31         """Products on page ``number`` (1-based) ..."""
#   32         if number < 1 or size < 1:
#   33             raise ValueError("number and size must be at least 1")
#   34         products = self.all()
#   35         start = (number - 1) * size
#   36         return products[start : start + size - 1]
SLICE = "        return products[start : start + size - 1]"
START = "        start = (number - 1) * size"
PRODUCTS = "        products = self.all()"


@pytest.fixture
def ctx(tmp_path) -> ReviewContext:
    return make_dev_workspace(tmp_path)


def draft(**kw) -> dict:
    base = {
        "file": "shop/catalog.py",
        "line_start": 36,
        "line_end": 36,
        "category": "off_by_one",
        "severity": "medium",
        "confidence": 0.8,
        "title": "Page is one item short",
        "explanation": "The slice end drops the last product of every page.",
        "evidence": SLICE,
    }
    base.update(kw)
    return base


def run(ctx, *drafts):
    return to_findings([FindingDraft(**d) for d in drafts], "single", "fake-model", ctx)


def reasons(dropped) -> list[str]:
    return [d.reason for d in dropped]


# --- Schemas ---------------------------------------------------------------------------------


def test_draft_has_finding_fields_without_agent_and_model():
    assert "agent" not in FindingDraft.model_fields
    assert "model" not in FindingDraft.model_fields
    assert set(FindingDraft.model_fields) == {
        "file",
        "line_start",
        "line_end",
        "category",
        "severity",
        "confidence",
        "title",
        "explanation",
        "evidence",
    }


def test_draft_schema_is_plain():
    schema = FindingDraft.model_json_schema()
    assert schema["properties"]["category"]["enum"] == list(ALL_CATEGORIES)
    assert "$defs" not in schema
    assert schema["additionalProperties"] is False
    assert ReviewOutput.model_json_schema()["additionalProperties"] is False


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FindingDraft(**draft(agent="x"))
    with pytest.raises(ValidationError):
        ReviewOutput(summary="s", findings=[], verdict="approve")


# --- Step 1: schema --------------------------------------------------------------------------


def test_valid_draft_kept_with_agent_and_model(ctx):
    kept, dropped = run(ctx, draft())
    assert dropped == []
    assert len(kept) == 1
    assert kept[0].agent == "single" and kept[0].model == "fake-model"
    assert kept[0].evidence == SLICE  # stored unstripped


@pytest.mark.parametrize(
    "bad",
    [
        {"line_start": 36, "line_end": 35},
        {"title": "   "},
        {"file": "shop\\catalog.py"},
        {"file": "/shop/catalog.py"},
    ],
)
def test_schema_failure_dropped(ctx, bad):
    kept, dropped = run(ctx, draft(**bad))
    assert kept == []
    assert reasons(dropped) == ["schema"]
    assert dropped[0].detail  # the validation error text


# --- Step 2: changed file --------------------------------------------------------------------


def test_file_not_in_diff_dropped(ctx):
    kept, dropped = run(ctx, draft(file="shop/__init__.py", evidence='"""A tiny shop'))
    assert kept == []
    assert reasons(dropped) == ["not_in_diff"]


def test_deleted_file_dropped(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "kept.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1 +0,0 @@\n"
        "-z = 3\n"
        "diff --git a/kept.py b/kept.py\n"
        "index 2222222..3333333 100644\n"
        "--- a/kept.py\n"
        "+++ b/kept.py\n"
        "@@ -1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )
    ctx = ReviewContext.from_text(ws, diff)
    gone = draft(file="gone.py", line_start=1, line_end=1, evidence="z = 3")
    kept_ok = draft(file="kept.py", line_start=2, line_end=2, evidence="y = 2")
    kept, dropped = run(ctx, gone, kept_ok)
    assert [f.file for f in kept] == ["kept.py"]
    assert reasons(dropped) == ["not_in_diff"]


def test_no_line_filter_in_post_processing(ctx):
    # Line 26 of shop/catalog.py is an unchanged context line; D9a is the scorer's job.
    ev = "    def all(self):"
    kept, dropped = run(ctx, draft(line_start=26, line_end=26, evidence=ev))
    assert len(kept) == 1 and dropped == []


# --- Step 3: evidence (D11) ------------------------------------------------------------------


def test_evidence_with_different_indentation_accepted(ctx):
    kept, dropped = run(ctx, draft(evidence="return products[start : start + size - 1]  "))
    assert len(kept) == 1 and dropped == []


def test_evidence_with_blank_edges_and_crlf_accepted(ctx):
    ev = "\n\n  " + START.strip() + "\r\n" + SLICE + "\n\n"
    kept, dropped = run(ctx, draft(line_start=35, line_end=36, evidence=ev))
    assert len(kept) == 1 and dropped == []


def test_evidence_spanning_three_lines_accepted(ctx):
    ev = "\n".join([PRODUCTS, START, SLICE])
    kept, dropped = run(ctx, draft(evidence=ev))  # lines 34-36, range 36-36
    assert len(kept) == 1 and dropped == []


def test_evidence_on_wrong_lines_dropped(ctx):
    # The slice is at line 36; a finding claiming line 20 cannot quote it.
    kept, dropped = run(ctx, draft(line_start=20, line_end=20))
    assert kept == []
    assert reasons(dropped) == ["evidence"]
    assert "36-36" in dropped[0].detail


def test_evidence_must_touch_the_range(ctx):
    # Line 33 is within 3 of 36 but no evidence line is inside [36, 36].
    ev = 'raise ValueError("number and size must be at least 1")'
    kept, dropped = run(ctx, draft(evidence=ev))
    assert kept == [] and reasons(dropped) == ["evidence"]


def test_evidence_must_stay_within_three_lines(ctx):
    # Lines 32-36 touch line 36, but line 32 is 4 lines before line_start 36.
    ev = "\n".join(
        [
            "if number < 1 or size < 1:",
            'raise ValueError("number and size must be at least 1")',
            PRODUCTS,
            START,
            SLICE,
        ]
    )
    kept, dropped = run(ctx, draft(evidence=ev))
    assert kept == [] and reasons(dropped) == ["evidence"]
    kept, dropped = run(ctx, draft(line_start=35, line_end=36, evidence=ev))
    assert len(kept) == 1  # 32 is within 3 of 35


def test_non_contiguous_evidence_dropped(ctx):
    ev = "\n".join([PRODUCTS, SLICE])  # lines 34 and 36, skipping 35
    kept, dropped = run(ctx, draft(evidence=ev))
    assert kept == [] and reasons(dropped) == ["evidence"]


def test_evidence_not_in_file_dropped(ctx):
    kept, dropped = run(ctx, draft(evidence="return products[start:start + size]"))
    assert kept == [] and reasons(dropped) == ["evidence"]


def test_blank_evidence_dropped(ctx):
    kept, dropped = run(ctx, draft(evidence="\n  \n"))
    # Finding rejects whitespace-only evidence, so step 1 drops it first.
    assert kept == [] and reasons(dropped) == ["schema"]


def test_evidence_lines_helper():
    assert evidence_lines("\n  a\n\n   b  \n\n") == ["a", "", "b"]


# --- Step 4: duplicates ----------------------------------------------------------------------


def test_duplicate_keeps_higher_confidence(ctx):
    low = draft(confidence=0.4, title="low")
    high = draft(line_start=35, line_end=36, confidence=0.9, title="high", evidence=SLICE)
    kept, dropped = run(ctx, low, high)
    assert [f.title for f in kept] == ["high"]
    assert reasons(dropped) == ["duplicate"]
    assert dropped[0].draft["title"] == "low"


def test_duplicate_tie_keeps_earlier(ctx):
    kept, dropped = run(ctx, draft(title="first"), draft(title="second"))
    assert [f.title for f in kept] == ["first"]
    assert reasons(dropped) == ["duplicate"]


def test_not_duplicate_when_category_or_lines_differ(ctx):
    other_cat = draft(category="other", title="b")
    other_lines = draft(line_start=35, line_end=35, evidence=START, title="c")
    kept, dropped = run(ctx, draft(title="a"), other_cat, other_lines)
    assert [f.title for f in kept] == ["a", "b", "c"]
    assert dropped == []


def test_kept_findings_stay_in_input_order(ctx):
    cart = draft(
        file="shop/cart.py",
        line_start=23,
        line_end=23,
        category="missing_none_check",
        evidence="if coupon.percent > 0:",
        confidence=0.5,
        title="cart",
    )
    kept, _ = run(ctx, cart, draft(confidence=0.9, title="catalog"))
    assert [f.title for f in kept] == ["cart", "catalog"]


# --- Order of the steps ----------------------------------------------------------------------


def test_steps_run_in_order(ctx):
    schema_and_file = draft(file="nope.py", line_start=5, line_end=4)  # fails 1 and 2
    file_and_evidence = draft(file="nope.py", evidence="not there")  # fails 2 and 3
    # A high-confidence draft with bad evidence is dropped at step 3, so it cannot win step 4.
    bad_evidence = draft(confidence=1.0, evidence="not there", title="bad")
    good = draft(confidence=0.3, title="good")
    kept, dropped = run(ctx, schema_and_file, file_and_evidence, bad_evidence, good)
    assert reasons(dropped) == ["schema", "not_in_diff", "evidence"]
    assert [f.title for f in kept] == ["good"]


# --- docs/categories.md ----------------------------------------------------------------------


def test_categories_doc_matches_finding_literal():
    text = (repo_root() / "docs" / "categories.md").read_text(encoding="utf-8")
    assert text.startswith("# Finding categories (D8)\n")
    names = re.findall(r"^- `([a-z_]+)`: \S", text, flags=re.MULTILINE)
    assert tuple(names) == ALL_CATEGORIES
