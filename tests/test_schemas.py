import pytest
from pydantic import ValidationError

from reviewer.schemas import PLANTED_CATEGORIES, Finding

VALID = {
    "file": "pkg/calc.py",
    "line_start": 3,
    "line_end": 5,
    "category": "off_by_one",
    "severity": "high",
    "confidence": 0.75,
    "title": "Loop stops one early",
    "explanation": "range(len(xs) - 1) skips the last element.",
    "evidence": "    for i in range(len(xs) - 1):\n",
    "agent": "bug_reviewer",
    "model": "gemini-3.1-flash-lite",
}


def make(**overrides):
    return Finding(**{**VALID, **overrides})


def test_round_trip():
    f = make()
    again = Finding.model_validate_json(f.model_dump_json())
    assert again == f


def test_field_order_matches_plan():
    assert list(Finding.model_fields) == [
        "file",
        "line_start",
        "line_end",
        "category",
        "severity",
        "confidence",
        "title",
        "explanation",
        "evidence",
        "agent",
        "model",
    ]


def test_planted_categories():
    assert len(PLANTED_CATEGORIES) == 7
    assert "other" not in PLANTED_CATEGORIES


def test_other_category_accepted():
    assert make(category="other").category == "other"


def test_unknown_category_rejected():
    with pytest.raises(ValidationError):
        make(category="style")


def test_unknown_severity_rejected():
    with pytest.raises(ValidationError):
        make(severity="critical")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Finding(**VALID, extra_field=1)


def test_line_start_must_be_positive():
    with pytest.raises(ValidationError):
        make(line_start=0, line_end=0)


def test_line_end_before_start_rejected():
    with pytest.raises(ValidationError):
        make(line_start=5, line_end=4)


def test_single_line_ok():
    assert make(line_start=5, line_end=5).line_end == 5


@pytest.mark.parametrize("c", [-0.01, 1.01])
def test_confidence_out_of_range(c):
    with pytest.raises(ValidationError):
        make(confidence=c)


@pytest.mark.parametrize("c", [0.0, 1.0])
def test_confidence_bounds_ok(c):
    assert make(confidence=c).confidence == c


@pytest.mark.parametrize(
    "path",
    [
        "",
        "./",
        "pkg\\calc.py",
        "/pkg/calc.py",
        "C:/pkg/calc.py",
        "c:pkg.py",
        "../x.py",
        "a/../x.py",
    ],
)
def test_bad_file_rejected(path):
    with pytest.raises(ValidationError):
        make(file=path)


def test_leading_dot_slash_stripped_once():
    assert make(file="./pkg/calc.py").file == "pkg/calc.py"


def test_a_b_prefix_not_stripped():
    assert make(file="b/pkg/calc.py").file == "b/pkg/calc.py"


@pytest.mark.parametrize("field", ["title", "explanation", "evidence", "agent", "model"])
@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_blank_text_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_evidence_kept_exactly():
    ev = "    x = y  \n"
    assert make(evidence=ev).evidence == ev
