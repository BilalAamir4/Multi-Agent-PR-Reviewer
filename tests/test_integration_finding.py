"""Integration: one traced call to the evaluation model returns a valid Finding (stage 0 (a))."""

import pytest

from scripts import check_finding

pytestmark = pytest.mark.integration


def test_traced_call_returns_finding():
    missing = check_finding.missing_requirements()
    if missing:
        pytest.skip(f"missing from .env: {', '.join(missing)}")
    res = check_finding.run_check()
    assert res.ok, res.error
    assert res.trace_url, res.trace_error
    assert res.key_in_trace is False
