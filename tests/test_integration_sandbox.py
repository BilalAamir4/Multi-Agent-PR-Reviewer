"""Integration: the sandbox probe passes all five checks (stage 0 (b))."""

import pytest

from reviewer.sandbox import SandboxUnavailable, get_client
from scripts import check_sandbox

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def result():
    try:
        get_client().close()
    except SandboxUnavailable as e:
        pytest.skip(str(e))
    return check_sandbox.run_check()


def test_probe_passes_all_checks(result):
    assert result.hello, result.probe.stdout
    for name in check_sandbox.CHECKS:
        assert result.checks.get(name), f"{name}: {result.details.get(name)}"


def test_timeout_kills_and_removes(result):
    assert result.timeout.timed_out
    assert result.leftover_containers == 0
    assert result.leftover_volumes == 0
