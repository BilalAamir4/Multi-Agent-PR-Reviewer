"""Integration: one dataset repository's suite passes in its sandbox image at image_commit."""

import pytest

from eval.check_repos import parse_pytest_summary
from eval.repos import clone_dir, get_repo, run_repo_tests, snapshot_files
from reviewer.sandbox import SandboxUnavailable, get_client

pytestmark = pytest.mark.integration

NAME = "marshmallow"  # the fastest suite (about 2 s)


def test_suite_passes_in_sandbox_at_image_commit():
    spec = get_repo(NAME)
    try:
        client = get_client()
    except SandboxUnavailable as e:
        pytest.skip(str(e))
    try:
        client.images.get(spec.image)
    except Exception:
        pytest.skip(
            f"image {spec.image} not built: docker build -t {spec.image} sandbox_images/{NAME}"
        )
    finally:
        client.close()
    if not (clone_dir(NAME) / ".git").exists():
        pytest.skip(f"no clone at {clone_dir(NAME)}: run eval.repos.ensure_clone({NAME!r})")

    res = run_repo_tests(NAME, snapshot_files(NAME, spec.image_commit))
    counts = parse_pytest_summary(res.stdout)
    assert res.exit_code == 0 and not res.timed_out, res.stdout[-3000:] + res.stderr[-2000:]
    assert counts.failed == 0 and counts.errors == 0 and counts.passed > 1000, counts.summary
