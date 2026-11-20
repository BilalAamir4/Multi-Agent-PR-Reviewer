import pytest

# Fixture projects contain their own test_*.py files; they are data, not tests of this repo.
collect_ignore_glob = ["fixtures/*"]


@pytest.fixture(autouse=True)
def _no_tracing_in_unit_tests(request, monkeypatch):
    """Unit tests make no network calls: LangSmith tracing stays off even when a test runs code
    that loads ``.env`` (``load_dotenv`` never overrides a variable that is already set)."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
