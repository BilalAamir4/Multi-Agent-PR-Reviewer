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


@pytest.fixture(autouse=True)
def _spawning_off_by_default(monkeypatch):
    """The stage 1 to 3 tests describe a review with spawning off (decision 0012, rule 2: with
    spawning off, behaviour is identical to stage 3). ``config/review.yaml`` has ``spawn.enabled:
    true`` and several dev fixtures have 4 or more ``.py`` files, so every place that reads the
    config for a review reads it with spawning switched off here. A test of spawning passes its
    own ``ReviewConfig`` (``tests/test_subagents.py``)."""
    import reviewer.agents.judge as judge
    import reviewer.agents.orchestrator as orchestrator
    import reviewer.agents.specialists as specialists
    import reviewer.agents.verifier as verifier
    import reviewer.cli as cli
    import reviewer.graph as graph
    from a2a_security import graph as service
    from eval import run as eval_run

    real = specialists.load_review_config

    def spawn_off(path=None):
        return real(path).with_spawn(enabled=False)

    for module in (specialists, orchestrator, cli, service, eval_run, graph, verifier, judge):
        monkeypatch.setattr(module, "load_review_config", spawn_off)
