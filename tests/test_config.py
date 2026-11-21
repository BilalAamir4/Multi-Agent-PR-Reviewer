from pathlib import Path

import pytest
from pydantic import ValidationError

from reviewer.config import (
    AgentLimits,
    SpawnConfig,
    load_review_config,
    repo_root,
    should_spawn,
)

SPAWN = (
    "spawn:\n  enabled: true\n  min_py_files: 4\n  max_specialist_prompt_tokens: 30000\n"
    "  max_files_per_subagent: 3\n  max_subagents: 5\n  max_concurrency: 3\n"
    "  batch_token_budget: 20000\n"
)
SPAWN_ARGS = dict(
    enabled=True, min_py_files=4, max_specialist_prompt_tokens=30000, max_files_per_subagent=3,
    max_subagents=5, max_concurrency=3, batch_token_budget=20000,
)  # fmt: skip
LIMITS = (
    "limits:\n"
    "  single:\n    max_model_calls: 25\n    max_tool_calls: 40\n"
    "  bug_reviewer:\n    max_model_calls: 20\n    max_tool_calls: 30\n"
    "  security_reviewer:\n    max_model_calls: 15\n    max_tool_calls: 25\n"
    "  test_reviewer:\n    max_model_calls: 20\n    max_tool_calls: 30\n"
    "    max_run_tests_calls: 3\n"
    "  orchestrator:\n    max_model_calls: 8\n    max_tool_calls: 12\n"
    "  subagent:\n    max_model_calls: 15\n    max_tool_calls: 20\n    max_run_tests_calls: 1\n"
    "  verifier_per_finding:\n    max_model_calls: 6\n    max_tool_calls: 8\n"
    "  judge:\n    max_model_calls: 8\n    max_tool_calls: 12\n"
)


def _review_yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "review.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_real_file_loads():
    cfg = load_review_config()
    assert cfg.spawn.min_py_files == 4
    assert cfg.spawn.max_specialist_prompt_tokens == 30000
    assert cfg.spawn == SpawnConfig(**SPAWN_ARGS)  # D12 as changed by decision 0012
    assert cfg.limits.subagent == AgentLimits(
        max_model_calls=15, max_tool_calls=20, max_run_tests_calls=1
    )
    assert cfg.limits.verifier_per_finding == AgentLimits(max_model_calls=6, max_tool_calls=8)
    assert cfg.limits.judge == AgentLimits(max_model_calls=8, max_tool_calls=12)
    assert cfg.verifier.max_concurrency == 3
    assert cfg.limits.single.max_model_calls == 25
    assert cfg.limits.single.max_tool_calls == 40
    assert cfg.limits.single.max_run_tests_calls is None
    assert cfg.limits.bug_reviewer == AgentLimits(max_model_calls=20, max_tool_calls=30)
    assert cfg.limits.security_reviewer == AgentLimits(max_model_calls=15, max_tool_calls=25)
    assert cfg.limits.test_reviewer == AgentLimits(
        max_model_calls=20, max_tool_calls=30, max_run_tests_calls=3
    )


def test_default_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_review_config().spawn.min_py_files == 4
    assert (repo_root() / "pyproject.toml").is_file()


def test_unknown_key_rejected(tmp_path: Path):
    p = _review_yaml(tmp_path, SPAWN + "  extra: 1\n" + LIMITS)
    with pytest.raises(ValidationError):
        load_review_config(p)


def test_unknown_top_level_key_rejected(tmp_path: Path):
    p = _review_yaml(tmp_path, SPAWN + LIMITS + "other: 1\n")
    with pytest.raises(ValidationError):
        load_review_config(p)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_rejected(bad):
    with pytest.raises(ValidationError):
        SpawnConfig(**{**SPAWN_ARGS, "min_py_files": bad})


@pytest.mark.parametrize(
    ("files", "tokens", "expected"),
    [(3, 30000, False), (4, 30000, True), (3, 30001, True)],
)
def test_should_spawn(files, tokens, expected):
    cfg = SpawnConfig(**SPAWN_ARGS)
    assert should_spawn(files, tokens, cfg) is expected


def test_valid_limits_load(tmp_path: Path):
    cfg = load_review_config(_review_yaml(tmp_path, SPAWN + LIMITS))
    assert cfg.limits.single == AgentLimits(max_model_calls=25, max_tool_calls=40)


def test_missing_limits_rejected(tmp_path: Path):
    with pytest.raises(ValidationError):
        load_review_config(_review_yaml(tmp_path, SPAWN))


def test_unknown_limit_key_rejected(tmp_path: Path):
    with pytest.raises(ValidationError):
        load_review_config(_review_yaml(tmp_path, SPAWN + LIMITS + "    max_tokens: 5\n"))


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_limits_rejected(bad):
    with pytest.raises(ValidationError):
        AgentLimits(max_model_calls=bad, max_tool_calls=40)
    with pytest.raises(ValidationError):
        AgentLimits(max_model_calls=25, max_tool_calls=bad)


def test_for_agent():
    limits = load_review_config().limits
    assert limits.for_agent("single") is limits.single
    assert limits.for_agent("test_reviewer").max_run_tests_calls == 3
    with pytest.raises(KeyError, match="no limits for agent"):
        limits.for_agent("nobody")


def test_missing_specialist_limits_rejected(tmp_path: Path):
    text = SPAWN + "limits:\n  single:\n    max_model_calls: 25\n    max_tool_calls: 40\n"
    with pytest.raises(ValidationError):
        load_review_config(_review_yaml(tmp_path, text))


def test_non_positive_run_tests_limit_rejected():
    with pytest.raises(ValidationError):
        AgentLimits(max_model_calls=1, max_tool_calls=1, max_run_tests_calls=0)


@pytest.mark.parametrize(
    "key",
    ["min_py_files", "max_files_per_subagent", "max_subagents", "max_concurrency",
     "batch_token_budget"],
)  # fmt: skip
def test_every_new_spawn_number_must_be_positive_and_present(key):
    with pytest.raises(ValidationError):
        SpawnConfig(**{**SPAWN_ARGS, key: 0})
    missing = {k: v for k, v in SPAWN_ARGS.items() if k != key}
    with pytest.raises(ValidationError):
        SpawnConfig(**missing)


def test_spawn_switch_is_required_and_a_bool():
    with pytest.raises(ValidationError):
        SpawnConfig(**{k: v for k, v in SPAWN_ARGS.items() if k != "enabled"})
    with pytest.raises(ValidationError):
        SpawnConfig(**{**SPAWN_ARGS, "enabled": "maybe"})


def test_missing_subagent_limits_rejected(tmp_path: Path):
    text = SPAWN + LIMITS.split("  subagent:")[0]
    with pytest.raises(ValidationError):
        load_review_config(_review_yaml(tmp_path, text))


def test_with_spawn_changes_only_what_it_is_given():
    cfg = load_review_config()
    off = cfg.with_spawn(enabled=False)
    assert off.spawn == cfg.spawn.model_copy(update={"enabled": False})
    wide = cfg.with_spawn(max_concurrency=1)
    assert (wide.spawn.max_concurrency, wide.spawn.enabled) == (1, cfg.spawn.enabled)
    assert cfg.with_spawn() == cfg
    assert off.limits == cfg.limits and off.a2a == cfg.a2a
    with pytest.raises(ValidationError):
        cfg.with_spawn(max_concurrency=0)
    assert cfg.limits.for_agent("subagent") is cfg.limits.subagent
