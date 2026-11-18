from pathlib import Path

import pytest
from pydantic import ValidationError

from reviewer.config import SpawnConfig, load_review_config, repo_root, should_spawn


def test_real_file_loads():
    cfg = load_review_config()
    assert cfg.spawn.min_py_files == 4
    assert cfg.spawn.max_specialist_prompt_tokens == 30000


def test_default_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_review_config().spawn.min_py_files == 4
    assert (repo_root() / "pyproject.toml").is_file()


def test_unknown_key_rejected(tmp_path: Path):
    p = tmp_path / "review.yaml"
    p.write_text(
        "spawn:\n  min_py_files: 4\n  max_specialist_prompt_tokens: 30000\n  extra: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_review_config(p)


def test_unknown_top_level_key_rejected(tmp_path: Path):
    p = tmp_path / "review.yaml"
    p.write_text(
        "spawn:\n  min_py_files: 4\n  max_specialist_prompt_tokens: 30000\nother: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_review_config(p)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_rejected(bad):
    with pytest.raises(ValidationError):
        SpawnConfig(min_py_files=bad, max_specialist_prompt_tokens=30000)


@pytest.mark.parametrize(
    ("files", "tokens", "expected"),
    [(3, 30000, False), (4, 30000, True), (3, 30001, True)],
)
def test_should_spawn(files, tokens, expected):
    cfg = SpawnConfig(min_py_files=4, max_specialist_prompt_tokens=30000)
    assert should_spawn(files, tokens, cfg) is expected
