from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from reviewer.config import load_models_config, repo_root

REAL = repo_root() / "config" / "models.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _real_data() -> dict:
    return yaml.safe_load(REAL.read_text(encoding="utf-8"))


def test_real_file_loads():
    cfg = load_models_config()
    assert cfg.evaluation.model == "gemini-3.1-flash-lite"
    assert cfg.evaluation.cross_model_fallback is False
    assert cfg.chains.specialist[-1] == "openai/gpt-oss-120b"
    for name in ("gemini-4-hydrogen", "meta-llama/llama-prompt-guard-2-86m"):
        assert name in cfg.models
        assert all(name not in chain for chain in cfg.chains.model_dump().values())


def test_default_path_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_models_config().evaluation.model == "gemini-3.1-flash-lite"


def test_unknown_model_in_chain_rejected(tmp_path: Path):
    data = _real_data()
    data["chains"]["worker"].append("no-such-model")
    with pytest.raises(ValidationError, match="unknown model 'no-such-model'"):
        load_models_config(_write(tmp_path, data))


def test_missing_evaluation_model_rejected(tmp_path: Path):
    data = _real_data()
    data["evaluation"]["model"] = "no-such-model"
    with pytest.raises(ValidationError, match="evaluation model"):
        load_models_config(_write(tmp_path, data))


def test_empty_chain_rejected(tmp_path: Path):
    data = _real_data()
    data["chains"]["verifier"] = []
    with pytest.raises(ValidationError, match="must not be empty"):
        load_models_config(_write(tmp_path, data))


def test_cross_model_fallback_true_rejected(tmp_path: Path):
    data = _real_data()
    data["evaluation"]["cross_model_fallback"] = True
    with pytest.raises(ValidationError):
        load_models_config(_write(tmp_path, data))


def test_unknown_key_rejected(tmp_path: Path):
    data = _real_data()
    data["models"]["gemini-3.1-flash-lite"]["extra"] = 1
    with pytest.raises(ValidationError):
        load_models_config(_write(tmp_path, data))
