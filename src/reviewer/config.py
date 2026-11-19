"""Configuration: ``config/review.yaml`` (D12 spawn limit) and ``config/models.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator


class SpawnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_py_files: PositiveInt
    max_specialist_prompt_tokens: PositiveInt


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spawn: SpawnConfig


def repo_root() -> Path:
    """Walk up from this module until a directory containing ``pyproject.toml`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"no pyproject.toml found above {here}")


def load_review_config(path: Path | None = None) -> ReviewConfig:
    """Load and validate the review config. Defaults to ``<repo root>/config/review.yaml``."""
    if path is None:
        path = repo_root() / "config" / "review.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ReviewConfig.model_validate(data)


class ModelSpec(BaseModel):
    """One model's provider and free-tier limits (routing doc quota table)."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["google_genai", "groq"]
    rpm: PositiveInt | None
    tpm: PositiveInt | None
    rpd: PositiveInt | None  # requests per day
    tpd: PositiveInt | None  # tokens per day (Groq only)
    tool_calling: bool | Literal["unknown"]


class Chains(BaseModel):
    """Per-role model chains for normal runs, tried left to right."""

    model_config = ConfigDict(extra="forbid")

    orchestrator: list[str]
    specialist: list[str]
    worker: list[str]
    verifier: list[str]


class EvaluationConfig(BaseModel):
    """D4 / S9: one pinned model for every role, no cross-model fallback."""

    model_config = ConfigDict(extra="forbid")

    model: str
    cross_model_fallback: Literal[False]


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelSpec]
    chains: Chains
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        for role, chain in self.chains.model_dump().items():
            if not chain:
                raise ValueError(f"chain {role!r} must not be empty")
            for name in chain:
                if name not in self.models:
                    raise ValueError(f"chain {role!r} names unknown model {name!r}")
        if self.evaluation.model not in self.models:
            raise ValueError(f"evaluation model {self.evaluation.model!r} is not under models")
        return self


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and validate the models config. Defaults to ``<repo root>/config/models.yaml``."""
    if path is None:
        path = repo_root() / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelsConfig.model_validate(data)


def should_spawn(changed_py_files: int, specialist_prompt_tokens: int, cfg: SpawnConfig) -> bool:
    """D12: spawn per-file workers when either limit is reached."""
    return (
        changed_py_files >= cfg.min_py_files
        or specialist_prompt_tokens > cfg.max_specialist_prompt_tokens
    )
