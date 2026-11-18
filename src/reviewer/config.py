"""Review configuration loaded from ``config/review.yaml`` (D12 spawn limit)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt


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


def should_spawn(changed_py_files: int, specialist_prompt_tokens: int, cfg: SpawnConfig) -> bool:
    """D12: spawn per-file workers when either limit is reached."""
    return (
        changed_py_files >= cfg.min_py_files
        or specialist_prompt_tokens > cfg.max_specialist_prompt_tokens
    )
