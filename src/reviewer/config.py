"""Configuration: ``config/review.yaml`` (D12 spawn limit) and ``config/models.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt, model_validator


class SpawnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_py_files: PositiveInt
    max_specialist_prompt_tokens: PositiveInt


class AgentLimits(BaseModel):
    """Call limits for one agent (plan § Production concerns). Reaching ``max_model_calls`` or
    ``max_tool_calls`` ends the review cleanly with status ``limit_reached`` (decision 0007).
    ``max_run_tests_calls`` caps ``run_tests`` separately; past it the tool is refused and the
    review goes on (decision 0008). It is set only for agents that have ``run_tests``."""

    model_config = ConfigDict(extra="forbid")

    max_model_calls: PositiveInt
    max_tool_calls: PositiveInt
    max_run_tests_calls: PositiveInt | None = None


class LimitsConfig(BaseModel):
    """``limits.<agent name>``: the stage 1 single agent, the stage 2 specialists and the
    orchestrator."""

    model_config = ConfigDict(extra="forbid")

    single: AgentLimits
    bug_reviewer: AgentLimits
    security_reviewer: AgentLimits
    test_reviewer: AgentLimits
    orchestrator: AgentLimits

    def for_agent(self, name: str) -> AgentLimits:
        if name not in type(self).model_fields:
            raise KeyError(
                f"no limits for agent {name!r}; known: {sorted(type(self).model_fields)}"
            )
        return getattr(self, name)


class A2AAgentConfig(BaseModel):
    """How the orchestrator reaches one specialist (stage 3, decision 0011): in this process
    (``local``) or over A2A at ``base_url`` (``remote``)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["local", "remote"]
    base_url: str


class A2AConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    security_reviewer: A2AAgentConfig


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spawn: SpawnConfig
    limits: LimitsConfig
    a2a: A2AConfig = A2AConfig(
        security_reviewer=A2AAgentConfig(mode="local", base_url="http://127.0.0.1:2025")
    )

    def with_security_mode(self, mode: Literal["local", "remote"]) -> ReviewConfig:
        """A copy with the security specialist's mode replaced (the CLI's ``--security-mode``)."""
        agent = self.a2a.security_reviewer.model_copy(update={"mode": mode})
        return self.model_copy(update={"a2a": A2AConfig(security_reviewer=agent)})


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
    # USD per million tokens, for cost estimates only (D5); set on the evaluation model.
    price_input_per_mtok: NonNegativeFloat | None = None
    price_output_per_mtok: NonNegativeFloat | None = None


class Chains(BaseModel):
    """Per-role model chains for normal runs, tried left to right."""

    model_config = ConfigDict(extra="forbid")

    orchestrator: list[str]
    specialist: list[str]
    single: list[str]  # stage 1 baseline reviewer
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
