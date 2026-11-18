"""Ground-truth format for dataset items (D8, D8a, D3).

Each item is a folder ``eval/dataset/<item_id>/`` containing ``pr.diff`` (unified git diff)
and ``ground_truth.json`` (validated by :class:`GroundTruth`).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from reviewer.schemas import ALL_CATEGORIES, PLANTED_CATEGORIES, Category, validate_posix_path

EQUIVALENCES_PATH = Path(__file__).resolve().parent / "equivalences.yaml"
MAX_BUGS_PER_ITEM = 3


class DatasetError(ValueError):
    """A dataset item or input file is missing or invalid."""


def describe_validation_error(e: ValidationError) -> str:
    """Compact one-line summary of a pydantic ValidationError."""
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
    )


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace")
    return value


class PlantedBug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bug_id: str  # unique within the item, e.g. "b1"
    file: str  # POSIX path, same rules as Finding.file
    line_start: int = Field(ge=1)  # line numbers in the NEW (post-PR) file
    line_end: int
    kind: Category  # never "other"
    accepted_kinds: list[Category]  # must include kind; never "other"; no duplicates
    description: str  # one sentence: what was planted

    @field_validator("bug_id", "description")
    @classmethod
    def _check_non_blank(cls, v: str) -> str:
        return _non_blank(v)

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str) -> str:
        return validate_posix_path(v)

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v == "other":
            raise ValueError("kind must not be 'other'")
        return v

    @field_validator("accepted_kinds")
    @classmethod
    def _check_accepted(cls, v: list[str]) -> list[str]:
        if "other" in v:
            raise ValueError("accepted_kinds must not contain 'other'")
        if len(set(v)) != len(v):
            raise ValueError("accepted_kinds must not contain duplicates")
        return v

    @model_validator(mode="after")
    def _check_model(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if self.kind not in self.accepted_kinds:
            raise ValueError("accepted_kinds must include kind")
        return self


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=r"^item_\d{3}$")  # matches the folder name
    source_repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")  # "owner/name"
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")  # 40-char lowercase hex SHA
    is_clean: bool
    bugs: list[PlantedBug]  # empty iff is_clean; 1 to 3 bugs otherwise

    @model_validator(mode="after")
    def _check_bugs(self) -> Self:
        if self.is_clean and self.bugs:
            raise ValueError("a clean item must have no bugs")
        if not self.is_clean and not (1 <= len(self.bugs) <= MAX_BUGS_PER_ITEM):
            raise ValueError(f"a non-clean item must have 1 to {MAX_BUGS_PER_ITEM} bugs")
        ids = [b.bug_id for b in self.bugs]
        if len(set(ids)) != len(ids):
            raise ValueError("bug_id values must be unique within an item")
        return self


def read_text_exact(path: Path) -> str:
    """Read a file as UTF-8 without any newline translation."""
    return path.read_bytes().decode("utf-8")


def load_item(item_dir: Path) -> tuple[GroundTruth, str]:
    """Load ``ground_truth.json`` and ``pr.diff`` from one item folder."""
    gt_path = item_dir / "ground_truth.json"
    diff_path = item_dir / "pr.diff"
    for p in (gt_path, diff_path):
        if not p.is_file():
            raise DatasetError(f"{item_dir.name}: missing {p.name} in {item_dir}")
    try:
        gt = GroundTruth.model_validate_json(gt_path.read_bytes())
    except ValidationError as e:
        raise DatasetError(
            f"{item_dir.name}: invalid ground_truth.json: {describe_validation_error(e)}"
        ) from e
    if gt.item_id != item_dir.name:
        raise DatasetError(
            f"{item_dir.name}: item_id {gt.item_id!r} does not match folder name {item_dir.name!r}"
        )
    try:
        diff_text = read_text_exact(diff_path)
    except UnicodeDecodeError as e:
        raise DatasetError(f"{item_dir.name}: pr.diff is not valid UTF-8: {e}") from e
    return gt, diff_text


def parse_diff(diff_text: str) -> PatchSet:
    try:
        return PatchSet(diff_text)
    except UnidiffParseError as e:
        raise DatasetError(f"cannot parse diff: {e}") from e


def changed_py_file_count(diff_text: str) -> int:
    """Number of added or modified ``.py`` files in the diff (deleted files do not count)."""
    return sum(
        1
        for pf in parse_diff(diff_text)
        if not pf.is_removed_file and PurePosixPath(pf.path).suffix == ".py"
    )


def size_bucket(diff_text: str) -> Literal["small", "medium", "large"]:
    """small: 0-3 changed .py files, medium: 4-7, large: 8 or more."""
    n = changed_py_file_count(diff_text)
    if n >= 8:
        return "large"
    if n >= 4:
        return "medium"
    return "small"


@cache
def load_equivalences(path: Path = EQUIVALENCES_PATH) -> dict[str, tuple[str, ...]]:
    """Load the D8a groups: planted kind -> extra accepted finding categories."""
    data = yaml.safe_load(read_text_exact(path)) or {}
    if not isinstance(data, dict):
        raise DatasetError(f"{path}: expected a mapping")
    out: dict[str, tuple[str, ...]] = {}
    for kind, extras in data.items():
        if kind not in PLANTED_CATEGORIES:
            raise DatasetError(f"{path}: unknown planted kind {kind!r}")
        if not isinstance(extras, list) or any(
            e not in ALL_CATEGORIES or e == "other" or e == kind for e in extras
        ):
            raise DatasetError(f"{path}: invalid extras for {kind!r}: {extras!r}")
        out[kind] = tuple(extras)
    return out


def default_accepted_kinds(kind: Category) -> list[Category]:
    """``[kind]`` plus the D8a extras from ``eval/equivalences.yaml``."""
    if kind not in PLANTED_CATEGORIES:
        raise ValueError(f"not a planted kind: {kind!r}")
    return [kind, *load_equivalences().get(kind, ())]
