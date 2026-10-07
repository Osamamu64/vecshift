"""The job spec, usually ``vecshift.yaml``.

One file describes a migration: where the vectors are, where the new ones go, which model
makes them, and the limits to respect. The CLI reads it today; the API and UI will read and
write the same file. It never holds credentials: the database connection string comes
from an environment variable the file names.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vecshift.embeddings.spec import SpecError, parse_spec

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


class JobError(ValueError):
    """The job file couldn't be read or isn't valid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _identifier(value: str | None, what: str) -> str | None:
    if value is not None and not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain identifier, got {value!r}")
    return value


class Source(_Strict):
    type: Literal["pgvector"] = "pgvector"
    dsn_env: str = "VECSHIFT_DSN"
    """Environment variable holding the connection string. The file never stores it."""
    table: str
    """``table`` or ``schema.table``."""
    vector_column: str | None = None
    text_column: str | None = None

    @field_validator("table")
    @classmethod
    def _table(cls, value: str) -> str:
        for part in value.split("."):
            _identifier(part, "table")
        if value.count(".") > 1:
            raise ValueError("table must be table or schema.table")
        return value

    @field_validator("vector_column", "text_column")
    @classmethod
    def _columns(cls, value: str | None) -> str | None:
        return _identifier(value, "column")


class VectorType(StrEnum):
    VECTOR = "vector"
    HALFVEC = "halfvec"


class IndexMethod(StrEnum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    NONE = "none"


class Metric(StrEnum):
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"


class Target(_Strict):
    column: str = "embedding_v2"
    """New column on the source table. Cutover swaps it with the old one by renaming."""
    vector_type: VectorType = VectorType.VECTOR
    index: IndexMethod = IndexMethod.HNSW
    metric: Metric | None = None
    """Distance for the index. Defaults to the existing index's metric, or cosine."""

    @field_validator("column")
    @classmethod
    def _column(cls, value: str) -> str:
        _identifier(value, "column")
        return value


class Limits(_Strict):
    budget_usd: float | None = Field(default=None, gt=0)
    """Stop before spending more than this. Plan fails when the estimate exceeds it."""
    tokens_per_minute: int | None = Field(default=None, gt=0)
    """The provider's rate limit, used to estimate duration."""
    requests_per_minute: int | None = Field(default=None, gt=0)


class JobSpec(_Strict):
    version: Literal[1] = 1
    name: str = "migration"
    source: Source
    target: Target = Target()
    model: str
    """A model spec, as for ``vecshift bench``: ``provider/model[,option=value]``."""
    limits: Limits = Limits()

    @field_validator("target", "limits", mode="before")
    @classmethod
    def _empty_section(cls, value: object) -> object:
        # A section whose settings are all commented out reads as null: use the defaults.
        return {} if value is None else value

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        try:
            spec = parse_spec(value)
        except SpecError as exc:
            raise ValueError(str(exc)) from None
        if spec.provider == "hash":
            raise ValueError("hash models are baselines for bench, not for migrating")
        return value


def format_errors(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "file"
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "extra_forbidden":
            message = "unknown setting"
        elif item["type"] == "missing":
            message = "required"
        lines.append(f"  {where}: {message}")
    return "\n".join(lines)


def load_job(path: Path) -> JobSpec:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JobError(f"Couldn't read {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark else ""
        raise JobError(f"{path} isn't valid YAML{where}") from exc
    if not isinstance(raw, dict):
        raise JobError(f"{path} should contain settings, starting with source: and model:")
    try:
        return JobSpec.model_validate(raw)
    except ValidationError as exc:
        raise JobError(f"{path} has problems:\n{format_errors(exc)}") from exc


def template(table: str, model: str, column: str = "embedding_v2") -> str:
    """A commented job file to start from. Values are quoted, so any input stays valid YAML."""
    name = json.dumps(table.split(".")[-1] + "-reembed")
    return f"""# vecshift job: re-embed a pgvector column with a new model, side by side.
# Check it with `vecshift plan`. Nothing is changed until `vecshift apply`.
version: 1
name: {name}

source:
  type: pgvector
  # The connection string is read from this environment variable, never from this file.
  dsn_env: VECSHIFT_DSN
  table: {json.dumps(table)}
  # vector_column: embedding     # needed only if the table has several
  # text_column: content         # detected automatically when it's a common name

target:
  # New vectors go in this column next to the old ones. Cutover renames the two columns
  # in one transaction, and rollback renames them back.
  column: {json.dumps(column)}
  vector_type: vector            # or halfvec: half the storage, indexable up to 4,000 dims
  index: hnsw                    # hnsw, ivfflat, or none
  # metric: cosine               # defaults to the current index's metric, or cosine

# The new model, written as for `vecshift bench`, e.g. openai/text-embedding-3-large,dims=1024
model: {json.dumps(model)}

limits:
  # budget_usd: 50               # plan fails if the estimate is higher; apply stops there
  # tokens_per_minute: 1000000   # your provider rate limit, for the duration estimate
  # requests_per_minute: 3000
"""
