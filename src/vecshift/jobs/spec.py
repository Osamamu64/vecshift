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
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from vecshift.embeddings.spec import SpecError, parse_spec

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_COLLECTION = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,254}$")
_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class JobError(ValueError):
    """The job file couldn't be read or isn't valid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _identifier(value: str | None, what: str) -> str | None:
    if value is not None and not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain identifier, got {value!r}")
    return value


def _collection(value: str | None) -> str | None:
    if value is not None and not _COLLECTION.match(value):
        raise ValueError(
            f"must be a Qdrant collection or alias name (letters, digits, _ . -), got {value!r}"
        )
    return value


def _env(value: str | None) -> str | None:
    if value is not None and not _ENV.match(value):
        raise ValueError(f"must be an environment variable name, got {value!r}")
    return value


def _check_model(value: str | None) -> str | None:
    if value is not None:
        try:
            parse_spec(value)
        except SpecError as exc:
            raise ValueError(str(exc)) from None
    return value


class Source(_Strict):
    """A pgvector table."""

    type: Literal["pgvector"] = "pgvector"
    dsn_env: str = "VECSHIFT_DSN"
    """Environment variable holding the connection string. The file never stores it."""
    table: str
    """``table`` or ``schema.table``."""
    vector_column: str | None = None
    text_column: str | None = None
    model: str | None = None
    """The model that made the current vectors, so ``eval`` can embed queries for them."""

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

    @field_validator("model")
    @classmethod
    def _old_model(cls, value: str | None) -> str | None:
        return _check_model(value)


DEFAULT_QDRANT_URL_ENV = "VECSHIFT_QDRANT_URL"
DEFAULT_QDRANT_KEY_ENV = "QDRANT_API_KEY"


class QdrantSource(_Strict):
    """A Qdrant collection, ideally reached through the alias the application searches."""

    type: Literal["qdrant"]
    url_env: str = DEFAULT_QDRANT_URL_ENV
    """Environment variable holding the Qdrant URL."""
    key_env: str = DEFAULT_QDRANT_KEY_ENV
    """Environment variable holding the API key, if the server needs one."""
    collection: str
    """The alias the application searches (or the collection itself, if it has none)."""
    vector: str | None = None
    """The named vector to re-embed, when the collection has several."""
    text_field: str | None = None
    """Payload field with the text, such as ``page_content`` or ``metadata.body``."""
    model: str | None = None
    """The model that made the current vectors, so ``eval`` can embed queries for them."""

    @field_validator("collection")
    @classmethod
    def _name(cls, value: str) -> str:
        _collection(value)
        return value

    @field_validator("url_env", "key_env")
    @classmethod
    def _envs(cls, value: str) -> str:
        _env(value)
        return value

    @field_validator("model")
    @classmethod
    def _old_model(cls, value: str | None) -> str | None:
        return _check_model(value)


STORES = ("pgvector", "qdrant")


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
    """A new column next to the old one, in the same pgvector table."""

    type: Literal["pgvector"] = "pgvector"
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


class QdrantTarget(_Strict):
    """A new Qdrant collection. Cutover points the application's alias at it."""

    type: Literal["qdrant"]
    url_env: str | None = None
    """Defaults to the source's, or VECSHIFT_QDRANT_URL when the source is PostgreSQL."""
    key_env: str | None = None
    collection: str | None = None
    """The new collection. Defaults to the alias (or table) name with ``_v2``."""
    vector: str | None = None
    """Name for the new vector. Defaults to the source's, so queries keep working."""
    alias: str | None = None
    """The alias cutover points at the new collection. Defaults to source.collection when
    that's an alias."""
    index: Literal["hnsw", "none"] = "hnsw"
    metric: Metric | None = None
    """Distance for the new collection. Defaults to the source's, or cosine."""

    @field_validator("collection", "alias")
    @classmethod
    def _names(cls, value: str | None) -> str | None:
        return _collection(value)

    @field_validator("url_env", "key_env")
    @classmethod
    def _envs(cls, value: str | None) -> str | None:
        return _env(value)


class Limits(_Strict):
    budget_usd: float | None = Field(default=None, gt=0)
    """Stop before spending more than this. Plan fails when the estimate exceeds it."""
    tokens_per_minute: int | None = Field(default=None, gt=0)
    """The provider's rate limit, used to estimate duration."""
    requests_per_minute: int | None = Field(default=None, gt=0)


AnySource = Annotated[Source | QdrantSource, Field(discriminator="type")]
AnyTarget = Annotated[Target | QdrantTarget, Field(discriminator="type")]


class JobSpec(_Strict):
    version: Literal[1] = 1
    name: str = "migration"
    source: AnySource
    target: AnyTarget = Target()
    model: str
    """A model spec, as for ``vecshift bench``: ``provider/model[,option=value]``."""
    limits: Limits = Limits()

    @model_validator(mode="before")
    @classmethod
    def _types(cls, data: Any) -> Any:
        """Fill in the store types: pgvector unless said otherwise, and the target's
        defaults to the source's."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("target", "limits"):
            # A section whose settings are all commented out reads as null: use the defaults.
            if key in data and data[key] is None:
                data[key] = {}
        source = data.get("source")
        if isinstance(source, dict) and "type" not in source:
            data["source"] = source = {**source, "type": "pgvector"}
        kind = source.get("type") if isinstance(source, dict) else None
        target = data.get("target", {})
        if isinstance(target, dict) and "type" not in target and kind in STORES:
            data["target"] = {**target, "type": kind}
        return data

    @model_validator(mode="after")
    def _pairs(self) -> JobSpec:
        if self.source.type == "qdrant" and self.target.type != "qdrant":
            raise ValueError("a Qdrant source needs a Qdrant target (target.type: qdrant)")
        return self

    @property
    def pg_source(self) -> Source:
        if not isinstance(self.source, Source):
            raise TypeError("not a pgvector source")
        return self.source

    @property
    def pg_target(self) -> Target:
        if not isinstance(self.target, Target):
            raise TypeError("not a pgvector target")
        return self.target

    @property
    def qdrant_target(self) -> QdrantTarget:
        if not isinstance(self.target, QdrantTarget):
            raise TypeError("not a Qdrant job")
        return self.target

    @property
    def target_url_env(self) -> str:
        target = self.qdrant_target
        if target.url_env:
            return target.url_env
        if isinstance(self.source, QdrantSource):
            return self.source.url_env
        return DEFAULT_QDRANT_URL_ENV

    @property
    def target_key_env(self) -> str:
        target = self.qdrant_target
        if target.key_env:
            return target.key_env
        if isinstance(self.source, QdrantSource):
            return self.source.key_env
        return DEFAULT_QDRANT_KEY_ENV

    @property
    def target_collection(self) -> str:
        target = self.qdrant_target
        if target.collection:
            return target.collection
        if isinstance(self.source, QdrantSource):
            base = target.alias or self.source.collection
        else:
            base = target.alias or self.source.table.split(".")[-1]
        return f"{base}_v2"

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        try:
            parse_spec(value)
        except SpecError as exc:
            raise ValueError(str(exc)) from None
        return value


def format_errors(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        # Union members show up in the location, as in source.qdrant.collection.
        parts = [str(p) for p in item["loc"]]
        parts = [p for i, p in enumerate(parts) if not (i == 1 and p in STORES)]
        where = ".".join(parts) or "file"
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


def _setting(key: str, value: object, example: str, note: str) -> str:
    """One line of a job file: the value quoted, or commented out with an example."""
    if value is None:
        return f"# {key}: {example}".ljust(29) + f"  # {note}"
    return f"{key}: {json.dumps(value)}".ljust(29) + f"  # {note}"


def _old_model(source_model: str | None) -> str:
    return _setting(
        "model",
        source_model,
        "openai/text-embedding-3-small",
        "the model that made the current vectors, for eval",
    )


def _pg_source(
    table: str, vector_column: str | None, text_column: str | None, source_model: str | None
) -> str:
    vector = _setting(
        "vector_column", vector_column, "embedding", "needed only if the table has several"
    )
    text = _setting(
        "text_column", text_column, "content", "detected automatically when it's a common name"
    )
    return f"""source:
  type: pgvector
  # The connection string is read from this environment variable, never from this file.
  dsn_env: VECSHIFT_DSN
  table: {json.dumps(table)}
  {vector}
  {text}
  {_old_model(source_model)}
"""


def _model_and_limits(model: str, budget_usd: float | None) -> str:
    budget = _setting(
        "budget_usd", budget_usd, "50", "plan fails if the estimate is higher; apply stops there"
    )
    return f"""\
# The new model, written as for `vecshift bench`, e.g. openai/text-embedding-3-large,dims=1024
model: {json.dumps(model)}

limits:
  {budget}
  # tokens_per_minute: 1000000   # your provider rate limit, for the duration estimate
  # requests_per_minute: 3000
"""


def template(
    table: str,
    model: str,
    column: str = "embedding_v2",
    *,
    vector_column: str | None = None,
    text_column: str | None = None,
    source_model: str | None = None,
    budget_usd: float | None = None,
) -> str:
    """A commented job file to start from. Values are quoted, so any input stays valid YAML.

    Settings left as ``None`` stay commented out, with an example.
    """
    name = json.dumps(table.split(".")[-1] + "-reembed")
    return f"""# vecshift job: re-embed a pgvector column with a new model, side by side.
# Check it with `vecshift plan`. Nothing is changed until `vecshift apply`.
version: 1
name: {name}

{_pg_source(table, vector_column, text_column, source_model)}
target:
  # New vectors go in this column next to the old ones. Cutover renames the two columns
  # in one transaction, and rollback renames them back.
  column: {json.dumps(column)}
  vector_type: vector            # or halfvec: half the storage, indexable up to 4,000 dims
  index: hnsw                    # hnsw, ivfflat, or none
  # metric: cosine               # defaults to the current index's metric, or cosine

{_model_and_limits(model, budget_usd)}"""


def qdrant_template(
    model: str,
    *,
    collection: str | None = None,
    table: str | None = None,
    vector: str | None = None,
    text_field: str | None = None,
    vector_column: str | None = None,
    source_model: str | None = None,
    alias: str | None = None,
    new_collection: str | None = None,
    budget_usd: float | None = None,
) -> str:
    """A commented job file for a migration into a new Qdrant collection.

    The source is a Qdrant ``collection`` or a pgvector ``table``, one or the other.
    ``text_field`` is the payload field for a Qdrant source and the text column for a
    pgvector one. Settings left as ``None`` stay commented out, with an example.
    """
    if collection is not None and table is not None:
        raise ValueError("pass a collection or a table, not both")
    if collection is not None:
        name = json.dumps(collection + "-reembed")
        what = "a Qdrant collection"
        named = _setting(
            "vector", vector, "dense", "needed only if the collection has several named vectors"
        )
        text = _setting(
            "text_field",
            text_field,
            "page_content",
            "payload field; detected automatically when it's a common name",
        )
        searched = _setting(
            "collection",
            collection,
            "",
            "what your application searches today" if alias else "the alias your app searches",
        )
        source = f"""source:
  type: qdrant
  # The URL and API key are read from these environment variables, never from this file.
  url_env: VECSHIFT_QDRANT_URL
  key_env: QDRANT_API_KEY
  {searched}
  {named}
  {text}
  {_old_model(source_model)}
"""
        connection = ""
        default_name = f"{alias or collection}_v2"
    elif table is not None:
        name = json.dumps(table.split(".")[-1] + "-reembed")
        what = "a pgvector table"
        source = _pg_source(table, vector_column, text_field, source_model)
        connection = (
            "  # The URL and API key are read from these environment variables.\n"
            "  # url_env: VECSHIFT_QDRANT_URL\n"
            "  # key_env: QDRANT_API_KEY\n"
        )
        default_name = f"{alias or table.split('.')[-1]}_v2"
    else:
        raise ValueError("pass a collection or a table")
    target_collection = _setting("collection", new_collection, default_name, "the new collection")
    target_alias = _setting(
        "alias", alias, "documents_live", "the alias cutover points at the new collection"
    )
    return f"""# vecshift job: re-embed {what} into a new Qdrant collection, side by side.
# Check it with `vecshift plan`. Nothing is changed until `vecshift apply`.
version: 1
name: {name}

{source}
target:
  type: qdrant
{connection}  # apply fills a new collection with the source's payloads and new vectors.
  # Cutover points the alias at it in one atomic request, and rollback points it back.
  {target_collection}
  {target_alias}
  index: hnsw                    # hnsw or none
  # metric: cosine               # defaults to the source's distance, or cosine

{_model_and_limits(model, budget_usd)}"""
