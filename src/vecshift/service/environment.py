"""Which environment variables vecshift reads, and whether they're set. Never their values."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vecshift.jobs import JobSpec

CONNECTION = "connection"
URL = "url"
KEY = "key"

DEFAULTS = {
    "VECSHIFT_DSN": CONNECTION,
    "DATABASE_URL": CONNECTION,
    "VECSHIFT_QDRANT_URL": URL,
    "QDRANT_API_KEY": KEY,
}


@dataclass(slots=True)
class Variable:
    name: str
    kind: str
    """``connection`` (a PostgreSQL connection string), ``url`` (Qdrant), or ``key``."""
    set: bool = False
    display: str | None = None
    """For a connection string or URL: where it points, without a password. Never a key."""
    problem: str | None = None
    """Why a value that's set can't be used, such as an invalid connection string."""
    used_by: list[str] = field(default_factory=list)
    """The jobs that read it."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "set": self.set,
            "display": self.display,
            "problem": self.problem,
            "used_by": self.used_by,
        }


def referenced(job: JobSpec) -> dict[str, str]:
    """The variables a job reads, and their kinds."""
    from vecshift.embeddings import SpecError, parse_spec
    from vecshift.jobs.spec import QdrantSource

    names: dict[str, str] = {}
    if isinstance(job.source, QdrantSource):
        names[job.source.url_env] = URL
        names[job.source.key_env] = KEY
    else:
        names[job.pg_source.dsn_env] = CONNECTION
    if job.target.type == "qdrant":
        names[job.target_url_env] = URL
        names[job.target_key_env] = KEY
    for model in (job.model, job.source.model):
        if not model:
            continue
        try:
            key_env = parse_spec(model).key_env
        except SpecError:
            continue
        if key_env:
            names[key_env] = KEY
    return names


def variables(
    jobs: Mapping[str, JobSpec], environ: Mapping[str, str] | None = None
) -> list[Variable]:
    """Every variable vecshift may read here: the defaults, model providers' keys, and the
    names ``jobs`` refer to. Values are only looked at to tell where they point."""
    from vecshift.embeddings.spec import PRESETS

    env = os.environ if environ is None else environ
    found = {name: Variable(name, kind) for name, kind in DEFAULTS.items()}
    for preset in PRESETS.values():
        if preset.key_env:
            found.setdefault(preset.key_env, Variable(preset.key_env, KEY))
    for job_name, job in jobs.items():
        for name, kind in referenced(job).items():
            found.setdefault(name, Variable(name, kind)).used_by.append(job_name)
    for variable in found.values():
        value = env.get(variable.name)
        variable.set = bool(value)
        if value:
            variable.display, variable.problem = _where(variable.kind, value)
    return sorted(found.values(), key=lambda v: v.name)


def _where(kind: str, value: str) -> tuple[str | None, str | None]:
    """A display form of a connection string or URL, or why it can't be used."""
    if kind == CONNECTION:
        from vecshift.connectors import pgvector

        try:
            return pgvector.prepare(value).display, None
        except pgvector.ConnectError as exc:
            return None, str(exc)
    if kind == URL:
        from vecshift.connectors import qdrant

        try:
            # A placeholder key variable: only the URL is checked here, not the key rule.
            return qdrant.prepare(value, "", key=None).display, None
        except qdrant.QdrantError as exc:
            return None, str(exc)
    return None, None
