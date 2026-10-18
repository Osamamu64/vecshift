"""Diagnosing a vector index: profile it read-only and run the checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vecshift.service.errors import ServiceError

if TYPE_CHECKING:
    from vecshift.doctor import Report


@dataclass(frozen=True, slots=True)
class Diagnosis:
    report: Report
    connection: str
    """Where, as shown to people: never a password or key."""
    kind: str
    """The kind of connection, such as "PostgreSQL" or "Supabase (pooler)"."""

    def to_dict(self) -> dict[str, Any]:
        return {"connection": self.connection, **self.report.to_dict()}


@dataclass(frozen=True, slots=True)
class Candidate:
    """A vector column or field the caller could pick."""

    name: str
    type: str
    dimensions: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "type": self.type, "dimensions": self.dimensions}


class NeedsChoice(ServiceError):
    """There are several vectors (or none matched): ``candidates`` lists them."""

    def __init__(self, message: str, candidates: Sequence[Any], heading: str) -> None:
        super().__init__(message, code="source.ambiguous")
        self.candidates = [Candidate(c.qualified, c.type, c.dimensions) for c in candidates]
        self.heading = heading


def doctor_pgvector(
    dsn: str,
    *,
    table: str | None = None,
    column: str | None = None,
    text_column: str | None = None,
    sample_size: int = 2000,
    timeout: int = 60,
) -> Diagnosis:
    """Profile a pgvector column and check it. Read-only."""
    import psycopg

    from vecshift.connectors import pgvector
    from vecshift.doctor import run_checks

    try:
        settings = pgvector.prepare(dsn)
        conn = pgvector.connect(settings, statement_timeout_s=timeout)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc

    try:
        profile = pgvector.inspect(
            conn, table=table, column=column, text_column=text_column, sample_size=sample_size
        )
    except pgvector.TargetSelectionError as exc:
        raise NeedsChoice(str(exc), exc.candidates or [], "Vector columns found:") from exc
    except psycopg.Error as exc:
        raise ServiceError(f"Inspection failed: {exc}", code="database.error") from exc
    finally:
        conn.rollback()
        conn.close()
    return Diagnosis(run_checks(profile), settings.display, settings.description)


def doctor_qdrant(
    url: str,
    *,
    collection: str | None = None,
    vector: str | None = None,
    text_field: str | None = None,
    sample_size: int = 2000,
    timeout: int = 60,
    key_env: str | None = None,
) -> Diagnosis:
    """Profile a Qdrant vector and check it. Read-only. The API key comes from ``key_env``."""
    from vecshift.connectors import qdrant
    from vecshift.doctor import run_checks

    try:
        settings = qdrant.prepare(url, key_env) if key_env else qdrant.prepare(url)
        with qdrant.QdrantClient(settings, timeout=float(timeout)) as client:
            profile = qdrant.inspect(client, collection, vector, text_field, sample_size)
    except qdrant.SelectionError as exc:
        raise NeedsChoice(str(exc), exc.candidates, "Vectors found:") from exc
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
    return Diagnosis(run_checks(profile), settings.display, settings.description)
