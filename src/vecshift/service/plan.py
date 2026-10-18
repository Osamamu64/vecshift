"""Planning a migration: load the job, inspect both sides read-only, and estimate it."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.consent import Consent, Sending, ask
from vecshift.service.errors import ServiceError
from vecshift.service.jobs import load, pg_settings, qdrant_settings, read_job

if TYPE_CHECKING:
    from vecshift.connectors.pgvector import ConnectionSettings
    from vecshift.connectors.pgvector.target import TargetState
    from vecshift.connectors.qdrant import QdrantSettings
    from vecshift.connectors.qdrant.target import QdrantTargetState
    from vecshift.doctor import IndexProfile
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.planning import Plan, ProbeResult, SampleStats

PROBE_DOCUMENTS = 16


@dataclass(slots=True)
class Prepared:
    """Everything known about a job before anything changes."""

    job: JobSpec
    settings: ConnectionSettings
    spec: ModelSpec
    profile: IndexProfile
    target: TargetState
    plan: Plan


@dataclass(slots=True)
class QdrantPrepared:
    """Everything known about a migration into Qdrant before anything changes."""

    job: JobSpec
    spec: ModelSpec
    profile: IndexProfile
    target: QdrantTargetState
    plan: Plan
    qdrant: QdrantSettings
    """Where the new collection goes."""
    text_field: str
    """The payload path (Qdrant source) or column (pgvector source) holding the text."""
    source_qdrant: QdrantSettings | None = None
    source_pg: ConnectionSettings | None = None
    pg: TargetState | None = None
    """The source table, for a pgvector source."""


PlanResult = Prepared | QdrantPrepared


def destination(spec: ModelSpec) -> str:
    """Where a model's text goes: empty when it stays on this machine."""
    return "" if spec.is_local else spec.url or spec.name


def prepare_plan(
    job_file: Path,
    *,
    sample_size: int = 2000,
    probe: bool = False,
    consent: Consent | None = None,
) -> PlanResult:
    """Load the job, inspect its stores read-only, and build the plan.

    With ``probe``, a few sample rows are embedded with the real model to measure it,
    after ``consent`` agrees if they'd leave the machine.
    """
    run = prepare_qdrant if read_job(job_file).target.type == "qdrant" else prepare
    return run(job_file, sample_size=sample_size, probe=probe, consent=consent)


async def _probe(spec: ModelSpec, texts: list[str]) -> ProbeResult:
    from vecshift.embeddings import create_embedder
    from vecshift.planning import ProbeResult

    embedder = create_embedder(spec)
    try:
        start = time.perf_counter()
        vectors = await embedder.embed(texts, "document")
        elapsed = time.perf_counter() - start
    finally:
        await embedder.aclose()
    chars = sum(len(spec.doc_prefix) + len(t) for t in texts)
    return ProbeResult(
        documents=len(texts),
        dimensions=len(vectors[0]),
        tokens_per_char=embedder.tokens / chars if chars else 0.25,
        docs_per_second=len(texts) / elapsed if elapsed > 0 else 0.0,
    )


def _measure(
    spec: ModelSpec, texts: list[str], *, probe: bool, consent: Consent | None
) -> tuple[SampleStats, ProbeResult | None]:
    """Sample statistics, and a probe of the real model if asked for."""
    from vecshift.embeddings import EmbeddingError
    from vecshift.planning import SampleStats

    stats = SampleStats(
        documents=len(texts), average_chars=sum(map(len, texts)) / len(texts) if texts else 0.0
    )
    measured = None
    if probe and texts:
        chosen = texts[:PROBE_DOCUMENTS]
        ask(consent, Sending("probe", ((spec.name, destination(spec)),), len(chosen), "rows"))
        try:
            measured = asyncio.run(_probe(spec, chosen))
        except EmbeddingError as exc:
            raise ServiceError(f"The probe failed: {exc}", code="model.failed") from exc

    return stats, measured


def prepare(
    job_file: Path,
    *,
    sample_size: int = 2000,
    probe: bool = False,
    consent: Consent | None = None,
) -> Prepared:
    """Plan a migration within pgvector."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.documents import sample_documents
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column
    from vecshift.planning import build_plan

    job, settings, spec = load(job_file)
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    try:
        source_column = resolve_source_column(
            conn, job.pg_source.table, job.pg_source.vector_column, job.pg_target.column
        )
        profile = pgvector.inspect(
            conn,
            table=job.pg_source.table,
            column=source_column,
            text_column=job.pg_source.text_column,
            sample_size=sample_size,
        )
        target = inspect_target(conn, job.pg_source.table, source_column, job.pg_target.column)
        texts: list[str] = []
        if profile.text_field:
            rows, _ = sample_documents(
                conn,
                job.pg_source.table,
                source_column,
                profile.text_field,
                min(sample_size, 500),
            )
            texts = [t for _, t in rows]
    except pgvector.TargetSelectionError as exc:
        raise ServiceError(str(exc), code="source.not_found") from exc
    finally:
        conn.rollback()
        conn.close()

    stats, measured = _measure(spec, texts, probe=probe, consent=consent)
    result = build_plan(job, profile, target, stats, measured)
    return Prepared(job, settings, spec, profile, target, result)


def prepare_qdrant(
    job_file: Path,
    *,
    sample_size: int = 2000,
    probe: bool = False,
    consent: Consent | None = None,
) -> QdrantPrepared:
    """Plan a migration into Qdrant."""
    from vecshift.connectors import pgvector, qdrant
    from vecshift.connectors.qdrant.target import inspect_target
    from vecshift.embeddings import parse_spec
    from vecshift.jobs.spec import QdrantSource
    from vecshift.migrate import JobState
    from vecshift.planning.qdrant import build_qdrant_plan

    job = read_job(job_file)
    spec = parse_spec(job.model)
    settings = qdrant_settings(job.target_url_env, job.target_key_env)
    target_spec = job.qdrant_target
    texts: list[str] = []
    pg: TargetState | None = None
    source_qdrant: QdrantSettings | None = None
    source_pg: ConnectionSettings | None = None
    try:
        if isinstance(job.source, QdrantSource):
            src = job.source
            source_qdrant = (
                settings
                if (src.url_env, src.key_env) == (job.target_url_env, job.target_key_env)
                else qdrant_settings(src.url_env, src.key_env)
            )
            with qdrant.QdrantClient(source_qdrant) as client:
                field, info = qdrant.select_field(client, src.collection, src.vector)
                profile = qdrant.inspect(
                    client, src.collection, src.vector, src.text_field, sample_size
                )
                text_field = (profile.text_field or "").removeprefix("payload.")
                if text_field:
                    texts = qdrant_texts(client, field.collection, text_field, sample_size)
            vector = target_spec.vector or field.name
            with qdrant.QdrantClient(settings) as client:
                target = inspect_target(
                    client,
                    job.target_collection,
                    vector,
                    target_spec.alias,
                    (src.collection, field, info),
                )
        else:
            source_pg = pg_settings(job.pg_source.dsn_env)
            profile, pg, texts = pg_source(job, source_pg, sample_size)
            text_field = profile.text_field or ""
            with qdrant.QdrantClient(settings) as client:
                target = inspect_target(
                    client, job.target_collection, target_spec.vector, target_spec.alias
                )
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    except pgvector.TargetSelectionError as exc:
        raise ServiceError(str(exc), code="source.not_found") from exc

    created = JobState.for_job(job_file, job.name).collections
    target = replace(target, owned=target.collection in created)
    stats, measured = _measure(spec, texts, probe=probe, consent=consent)
    result = build_qdrant_plan(job, profile, target, stats, measured, pg)
    return QdrantPrepared(
        job=job,
        spec=spec,
        profile=profile,
        target=target,
        plan=result,
        qdrant=settings,
        text_field=text_field,
        source_qdrant=source_qdrant,
        source_pg=source_pg,
        pg=pg,
    )


def qdrant_texts(client: Any, collection: str, text_field: str, size: int) -> list[str]:
    """Sample texts from a Qdrant collection's payloads."""
    from vecshift.connectors.qdrant.inspect import _get, _points

    points, _ = _points(client, collection, False, min(size, 500), size + 1)
    texts = [_get(p.get("payload") or {}, text_field) for p in points]
    return [t for t in texts if isinstance(t, str) and t.strip()]


def pg_source(
    job: JobSpec, settings: ConnectionSettings, sample_size: int
) -> tuple[IndexProfile, TargetState, list[str]]:
    """Profile a pgvector table that's the source of a migration elsewhere."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.documents import sample_documents
    from vecshift.connectors.pgvector.target import inspect_target

    source = job.pg_source
    conn = pgvector.connect(settings)
    try:
        profile = pgvector.inspect(
            conn,
            table=source.table,
            column=source.vector_column,
            text_column=source.text_column,
            sample_size=sample_size,
        )
        column = profile.target.rsplit(".", 1)[-1]
        state = inspect_target(conn, source.table, column, "")
        texts: list[str] = []
        if profile.text_field:
            rows, _ = sample_documents(
                conn, source.table, column, profile.text_field, min(sample_size, 500)
            )
            texts = [t for _, t in rows]
    finally:
        conn.rollback()
        conn.close()
    return profile, state, texts
