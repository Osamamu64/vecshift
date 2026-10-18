"""Where a migration stands, and what to run next.

Read-only: it connects like ``plan``, inside a read-only transaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import load, qdrant_settings, read_job

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

STAGES = {
    "not_started": "Not started",
    "filling": "Filling the new column",
    "ready": "Ready to cut over",
    "cut_over": "Cut over",
    "cleaned_up": "Done",
    "conflict": "Needs attention",
}


@dataclass(frozen=True, slots=True)
class Status:
    stage: str
    table: str
    live: str
    new: str
    previous: str
    rows: int | None
    """Rows with text, which should all get a new vector."""
    missing: int | None
    """Of those, rows still without one."""
    index: str | None
    """The ANN index on the new vectors, such as ``hnsw``."""
    trigger: bool
    runs: int
    rows_written: int
    spent_usd: float
    failed: int
    last_event: dict[str, str] | None
    next: str
    source_store: str = "pgvector"
    store: str = "pgvector"
    """Where the new vectors go. For ``qdrant``, ``live`` is the alias, ``new`` the new
    collection, and ``previous`` the source."""

    @property
    def filled(self) -> float | None:
        if self.rows is None or self.missing is None:
            return None
        return 1.0 if self.rows == 0 else (self.rows - self.missing) / self.rows

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "filled": self.filled}


def status(job_file: Path) -> Status:
    """Where the job's migration stands. Read-only."""
    if read_job(job_file).target.type == "qdrant":
        return collect_qdrant(job_file)
    return collect(job_file)


def _exists(conn: psycopg.Connection, relid: int, column: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pg_attribute WHERE attrelid = %s AND attname = %s "
        "AND attnum > 0 AND NOT attisdropped",
        (relid, column),
    ).fetchone()
    return row is not None


def _index(conn: psycopg.Connection, relid: int, column: str) -> str | None:
    row = conn.execute(
        """
        SELECT am.amname FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_am am ON am.oid = ic.relam
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
        WHERE i.indrelid = %s AND a.attname = %s AND i.indisvalid
          AND am.amname IN ('hnsw', 'ivfflat')
        ORDER BY am.amname LIMIT 1
        """,
        (relid, column),
    ).fetchone()
    return str(row[0]) if row else None


def _trigger(conn: psycopg.Connection, relid: int, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM pg_trigger WHERE tgrelid = %s AND tgname = %s AND NOT tgisinternal",
        (relid, name),
    ).fetchone()
    return row is not None


def _counts(
    conn: psycopg.Connection, table: sql.Identifier, text: str, vectors: str
) -> tuple[int, int]:
    from psycopg import sql

    query = sql.SQL(
        "SELECT count(*), count(*) FILTER (WHERE {vectors} IS NULL) FROM {table} "
        "WHERE {text} IS NOT NULL AND btrim({text}::text) <> ''"
    ).format(table=table, text=sql.Identifier(text), vectors=sql.Identifier(vectors))
    row = conn.execute(query).fetchone()
    return (int(row[0]), int(row[1])) if row else (0, 0)


def _next(stage: str, missing: int | None) -> str:
    if stage == "not_started":
        return "Check the plan with vecshift plan, then run vecshift apply."
    if stage == "filling":
        if missing:
            return "Run vecshift apply to embed the remaining rows; it picks up where it left off."
        return "Run vecshift apply to build the index."
    if stage == "ready":
        return "Compare old and new with vecshift eval, then switch with vecshift cutover."
    if stage == "cut_over":
        return (
            "Switch your application to the new model. Undo with vecshift rollback; when "
            "you're sure, free the old column with vecshift cleanup."
        )
    if stage == "cleaned_up":
        return "Nothing left to do: the old vectors are gone and the new ones are live."
    return "Both the new column and the kept old column exist. Read vecshift cutover --check."


def _next_qdrant(stage: str, missing: int | None, alias: str) -> str:
    if stage == "not_started":
        return "Check the plan with vecshift plan, then run vecshift apply."
    if stage == "filling":
        if missing:
            return (
                "Run vecshift apply to embed the remaining points; it picks up where it left off."
            )
        return "Qdrant is still indexing the new collection. Check again in a while."
    if stage == "ready":
        return (
            f"Compare old and new with vecshift eval, then point {alias or 'your alias'} at "
            "the new collection with vecshift cutover. Run apply again first to copy recent "
            "changes."
        )
    if stage == "cut_over":
        return (
            "Switch your application to the new model. Undo with vecshift rollback; when "
            "you're sure, delete the old collection with vecshift cleanup."
        )
    return "Nothing left to do: the new collection is live and the old one is gone."


def collect_qdrant(job_file: Path) -> Status:
    """Where a migration into Qdrant stands. Read-only, but it reads the whole source."""
    import psycopg

    from vecshift.connectors import qdrant
    from vecshift.connectors.qdrant.writer import QdrantWriter
    from vecshift.jobs.spec import QdrantSource as QdrantSourceSpec
    from vecshift.migrate import JobState
    from vecshift.service.qdrant import history, locate

    job = read_job(job_file)
    target = job.qdrant_target
    state = JobState.for_job(job_file, job.name)
    collection = job.target_collection
    if isinstance(job.source, QdrantSourceSpec):
        label = previous = job.source.collection
    else:
        label = previous = job.pg_source.table
    try:
        with qdrant.QdrantClient(qdrant_settings(job.target_url_env, job.target_key_env)) as c:
            aliases = dict(c.aliases())
            exists = c.exists(collection)
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
    alias = target.alias
    if alias is None and isinstance(job.source, QdrantSourceSpec):
        alias = job.source.collection if job.source.collection in aliases else None
    events = history(state, alias, collection)
    last = events[-1] if events else None
    if alias and aliases.get(alias) == collection:
        stage = "cleaned_up" if last and last.get("event") == "cleanup" else "cut_over"
    elif exists:
        stage = "filling"
    else:
        stage = "not_started"

    rows = missing = None
    index = None
    recorded = any(e.get("event") == "cutover" for e in events)
    unknown = stage == "cut_over" and isinstance(job.source, QdrantSourceSpec) and not recorded
    if unknown:
        # Cut over from another folder: the old collection isn't known here.
        previous = "unknown (no cutover record in this folder)"
    if stage in {"filling", "cut_over"} and not unknown:
        found = locate(job_file)
        try:
            previous = found.previous or previous
            if isinstance(job.source, QdrantSourceSpec) and previous != job.source.collection:
                label = f"{job.source.collection} → {previous}"
            counter = QdrantWriter(
                found.client, found.writer.source, found.writer.layout, read_only=True
            )
            rows = counter.total_count()
            missing = counter.pending_count(tuple(state.failed))
            if found.client.collection(collection).get("status") == "green":
                index = "hnsw"
        except qdrant.QdrantError as exc:
            raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
        except psycopg.Error as exc:
            raise ServiceError(
                f"The database reported an error: {first_line(exc)}", code="database.error"
            ) from exc
        finally:
            found.close()
    if stage == "filling" and missing == 0 and (index or target.index == "none"):
        stage = "ready"

    return Status(
        stage=stage,
        table=label,
        live=alias or "",
        new=collection,
        previous=previous,
        rows=rows,
        missing=missing,
        index=index,
        trigger=False,
        runs=state.runs,
        rows_written=state.rows_written,
        spent_usd=state.spent_usd,
        failed=len(state.failed),
        last_event=last,
        next=_next_qdrant(stage, missing, alias or ""),
        source_store=job.source.type,
        store="qdrant",
    )


def collect(job_file: Path) -> Status:
    """Where a migration within pgvector stands."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.inspect import (
        TEXT_COLUMNS,
        TEXT_TYPES,
        _columns,
        _pick,
        select_column,
    )
    from vecshift.connectors.pgvector.target import resolve_source_column
    from vecshift.connectors.pgvector.writer import Layout
    from vecshift.migrate import JobState

    job, settings, _ = load(job_file)
    state = JobState.for_job(job_file, job.name)
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    try:
        live = resolve_source_column(
            conn, job.pg_source.table, job.pg_source.vector_column, job.pg_target.column
        )
        source = select_column(pgvector.find_vector_columns(conn), job.pg_source.table, live)
        columns = _columns(conn, source.relid)
        text = job.pg_source.text_column or _pick(columns, TEXT_COLUMNS, TEXT_TYPES)
        layout = Layout(
            schema=source.schema,
            table=source.table,
            pk="",
            text=text or "",
            target=job.pg_target.column,
            vector_type=job.pg_target.vector_type.value,
            dims=0,
            extension_schema="",
            live=live,
        )
        has_new = _exists(conn, source.relid, layout.target)
        has_previous = _exists(conn, source.relid, layout.previous)
        last = state.history[-1] if state.history else None
        if has_new and has_previous:
            stage = "conflict"
        elif has_previous:
            stage = "cut_over"
        elif has_new:
            stage = "filling"
        elif last and last.get("event") == "cleanup":
            stage = "cleaned_up"
        else:
            stage = "not_started"

        # The new vectors are in the new column until cutover, then under the live name.
        holder = {"filling": layout.target, "conflict": layout.target, "cut_over": live}.get(stage)
        rows = missing = None
        if holder and text:
            rows, missing = _counts(conn, layout.qualified, text, holder)
        index = _index(conn, source.relid, holder) if holder else None
        if stage == "filling" and missing == 0 and (index or job.pg_target.index.value == "none"):
            stage = "ready"
        trigger = _trigger(conn, source.relid, layout.trigger_name())
    except pgvector.TargetSelectionError as exc:
        raise ServiceError(str(exc), code="source.not_found") from exc
    finally:
        conn.rollback()
        conn.close()

    return Status(
        stage=stage,
        table=f"{source.schema}.{source.table}",
        live=live,
        new=layout.target,
        previous=layout.previous,
        rows=rows,
        missing=missing,
        index=index,
        trigger=trigger,
        runs=state.runs,
        rows_written=state.rows_written,
        spent_usd=state.spent_usd,
        failed=len(state.failed),
        last_event=last,
        next=_next(stage, missing),
    )
