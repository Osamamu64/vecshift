"""Switching searches to the new vectors (cutover), back (rollback), and freeing the old
ones for good (cleanup).

Each asks the caller's ``confirm`` callback before it changes anything, while it holds the
job's lock, so what was checked is still true when the switch happens.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.consent import confirmed
from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import into_qdrant, load

if TYPE_CHECKING:
    import psycopg

    from vecshift.connectors.pgvector.switch import PgSwitch
    from vecshift.connectors.pgvector.switch import Readiness as PgReadiness
    from vecshift.connectors.pgvector.writer import Layout
    from vecshift.connectors.qdrant.switch import Readiness as QdrantReadiness
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.migrate import JobState

    Readiness = PgReadiness | QdrantReadiness

CATCH_UP_TRIES = 3


@dataclass(frozen=True, slots=True)
class SwitchNotice:
    """A cutover or rollback about to happen, for the caller to confirm."""

    action: str
    """``cutover`` or ``rollback``."""
    job: str
    message: str
    """What changes, for people: which names move, and what to switch in the application."""
    readiness: Readiness | None = None
    """Cutover's checks."""


@dataclass(frozen=True, slots=True)
class CleanupNotice:
    """Old vectors about to be deleted. The caller confirms by giving ``name`` back."""

    job: str
    name: str
    """The column or collection to delete."""
    message: str
    store: str = "pgvector"


@dataclass(frozen=True, slots=True)
class CutoverOutcome:
    status: str
    """``ready`` or ``blocked`` (a check only, or blocked by an error), or ``cut_over``."""
    store: str
    job: str
    readiness: Readiness
    live: str | None
    """The name searches use: the column, or the alias for Qdrant."""
    previous: str | None
    """Where the old vectors are kept."""
    new: str
    table: str
    """For pgvector, the table."""
    model: str
    """The new model."""
    caught_up: int = 0
    """Rows or points embedded first, because they changed since the last apply."""
    late: int = 0
    """Qdrant: points of the old collection that differ from the new one after the switch."""

    def to_dict(self) -> dict[str, Any]:
        r = self.readiness
        names = (
            {"alias": self.live, "previous": self.previous, "new": self.new}
            if self.store == "qdrant"
            else {"live": self.live, "previous": self.previous, "new": self.new}
        )
        late = {"late": self.late} if self.store == "qdrant" else {}
        return {
            "status": self.status,
            **names,
            "pending": r.pending,
            "caught_up": self.caught_up,
            **late,
            "findings": [f.to_dict() for f in r.findings],
        }


@dataclass(frozen=True, slots=True)
class RollbackOutcome:
    store: str
    job: str
    live: str | None
    """The column, or the alias for Qdrant."""
    previous: str | None
    """Qdrant: the collection the alias points at again."""
    new: str
    table: str
    missing: int
    """Rows or points added or edited since cutover, without an old-model vector."""

    def to_dict(self) -> dict[str, Any]:
        if self.store == "qdrant":
            return {
                "status": "rolled_back",
                "alias": self.live,
                "live": self.previous,
                "new": self.new,
                "changed": self.missing,
            }
        return {
            "status": "rolled_back",
            "live": self.live,
            "new": self.new,
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class CleanupOutcome:
    store: str
    job: str
    deleted: str

    def to_dict(self) -> dict[str, Any]:
        key = "deleted" if self.store == "qdrant" else "dropped"
        return {"status": "cleaned_up", key: self.deleted}


ConfirmSwitch = Callable[[SwitchNotice], bool]
ConfirmCleanup = Callable[[CleanupNotice], str | None]
"""Returns the name typed back; cleanup goes ahead only if it matches."""


def cutover(
    job_file: Path,
    *,
    check: bool = False,
    allow_missing: bool = False,
    confirm: ConfirmSwitch | None = None,
) -> CutoverOutcome:
    """Switch searches to the new vectors: rename the columns, or move the alias.

    Rows changed since the last apply are embedded first. With ``check``, only report
    whether it's safe. ``allow_missing`` switches even though the provider rejected some
    rows, which then have no vector.
    """
    if into_qdrant(job_file):
        from vecshift.service import qdrant

        return qdrant.cutover(job_file, check=check, allow_missing=allow_missing, confirm=confirm)
    return _pg_cutover(job_file, check=check, allow_missing=allow_missing, confirm=confirm)


def rollback(job_file: Path, *, confirm: ConfirmSwitch | None = None) -> RollbackOutcome:
    """Undo a cutover: the old vectors get their column name, or the alias, back."""
    if into_qdrant(job_file):
        from vecshift.service import qdrant

        return qdrant.rollback(job_file, confirm=confirm)
    return _pg_rollback(job_file, confirm=confirm)


def cleanup(job_file: Path, *, confirm: ConfirmCleanup | None = None) -> CleanupOutcome:
    """After cutover, delete the old vectors for good. Rollback is impossible afterwards."""
    if into_qdrant(job_file):
        from vecshift.service import qdrant

        return qdrant.cleanup(job_file, confirm=confirm)
    return _pg_cleanup(job_file, confirm=confirm)


def typed_back(confirm: ConfirmCleanup | None, notice: CleanupNotice, mismatch: str) -> None:
    """Go ahead only if the caller gives the name of what's deleted."""
    if confirm is None:
        raise ServiceError("Not deleting anything without confirmation.", code="confirm.required")
    if confirm(notice) != notice.name:
        raise ServiceError(mismatch, code="confirm.mismatch")


# --- pgvector


@dataclass(slots=True)
class _Located:
    job: JobSpec
    spec: ModelSpec
    conn: psycopg.Connection
    layout: Layout
    state: JobState


def _locate(job_file: Path) -> _Located:
    """Find the live, new, and previous columns, over a connection that can write."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.inspect import (
        TEXT_COLUMNS,
        TEXT_TYPES,
        _columns,
        _pick,
    )
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column
    from vecshift.connectors.pgvector.writer import Layout
    from vecshift.migrate import JobState

    job, settings, spec = load(job_file, command="vecshift cutover, rollback, and cleanup")
    try:
        conn = pgvector.connect_writer(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    try:
        live = resolve_source_column(
            conn, job.pg_source.table, job.pg_source.vector_column, job.pg_target.column
        )
        target = inspect_target(conn, job.pg_source.table, live, job.pg_target.column)
        text = job.pg_source.text_column or _pick(
            _columns(conn, target.source.relid), TEXT_COLUMNS, TEXT_TYPES
        )
    except pgvector.TargetSelectionError as exc:
        conn.close()
        raise ServiceError(str(exc), code="source.not_found") from exc
    if len(target.primary_key) != 1 or text is None:
        conn.close()
        raise ServiceError(
            "Cutover needs a single-column primary key and a text column.",
            code="source.unsupported",
        )
    dims = target.column_dimensions or target.source.dimensions or spec.dimensions or 0
    layout = Layout(
        schema=target.source.schema,
        table=target.source.table,
        pk=target.primary_key[0],
        text=text,
        target=job.pg_target.column,
        vector_type=job.pg_target.vector_type.value,
        dims=dims,
        extension_schema=target.extension_schema,
        live=live,
    )
    return _Located(job, spec, conn, layout, JobState.for_job(job_file, job.name))


def _record(state: JobState, event: str, layout: Layout) -> None:
    state.history.append(
        {
            "event": event,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "table": f"{layout.schema}.{layout.table}",
            "live": layout.live,
            "previous": layout.previous,
            "new": layout.target,
        }
    )
    state.save()


def _catch_up(found: _Located, switch: PgSwitch) -> int:
    """Embed rows that arrived or changed since the last apply. Returns rows written."""
    from vecshift.connectors.pgvector.writer import PgWriter
    from vecshift.embeddings import create_embedder
    from vecshift.embeddings.providers import CONCURRENCY
    from vecshift.service.apply import run_engine

    spec = found.spec
    result = asyncio.run(
        run_engine(
            PgWriter(found.conn, found.layout),
            create_embedder(spec),
            found.state,
            dims=found.layout.dims,
            price_per_million=spec.price,
            budget_usd=found.job.limits.budget_usd,
            chunk_rows=spec.batch_size * CONCURRENCY,
            index=None,
        )
    )
    if result.status not in {"complete", "stopped"}:
        raise ServiceError(
            f"The final catch-up stopped: {result.message}",
            "Nothing was switched.",
            "cutover.catch_up_failed",
        )
    return result.rows_written


def _filler(found: _Located) -> Any:
    """Embeds the few rows that arrive between the last catch-up and the switch."""
    from vecshift.embeddings import EmbeddingError, create_embedder
    from vecshift.migrate.engine import MAX_CHARS

    spec, state, dims = found.spec, found.state, found.layout.dims

    def fill(rows: list[Any]) -> list[tuple[Any, list[float]]]:
        async def embed() -> tuple[list[list[float]], int]:
            embedder = create_embedder(spec)
            try:
                vectors = await embedder.embed([r.text[:MAX_CHARS] for r in rows], "document")
                return vectors, embedder.tokens
            finally:
                await embedder.aclose()

        vectors, tokens = asyncio.run(embed())
        if any(len(v) != dims for v in vectors):
            raise EmbeddingError(f"{spec.name} returned vectors of the wrong size.")
        state.tokens += tokens
        if spec.price is not None:
            state.spent_usd += tokens * spec.price / 1_000_000
        return list(zip(rows, vectors, strict=True))

    return fill


def _guarded(run: Any) -> Any:
    """Turn database and lock problems into clean messages."""
    import psycopg

    from vecshift.connectors.pgvector.writer import AlreadyRunning, Busy
    from vecshift.embeddings import EmbeddingError

    try:
        return run()
    except AlreadyRunning as exc:
        raise ServiceError(
            str(exc), "Wait for it to finish, or stop it first.", "job.running"
        ) from exc
    except Busy as exc:
        raise ServiceError(str(exc), "Nothing was switched; try again.", "database.busy") from exc
    except EmbeddingError as exc:
        raise ServiceError(str(exc), "Nothing was switched; try again.", "model.failed") from exc
    except psycopg.Error as exc:
        raise ServiceError(
            f"The database reported an error: {first_line(exc)}",
            "Nothing was switched.",
            "database.error",
        ) from exc


def _pg_cutover(
    job_file: Path, *, check: bool, allow_missing: bool, confirm: ConfirmSwitch | None
) -> CutoverOutcome:
    from vecshift.connectors.pgvector.switch import PgSwitch, StillPending
    from vecshift.connectors.pgvector.writer import PgWriter

    found = _locate(job_file)
    lay, state = found.layout, found.state
    writer = PgWriter(found.conn, lay)
    switch = PgSwitch(writer)
    index = found.job.pg_target.index.value
    method = None if index == "none" else index

    def outcome(status: str, ready: PgReadiness, caught_up: int = 0) -> CutoverOutcome:
        return CutoverOutcome(
            status=status,
            store="pgvector",
            job=found.job.name,
            readiness=ready,
            live=lay.live,
            previous=lay.previous,
            new=lay.target,
            table=lay.table,
            model=found.spec.name,
            caught_up=caught_up,
        )

    def run() -> CutoverOutcome:
        writer.acquire()
        try:
            ready = switch.check(method)
            blocked = any(
                f.severity.rank == 3 and f.id != "cutover.pending" for f in ready.findings
            )
            if check or blocked:
                return outcome("ready" if ready.ok else "blocked", ready)
            where = "your machine" if found.spec.is_local else found.spec.url
            sends = (
                f" First it embeds {ready.pending:,} rows that changed since the last apply, "
                f"sending their text to {where}."
                if ready.pending
                else ""
            )
            message = (
                f"Cutover renames {lay.live} to {lay.previous} and {lay.target} to "
                f"{lay.live} on {lay.schema}.{lay.table}, so searches use {found.spec.name} "
                f"vectors.{sends} Switch your application to {found.spec.name} at the same time."
            )
            notice = SwitchNotice("cutover", found.job.name, message, ready)
            confirmed(confirm, notice, "changing the database")
            allowed = len(state.failed) if allow_missing else 0
            caught_up = _catch_up(found, switch) if ready.pending else 0
            switch.prepare()
            try:
                for attempt in range(1, CATCH_UP_TRIES + 1):
                    try:
                        caught_up += switch.cutover(
                            allowed, fill=_filler(found), exclude=tuple(state.failed)
                        )
                        break
                    except StillPending as exc:
                        if attempt == CATCH_UP_TRIES:
                            hint = (
                                "The provider rejected some rows: fix their text, or pass "
                                "--allow-missing."
                                if state.failed
                                else "Rows keep changing; try again when writes are quieter."
                            )
                            raise ServiceError(
                                f"{exc.rows:,} rows still have no new vector.",
                                hint,
                                "cutover.pending",
                            ) from exc
                        caught_up += _catch_up(found, switch)
            finally:
                switch.finish()
            _record(state, "cutover", lay)
            return outcome("cut_over", ready, caught_up)
        finally:
            writer.release()

    try:
        result: CutoverOutcome = _guarded(run)
        return result
    finally:
        found.conn.close()


def _pg_rollback(job_file: Path, *, confirm: ConfirmSwitch | None) -> RollbackOutcome:
    from vecshift.connectors.pgvector.switch import PgSwitch
    from vecshift.connectors.pgvector.writer import PgWriter

    found = _locate(job_file)
    lay, state = found.layout, found.state
    writer = PgWriter(found.conn, lay)
    switch = PgSwitch(writer)

    def run() -> int:
        writer.acquire()
        try:
            stage = switch.stage()
            if stage != "cut_over":
                raise ServiceError(
                    "There's no cutover to roll back.",
                    f"Rollback needs {lay.previous}, holding the old vectors, and no "
                    f"{lay.target} column.",
                    "rollback.no_cutover",
                )
            bound = switch.dependents(lay.live) + switch.dependents(lay.previous)
            if bound:
                raise ServiceError(
                    f"Views or functions are bound to the vector columns: {', '.join(bound)}.",
                    "Drop them before rollback and recreate them after it.",
                    "rollback.dependents",
                )
            message = (
                f"Rollback renames {lay.live} to {lay.target} and {lay.previous} to {lay.live} "
                f"on {lay.schema}.{lay.table}, so searches use the old vectors again. Switch "
                "your application back to the old model at the same time."
            )
            confirmed(
                confirm, SwitchNotice("rollback", found.job.name, message), "changing the database"
            )
            missing = switch.rollback()
            _record(state, "rollback", lay)
            return missing
        finally:
            writer.release()

    try:
        missing: int = _guarded(run)
    finally:
        found.conn.close()
    return RollbackOutcome(
        "pgvector", found.job.name, lay.live, None, lay.target, lay.table, missing
    )


def _pg_cleanup(job_file: Path, *, confirm: ConfirmCleanup | None) -> CleanupOutcome:
    from vecshift.connectors.pgvector.switch import PgSwitch
    from vecshift.connectors.pgvector.writer import PgWriter

    found = _locate(job_file)
    lay, state = found.layout, found.state
    writer = PgWriter(found.conn, lay)
    switch = PgSwitch(writer)

    def run() -> None:
        writer.acquire()
        try:
            if switch.stage() != "cut_over":
                raise ServiceError(
                    "There's no cutover to clean up after.",
                    f"Cleanup drops {lay.previous}, which cutover creates.",
                    "cleanup.no_cutover",
                )
            bound = switch.dependents(lay.previous)
            if bound:
                raise ServiceError(
                    f"Views or functions use {lay.previous}: {', '.join(bound)}.",
                    "Drop or change them first.",
                    "cleanup.dependents",
                )
            message = (
                f"Cleanup drops {lay.schema}.{lay.table}.{lay.previous} and its index. The "
                "old vectors are gone for good, and rollback is no longer possible."
            )
            typed_back(
                confirm,
                CleanupNotice(found.job.name, lay.previous, message, "pgvector"),
                "That didn't match, so nothing was dropped.",
            )
            switch.cleanup()
            _record(state, "cleanup", lay)
        finally:
            writer.release()

    try:
        _guarded(run)
    finally:
        found.conn.close()
    return CleanupOutcome("pgvector", found.job.name, lay.previous)
