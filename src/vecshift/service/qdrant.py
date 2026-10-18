"""Migrations into Qdrant: finding one as it stands (the new collection, its source, the
alias), and cutover, rollback, and cleanup.

Cutover catches up with the source, then points the application's alias at the new
collection in one atomic request; rollback points it back; cleanup deletes the old
collection once you're sure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from vecshift.service.consent import confirmed
from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import pg_settings, qdrant_settings, read_job
from vecshift.service.plan import pg_source
from vecshift.service.switching import (
    CATCH_UP_TRIES,
    CleanupNotice,
    CleanupOutcome,
    ConfirmCleanup,
    ConfirmSwitch,
    CutoverOutcome,
    RollbackOutcome,
    SwitchNotice,
    typed_back,
)

R = TypeVar("R")

if TYPE_CHECKING:
    from vecshift.connectors.qdrant import QdrantClient
    from vecshift.connectors.qdrant.switch import QdrantSwitch
    from vecshift.connectors.qdrant.switch import Readiness as QdrantReadiness
    from vecshift.connectors.qdrant.writer import QdrantWriter
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.migrate import JobState


@dataclass(slots=True)
class Located:
    job: JobSpec
    spec: ModelSpec
    state: JobState
    client: QdrantClient
    writer: QdrantWriter
    switch: QdrantSwitch
    text_field: str
    vector: str | None
    from_qdrant: bool
    same_server: bool = True
    """Whether a Qdrant source is on the target's server, so old and new can be compared."""
    closers: list[Any] = field(default_factory=list)

    @property
    def alias(self) -> str | None:
        return self.switch.alias

    @property
    def new(self) -> str:
        return self.switch.new

    @property
    def previous(self) -> str | None:
        return self.switch.previous

    def close(self) -> None:
        for close in reversed(self.closers):
            close()


def history(state: JobState, alias: str | None, new: str) -> list[dict[str, str]]:
    """This migration's events: other jobs can share a state file (same ``name``)."""
    return [
        h
        for h in state.history
        if h.get("store") == "qdrant" and h.get("alias") == (alias or "") and h.get("new") == new
    ]


def locate(job_file: Path, dsn_env: str | None = None) -> Located:
    """The new collection, the source to catch up from, and the alias, as they are now.

    ``previous`` is the collection the alias named before cutover, on the target server.
    Before cutover that's whatever the alias names now; after it, the alias names the new
    collection, so it comes from this migration's own history, or is unknown.
    """
    import psycopg

    from vecshift.connectors import pgvector, qdrant
    from vecshift.connectors.qdrant.inspect import _fields
    from vecshift.connectors.qdrant.switch import QdrantSwitch
    from vecshift.connectors.qdrant.writer import DISTANCES, Layout, QdrantSource, QdrantWriter
    from vecshift.embeddings import parse_spec
    from vecshift.jobs.spec import QdrantSource as QdrantSourceSpec
    from vecshift.migrate import JobState

    job = read_job(job_file)
    spec = parse_spec(job.model)
    state = JobState.for_job(job_file, job.name)
    target = job.qdrant_target
    settings = qdrant_settings(job.target_url_env, job.target_key_env)
    closers: list[Any] = []

    def close() -> None:
        for fn in reversed(closers):
            fn()

    try:
        client = qdrant.QdrantClient(settings)
        closers.append(client.close)
        aliases = dict(client.aliases())
        new = job.target_collection
        same_server = True
        if isinstance(job.source, QdrantSourceSpec):
            src = job.source
            same_server = (src.url_env, src.key_env) == (job.target_url_env, job.target_key_env)
            alias = target.alias or (
                src.collection if same_server and src.collection in aliases else None
            )
        else:
            alias = target.alias
        if alias and aliases.get(alias) == new:
            cutovers = [h for h in history(state, alias, new) if h.get("event") == "cutover"]
            previous = (cutovers[-1].get("previous") or None) if cutovers else None
        else:
            previous = aliases.get(alias) if alias else None
            if previous is None and same_server and isinstance(job.source, QdrantSourceSpec):
                # A new alias over a collection the application searched by name: that
                # collection is what rollback returns to, and what cleanup deletes.
                previous = aliases.get(job.source.collection, job.source.collection)
        if previous == new:
            previous = None

        source: Any
        if isinstance(job.source, QdrantSourceSpec):
            src = job.source
            reader = client
            if not same_server:
                reader = qdrant.QdrantClient(qdrant_settings(src.url_env, src.key_env))
                closers.append(reader.close)
            # The collection to copy from: on this server, the one the alias named before
            # cutover; on another server, whatever the source names there.
            origin = previous if same_server and alias and aliases.get(alias) == new else None
            if origin is None:
                origin = dict(reader.aliases()).get(src.collection, src.collection)
            if origin == new:
                raise ServiceError(
                    f"The source {src.collection} now names the new collection {new}, and "
                    "this folder has no record of the collection before it.",
                    "Run vecshift from the folder where you ran cutover (its record is in "
                    ".vecshift/).",
                    "state.no_record",
                )
            field_, _ = qdrant.select_field(reader, origin, src.vector)
            vector = target.vector or field_.name
            text = src.text_field
            if text is None:
                profile = qdrant.inspect(reader, origin, src.vector, None, 50)
                text = (profile.text_field or "").removeprefix("payload.")
            if not text:
                raise ServiceError(
                    "No text payload field found.", "Set source.text_field.", "source.no_text"
                )
            source = QdrantSource(reader, origin, text, None)
        else:
            from vecshift.connectors.pgvector.reader import PgSource

            pg = pg_settings(dsn_env or job.pg_source.dsn_env)
            profile, table, _ = pg_source(job, pg, 50)
            text = profile.text_field or ""
            if not text or len(table.primary_key) != 1:
                raise ServiceError(
                    "The table needs a single-column primary key and a text column.",
                    code="source.unsupported",
                )
            conn = pgvector.connect_reader(pg)
            closers.append(conn.close)
            source = PgSource(
                conn, table.source.schema, table.source.table, table.primary_key[0], text
            )
            vector = target.vector

        dims, distance = spec.dimensions or spec.native_dimensions or 0, "Cosine"
        if target.metric:
            distance = DISTANCES[target.metric.value]
        if client.exists(new):
            found = [f for f in _fields(new, client.collection(new)) if f.name == vector]
            if found:
                dims = found[0].dimensions or dims
                distance = found[0].distance or distance
        layout = Layout(new, vector, dims, distance)
        writer = QdrantWriter(
            client,
            source,
            layout,
            state.path.with_suffix(".lock"),
            owned=state.collections,
            on_create=lambda name: owns(state, name),
        )
        switch = QdrantSwitch(client, writer, alias, previous)
    except qdrant.QdrantError as exc:
        close()
        raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
    except (pgvector.ConnectError, pgvector.TargetSelectionError) as exc:
        close()
        raise ServiceError(str(exc), getattr(exc, "hint", None), "connect.failed") from exc
    except psycopg.Error as exc:
        close()
        raise ServiceError(
            f"The database reported an error: {first_line(exc)}", code="database.error"
        ) from exc
    except BaseException:
        close()
        raise
    return Located(
        job=job,
        spec=spec,
        state=state,
        client=client,
        writer=writer,
        switch=switch,
        text_field=text,
        vector=vector,
        from_qdrant=isinstance(job.source, QdrantSourceSpec),
        same_server=same_server,
        closers=closers,
    )


def owns(state: JobState, collection: str) -> None:
    """Record a collection this job created, at once, so a crash can't lose the record."""
    if collection not in state.collections:
        state.collections.append(collection)
        state.save()


def record(found: Located, event: str) -> None:
    """Add a cutover, rollback, or cleanup to this migration's history."""
    found.state.history.append(
        {
            "event": event,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "store": "qdrant",
            "alias": found.alias or "",
            "previous": found.previous or "",
            "new": found.new,
        }
    )
    found.state.save()


def _guarded(found: Located, run: Callable[[], R]) -> R:
    """Turn Qdrant, database, and lock problems into clean messages; always close."""
    import psycopg

    from vecshift.connectors.qdrant import QdrantError
    from vecshift.connectors.qdrant.writer import AlreadyRunning
    from vecshift.embeddings import EmbeddingError

    try:
        return run()
    except AlreadyRunning as exc:
        raise ServiceError(
            str(exc), "Wait for it to finish, or stop it first.", "job.running"
        ) from exc
    except QdrantError as exc:
        raise ServiceError(str(exc), exc.hint or "Nothing was switched.", "qdrant.error") from exc
    except EmbeddingError as exc:
        raise ServiceError(str(exc), "Nothing was switched.", "model.failed") from exc
    except psycopg.Error as exc:
        raise ServiceError(
            f"The database reported an error: {first_line(exc)}",
            "Nothing was switched.",
            "database.error",
        ) from exc
    finally:
        found.close()


def _catch_up(found: Located) -> tuple[int, int]:
    """Embed what changed in the source since the last apply, and remove what was deleted.

    Returns (points written, points still missing a vector)."""
    from vecshift.embeddings import create_embedder
    from vecshift.embeddings.providers import CONCURRENCY
    from vecshift.service.apply import run_engine

    spec = found.spec
    result = asyncio.run(
        run_engine(
            found.writer,
            create_embedder(spec),
            found.state,
            dims=found.writer.layout.dims,
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
    found.writer.remove_deleted()
    # Count again after deleting: the switch must never leave searches short of points.
    return result.rows_written, found.writer.pending_count(tuple(found.state.failed))


def _differences(found: Located, source: str, target: str) -> int:
    """Points of ``source`` missing from ``target`` or with different text there. Read-only."""
    from vecshift.connectors.qdrant.writer import Layout, QdrantSource, QdrantWriter

    lay = found.writer.layout
    reader = QdrantWriter(
        found.client,
        QdrantSource(found.client, source, found.text_field, None),
        Layout(target, lay.vector, lay.dims, lay.distance),
        read_only=True,
    )
    return reader.pending_count()


def cutover(
    job_file: Path,
    *,
    check: bool = False,
    allow_missing: bool = False,
    confirm: ConfirmSwitch | None = None,
) -> CutoverOutcome:
    """Catch up with the source, then point the alias at the new collection atomically."""
    found = locate(job_file)
    state = found.state

    def outcome(
        status: str, ready: QdrantReadiness, caught_up: int = 0, late: int = 0
    ) -> CutoverOutcome:
        return CutoverOutcome(
            status=status,
            store="qdrant",
            job=found.job.name,
            readiness=ready,
            live=found.alias,
            previous=found.previous,
            new=found.new,
            table="",
            model=found.spec.name,
            caught_up=caught_up,
            late=late,
        )

    def run() -> CutoverOutcome:
        found.writer.acquire()
        try:
            ready = found.switch.check()
            blocked = any(
                f.severity.rank == 3 and f.id != "cutover.pending" for f in ready.findings
            )
            if check or blocked:
                return outcome("ready" if ready.ok else "blocked", ready)
            where = "your machine" if found.spec.is_local else found.spec.url
            sends = (
                f" First it embeds {ready.pending:,} points that changed since the last "
                f"apply, sending their text to {where}."
                if ready.pending
                else ""
            )
            now = f" (now {found.previous})" if found.previous else ""
            message = (
                f"Cutover points the alias {found.alias}{now} at {found.new}, so searches "
                f"use {found.spec.name} vectors.{sends} Switch your application to "
                f"{found.spec.name} at the same time."
            )
            notice = SwitchNotice("cutover", found.job.name, message, ready)
            confirmed(confirm, notice, "changing the database")
            caught_up = 0
            for _ in range(CATCH_UP_TRIES):
                written, remaining = _catch_up(found)
                caught_up += written
                if not remaining:
                    break
            else:
                raise ServiceError(
                    f"{remaining:,} points still have no new vector.",
                    "Points keep changing; try again when writes are quieter.",
                    "cutover.pending",
                )
            if state.failed and not allow_missing:
                raise ServiceError(
                    f"The provider rejected {len(state.failed):,} points, which would be "
                    "missing from the new collection.",
                    "Fix their text, or pass --allow-missing.",
                    "cutover.rejected",
                )
            found.switch.point(found.new)
            record(found, "cutover")
            late = (
                _differences(found, found.previous, found.new)
                if found.from_qdrant and found.same_server and found.previous
                else 0
            )
            return outcome("cut_over", ready, caught_up, late)
        finally:
            found.writer.release()

    return _guarded(found, run)


def rollback(job_file: Path, *, confirm: ConfirmSwitch | None = None) -> RollbackOutcome:
    """Point the alias back at the collection it named before cutover."""
    found = locate(job_file)

    def run() -> int:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise ServiceError(
                    "There's no cutover to roll back.",
                    f"The alias {found.alias or '(none)'} doesn't point at {found.new}.",
                    "rollback.no_cutover",
                )
            previous = found.previous
            recorded = [
                h
                for h in history(found.state, found.alias, found.new)
                if h.get("event") == "cutover"
            ]
            if not previous and not recorded:
                raise ServiceError(
                    "This folder has no record of the cutover, so vecshift doesn't know "
                    "which collection to point the alias back at.",
                    "Run rollback from the folder where you ran cutover (its record is in "
                    ".vecshift/), or move the alias yourself.",
                    "state.no_record",
                )
            if not previous:
                raise ServiceError(
                    "Before cutover the alias didn't exist, so there's nothing to point it "
                    "back at.",
                    "To stop using Qdrant, point your application back at its previous "
                    "store; the source was never changed.",
                    "rollback.no_previous",
                )
            if not found.client.exists(previous):
                raise ServiceError(
                    f"The old collection {previous} no longer exists.", code="rollback.gone"
                )
            message = (
                f"Rollback points the alias {found.alias} back at {previous}, so searches use "
                "the old vectors again. Switch your application back to the old model at "
                "the same time."
            )
            notice = SwitchNotice("rollback", found.job.name, message)
            confirmed(confirm, notice, "changing the database")
            found.switch.point(previous)
            record(found, "rollback")
            if found.from_qdrant and found.same_server:
                return _differences(found, found.new, previous)
            return 0
        finally:
            found.writer.release()

    changed = _guarded(found, run)
    return RollbackOutcome(
        "qdrant", found.job.name, found.alias, found.previous, found.new, "", changed
    )


def cleanup(job_file: Path, *, confirm: ConfirmCleanup | None = None) -> CleanupOutcome:
    """Delete the collection the alias named before cutover."""
    found = locate(job_file)
    previous = found.previous

    def run() -> str:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise ServiceError(
                    "There's no cutover to clean up after.",
                    "Cleanup deletes the old collection once the alias points at the new one.",
                    "cleanup.no_cutover",
                )
            if not previous:
                raise ServiceError(
                    "vecshift doesn't know of an old collection to delete: either the alias "
                    "didn't exist before cutover, or this folder has no record of it.",
                    "Delete the old collection yourself when you're sure. A pgvector source "
                    "table is untouched; drop it yourself too.",
                    "cleanup.no_previous",
                )
            if previous == found.new or previous in dict(found.client.aliases()).values():
                raise ServiceError(
                    f"{previous} is still in use, so it won't be deleted.", code="cleanup.in_use"
                )
            if not found.client.exists(previous):
                raise ServiceError(f"{previous} is already gone.", code="cleanup.gone")
            message = (
                f"Cleanup deletes the collection {previous} and every point in it. The "
                "old vectors are gone for good, and rollback is no longer possible."
            )
            typed_back(
                confirm,
                CleanupNotice(found.job.name, previous, message, "qdrant"),
                "That didn't match, so nothing was deleted.",
            )
            found.client.delete_collection(previous)
            record(found, "cleanup")
            return previous
        finally:
            found.writer.release()

    return CleanupOutcome("qdrant", found.job.name, _guarded(found, run))
