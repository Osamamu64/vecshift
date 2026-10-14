"""``cutover``, ``rollback``, and ``cleanup`` for migrations into Qdrant.

The commands in ``cli_cutover`` hand Qdrant jobs to these. Cutover catches up with the
source, then points the application's alias at the new collection in one atomic request;
rollback points it back; cleanup deletes the old collection once you're sure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from vecshift import ui
from vecshift.cli_plan import _fail
from vecshift.cli_style import finding_lines

if TYPE_CHECKING:
    from vecshift.connectors.qdrant import QdrantClient
    from vecshift.connectors.qdrant.switch import QdrantSwitch
    from vecshift.connectors.qdrant.writer import QdrantWriter
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.migrate import JobState

CATCH_UP_TRIES = 3


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


def into_qdrant(job_file: Path) -> bool:
    from vecshift.cli_plan import read_job

    return read_job(job_file).target.type == "qdrant"


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

    from vecshift.cli_plan import _pg_source, pg_settings, qdrant_settings, read_job
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
                raise _fail(
                    f"The source {src.collection} now names the new collection {new}, and "
                    "this folder has no record of the collection before it.",
                    "Run vecshift from the folder where you ran cutover (its record is in "
                    ".vecshift/).",
                )
            field_, _ = qdrant.select_field(reader, origin, src.vector)
            vector = target.vector or field_.name
            text = src.text_field
            if text is None:
                profile = qdrant.inspect(reader, origin, src.vector, None, 50)
                text = (profile.text_field or "").removeprefix("payload.")
            if not text:
                raise _fail("No text payload field found.", "Set source.text_field.")
            source = QdrantSource(reader, origin, text, None)
        else:
            from vecshift.connectors.pgvector.reader import PgSource

            pg = pg_settings(dsn_env or job.pg_source.dsn_env)
            profile, table, _ = _pg_source(job, pg, 50)
            text = profile.text_field or ""
            if not text or len(table.primary_key) != 1:
                raise _fail("The table needs a single-column primary key and a text column.")
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
            on_create=lambda name: _owns(state, name),
        )
        switch = QdrantSwitch(client, writer, alias, previous)
    except qdrant.QdrantError as exc:
        close()
        raise _fail(str(exc), exc.hint) from exc
    except (pgvector.ConnectError, pgvector.TargetSelectionError) as exc:
        close()
        raise _fail(str(exc), getattr(exc, "hint", None)) from exc
    except psycopg.Error as exc:
        close()
        raise _fail(f"The database reported an error: {_first_line(exc)}") from exc
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


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _owns(state: JobState, collection: str) -> None:
    if collection not in state.collections:
        state.collections.append(collection)
        state.save()


def _record(found: Located, event: str) -> None:
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


def _guarded(found: Located, run: Any) -> Any:
    """Turn Qdrant, database, and lock problems into clean messages; always close."""
    import psycopg

    from vecshift.connectors.qdrant import QdrantError
    from vecshift.connectors.qdrant.writer import AlreadyRunning
    from vecshift.embeddings import EmbeddingError

    try:
        return run()
    except AlreadyRunning as exc:
        raise _fail(str(exc), "Wait for it to finish, or stop it first.") from exc
    except (QdrantError, EmbeddingError) as exc:
        raise _fail(str(exc), getattr(exc, "hint", None) or "Nothing was switched.") from exc
    except psycopg.Error as exc:
        raise _fail(
            f"The database reported an error: {_first_line(exc)}", "Nothing was switched."
        ) from exc
    finally:
        found.close()


def _catch_up(found: Located) -> tuple[int, int]:
    """Embed what changed in the source since the last apply, and remove what was deleted.

    Returns (points written, points still missing a vector)."""
    from vecshift.cli_apply import _run
    from vecshift.embeddings import create_embedder
    from vecshift.embeddings.providers import CONCURRENCY

    spec = found.spec
    result = asyncio.run(
        _run(
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
        raise _fail(f"The final catch-up stopped: {result.message}", "Nothing was switched.")
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
    job_file: Path, *, check: bool, yes: bool, allow_missing: bool, output_json: bool
) -> None:
    from vecshift.cli_cutover import _confirm

    found = locate(job_file)
    state = found.state

    def run() -> dict[str, Any]:
        found.writer.acquire()
        try:
            ready = found.switch.check()
            blocked = any(
                f.severity.rank == 3 and f.id != "cutover.pending" for f in ready.findings
            )
            if check or blocked:
                return {"status": "ready" if ready.ok else "blocked", "readiness": ready}
            if not output_json:
                ui.title("cutover", found.job.name)
                finding_lines([f for f in ready.findings if f.severity.rank >= 1])
            where = "your machine" if found.spec.is_local else found.spec.url
            sends = (
                f" First it embeds {ready.pending:,} points that changed since the last "
                f"apply, sending their text to {where}."
                if ready.pending
                else ""
            )
            now = f" (now {found.previous})" if found.previous else ""
            _confirm(
                f"\nCutover points the alias {found.alias}{now} at {found.new}, so searches "
                f"use {found.spec.name} vectors.{sends} Switch your application to "
                f"{found.spec.name} at the same time.",
                yes,
            )
            caught_up = 0
            for _ in range(CATCH_UP_TRIES):
                written, remaining = _catch_up(found)
                caught_up += written
                if not remaining:
                    break
            else:
                raise _fail(
                    f"{remaining:,} points still have no new vector.",
                    "Points keep changing; try again when writes are quieter.",
                )
            if state.failed and not allow_missing:
                raise _fail(
                    f"The provider rejected {len(state.failed):,} points, which would be "
                    "missing from the new collection.",
                    "Fix their text, or pass --allow-missing.",
                )
            found.switch.point(found.new)
            _record(found, "cutover")
            late = (
                _differences(found, found.previous, found.new)
                if found.from_qdrant and found.same_server and found.previous
                else 0
            )
            return {"status": "cut_over", "readiness": ready, "caught_up": caught_up, "late": late}
        finally:
            found.writer.release()

    outcome = _guarded(found, run)
    ready = outcome["readiness"]
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": outcome["status"],
                    "alias": found.alias,
                    "previous": found.previous,
                    "new": found.new,
                    "pending": ready.pending,
                    "caught_up": outcome.get("caught_up", 0),
                    "late": outcome.get("late", 0),
                    "findings": [f.to_dict() for f in ready.findings],
                }
            )
        )
    elif outcome["status"] in {"ready", "blocked"}:
        ui.title("cutover", found.job.name)
        finding_lines(ready.findings)
        if ready.ok:
            ui.verdict("Ready to cut over.")
            if ui.fancy():
                ui.next_step("vecshift cutover")
        else:
            ui.verdict("Not ready to cut over.", "error")
    else:
        ui.verdict("Cut over.")
        ui.detail(f"The alias {found.alias} now points at {found.new}.")
        if outcome.get("caught_up"):
            ui.detail(f"Embedded {outcome['caught_up']:,} late points first.")
        if outcome.get("late"):
            ui.detail(
                f"{outcome['late']:,} points of {found.previous} differ from {found.new}: "
                "edits made in the moments before the switch, or written to the old "
                "collection by name. Check them."
            )
        undo = (
            f"Undo with: vecshift rollback. When you're sure, delete {found.previous} with "
            "vecshift cleanup."
            if found.previous
            else "The source is unchanged."
        )
        ui.next_step(
            f"make sure your application embeds queries and new points with "
            f"{found.spec.name}.\n{undo}"
        )
    if outcome["status"] == "blocked":
        raise typer.Exit(1)


def rollback(job_file: Path, *, yes: bool, output_json: bool) -> None:
    from vecshift.cli_cutover import _confirm

    found = locate(job_file)

    def run() -> int:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise _fail(
                    "There's no cutover to roll back.",
                    f"The alias {found.alias or '(none)'} doesn't point at {found.new}.",
                )
            previous = found.previous
            recorded = [
                h
                for h in history(found.state, found.alias, found.new)
                if h.get("event") == "cutover"
            ]
            if not previous and not recorded:
                raise _fail(
                    "This folder has no record of the cutover, so vecshift doesn't know "
                    "which collection to point the alias back at.",
                    "Run rollback from the folder where you ran cutover (its record is in "
                    ".vecshift/), or move the alias yourself.",
                )
            if not previous:
                raise _fail(
                    "Before cutover the alias didn't exist, so there's nothing to point it "
                    "back at.",
                    "To stop using Qdrant, point your application back at its previous "
                    "store; the source was never changed.",
                )
            if not found.client.exists(previous):
                raise _fail(f"The old collection {previous} no longer exists.")
            _confirm(
                f"Rollback points the alias {found.alias} back at {previous}, so searches use "
                "the old vectors again. Switch your application back to the old model at "
                "the same time.",
                yes,
            )
            found.switch.point(previous)
            _record(found, "rollback")
            if found.from_qdrant and found.same_server:
                return _differences(found, found.new, previous)
            return 0
        finally:
            found.writer.release()

    changed = _guarded(found, run)
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "rolled_back",
                    "alias": found.alias,
                    "live": found.previous,
                    "new": found.new,
                    "changed": changed,
                }
            )
        )
        return
    if ui.fancy():
        ui.title("rollback", found.job.name)
    ui.verdict("Rolled back.")
    ui.detail(
        f"The alias {found.alias} points at {found.previous} again; {found.new} is kept for "
        "another cutover."
    )
    if changed:
        ui.detail(
            f"{changed:,} points were added or edited in {found.new} after cutover and are "
            f"missing or different in {found.previous}. Your application needs to write them "
            "again, embedded with the old model."
        )
    if ui.fancy():
        ui.console.print()


def cleanup(job_file: Path, *, yes: bool, output_json: bool) -> None:
    import sys

    found = locate(job_file)
    previous = found.previous

    def run() -> None:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise _fail(
                    "There's no cutover to clean up after.",
                    "Cleanup deletes the old collection once the alias points at the new one.",
                )
            if not previous:
                raise _fail(
                    "vecshift doesn't know of an old collection to delete: either the alias "
                    "didn't exist before cutover, or this folder has no record of it.",
                    "Delete the old collection yourself when you're sure. A pgvector source "
                    "table is untouched; drop it yourself too.",
                )
            if previous == found.new or previous in dict(found.client.aliases()).values():
                raise _fail(f"{previous} is still in use, so it won't be deleted.")
            if not found.client.exists(previous):
                raise _fail(f"{previous} is already gone.")
            if not yes:
                warning = (
                    f"Cleanup deletes the collection {previous} and every point in it. The "
                    "old vectors are gone for good, and rollback is no longer possible."
                )
                if ui.fancy():
                    ui.warn(warning)
                    typed = ui.text(f"Type {previous} to confirm")
                else:
                    typer.echo(warning, err=True)
                    if not sys.stdin.isatty():
                        raise _fail("Not deleting anything without confirmation.", "Pass --yes.")
                    typed = typer.prompt(f"Type {previous} to confirm", err=True, default="")
                if typed != previous:
                    raise _fail("That didn't match, so nothing was deleted.")
            found.client.delete_collection(previous)
            _record(found, "cleanup")
        finally:
            found.writer.release()

    _guarded(found, run)
    if output_json:
        typer.echo(json.dumps({"status": "cleaned_up", "deleted": previous}))
    else:
        if ui.fancy():
            ui.title("cleanup", found.job.name)
        ui.verdict(f"Deleted {previous}. The migration is complete.")
        if ui.fancy():
            ui.console.print()
