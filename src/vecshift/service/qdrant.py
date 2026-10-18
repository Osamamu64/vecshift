"""Finding a migration into Qdrant as it stands: the new collection, its source, the alias."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import pg_settings, qdrant_settings, read_job
from vecshift.service.plan import pg_source

if TYPE_CHECKING:
    from vecshift.connectors.qdrant import QdrantClient
    from vecshift.connectors.qdrant.switch import QdrantSwitch
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
