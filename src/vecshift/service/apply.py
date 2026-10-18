"""Running (or resuming) a migration: add the column or collection, embed, and index."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.consent import confirmed
from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import read_job
from vecshift.service.plan import Prepared, QdrantPrepared, prepare, prepare_qdrant
from vecshift.service.qdrant import owns

if TYPE_CHECKING:
    from vecshift.jobs import JobSpec
    from vecshift.migrate import ApplyResult, Event, JobState
    from vecshift.planning import Plan

SAVED = "Every finished batch is saved; run apply again to continue."


@dataclass(frozen=True, slots=True)
class ApplyOptions:
    no_index: bool = False
    """Backfill only; build the index on a later run."""
    until: int | None = None
    """Stop once this percentage of rows has a new vector."""


@dataclass(frozen=True, slots=True)
class ApplyNotice:
    """What apply is about to change, and where row text goes, for the caller to confirm."""

    plan: Plan
    job: JobSpec
    state: JobState
    """Earlier runs: rows written and money spent so far."""
    into_qdrant: bool
    changes: str
    """Such as "This changes public.docs"."""
    destination: str
    """Where row text goes: "your machine", or the model's URL."""

    @property
    def message(self) -> str:
        return f"{self.changes} and sends row text to {self.destination}."


@dataclass(frozen=True, slots=True)
class ApplyOutcome:
    result: ApplyResult
    into_qdrant: bool
    removed: int = 0
    """Points deleted from the source, removed from the new collection."""
    copied: int = 0
    """Payload-only edits copied to the new collection."""

    def to_dict(self) -> dict[str, Any]:
        extra = {"deleted": self.removed, "payloads_copied": self.copied}
        return {**asdict(self.result), **(extra if self.into_qdrant else {})}


class PlanBlocked(ServiceError):
    """The plan has errors, so apply changed nothing. ``plan`` has the findings."""

    def __init__(self, plan: Plan) -> None:
        super().__init__("The plan has errors, so nothing was changed.", code="plan.errors")
        self.plan = plan


def run_apply(
    job_file: Path,
    options: ApplyOptions | None = None,
    *,
    confirm: Callable[[ApplyNotice], bool] | None = None,
    on_start: Callable[[], None] | None = None,
    on_event: Callable[[Event], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ApplyOutcome:
    """Plan the job, ask ``confirm``, then fill the new vectors and build the index.

    ``on_start`` is called once everything is connected, just before the first change.
    ``should_stop`` is polled between batches: a stop is clean, and the next run resumes.
    """
    import psycopg

    from vecshift.connectors import qdrant
    from vecshift.connectors.pgvector.writer import AlreadyRunning, Busy
    from vecshift.connectors.qdrant.writer import AlreadyRunning as QdrantRunning
    from vecshift.embeddings import EmbeddingError, create_embedder
    from vecshift.embeddings.providers import CONCURRENCY
    from vecshift.migrate import JobState

    o = options or ApplyOptions()
    into_qdrant = read_job(job_file).target.type == "qdrant"
    ready: Prepared | QdrantPrepared = (
        prepare_qdrant(job_file) if into_qdrant else prepare(job_file)
    )
    plan, job, spec = ready.plan, ready.job, ready.spec
    if not plan.ok:
        raise PlanBlocked(plan)
    if plan.dimensions is None:
        raise ServiceError(
            "The new vector size is unknown.",
            "Add dims= to the model spec, or check it with: vecshift plan --probe",
            "model.dimensions_unknown",
        )
    if job.limits.budget_usd is not None and spec.price is None:
        raise ServiceError(
            "A budget is set, but the model has no price, so it can't be enforced.",
            "Add price= (USD per million tokens) to the model spec.",
            "job.budget_without_price",
        )
    if isinstance(ready, Prepared) and (
        len(ready.target.primary_key) != 1 or ready.profile.text_field is None
    ):
        raise ServiceError(
            "Apply needs a single-column primary key and a text column.",
            code="source.unsupported",
        )

    state = JobState.for_job(job_file, job.name)
    changes = (
        f"This creates and fills the collection {plan.target_column.split(':')[0]}"
        if into_qdrant
        else f"This changes {plan.source.split()[0]}"
    )
    where = "your machine" if spec.is_local else spec.url or spec.name
    notice = ApplyNotice(plan, job, state, into_qdrant, changes, where)
    confirmed(confirm, notice, "changing anything")

    index = None
    if not o.no_index and job.target.index != "none":
        index = (str(getattr(job.target.index, "value", job.target.index)), plan.metric)

    closers: list[Any] = []
    try:
        writer = (
            _qdrant_writer(ready, state, plan.dimensions, closers)
            if isinstance(ready, QdrantPrepared)
            else _pg_writer(ready, plan.dimensions, closers)
        )
    except BaseException:
        for close in closers:
            close()
        raise

    removed = 0
    try:
        if on_start is not None:
            on_start()
        result = asyncio.run(
            run_engine(
                writer,
                create_embedder(spec),
                state,
                dims=plan.dimensions,
                price_per_million=spec.price,
                budget_usd=job.limits.budget_usd,
                chunk_rows=spec.batch_size * CONCURRENCY,
                index=index,
                on_event=on_event or (lambda e: None),
                should_stop=should_stop or (lambda: False),
                until=o.until / 100 if o.until else None,
            )
        )
        if into_qdrant and result.status == "complete":
            removed = writer.remove_deleted()
    except (AlreadyRunning, QdrantRunning) as exc:
        raise ServiceError(
            str(exc), "Wait for it to finish, or stop it first.", "job.running"
        ) from exc
    except Busy as exc:
        raise ServiceError(
            f"{exc}", "Progress is saved; run apply again to continue.", "database.busy"
        ) from exc
    except EmbeddingError as exc:
        raise ServiceError(
            f"{exc}", "Progress is saved; run apply again to continue.", "model.failed"
        ) from exc
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint or SAVED, "qdrant.error") from exc
    except psycopg.Error as exc:
        detail = first_line(exc)
        hint = SAVED
        if "shared memory segment" in detail:
            hint = (
                "The parallel index build ran out of shared memory, sized by "
                "maintenance_work_mem. In Docker, /dev/shm is 64 MB unless the container "
                "runs with --shm-size (e.g. 2g); or lower maintenance_work_mem. " + hint
            )
        raise ServiceError(
            f"The database reported an error: {detail}", hint, "database.error"
        ) from exc
    finally:
        for close in closers:
            close()

    copied = getattr(writer, "payloads_copied", 0)
    return ApplyOutcome(result, into_qdrant, removed, copied)


def _pg_writer(ready: Prepared, dims: int, closers: list[Any]) -> Any:
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.writer import Layout, PgWriter

    job = ready.job
    if ready.profile.text_field is None:  # checked by the caller
        raise ServiceError("Apply needs a text column.", code="source.unsupported")
    layout = Layout(
        schema=ready.target.source.schema,
        table=ready.target.source.table,
        pk=ready.target.primary_key[0],
        text=ready.profile.text_field,
        target=job.pg_target.column,
        vector_type=job.pg_target.vector_type.value,
        dims=dims,
        extension_schema=ready.target.extension_schema,
    )
    try:
        conn = pgvector.connect_writer(ready.settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    closers.append(conn.close)
    return PgWriter(conn, layout)


def _qdrant_writer(ready: QdrantPrepared, state: JobState, dims: int, closers: list[Any]) -> Any:
    from vecshift.connectors import pgvector
    from vecshift.connectors.qdrant import QdrantClient
    from vecshift.connectors.qdrant.writer import DISTANCES, Layout, QdrantSource, QdrantWriter

    client = QdrantClient(ready.qdrant)
    closers.append(client.close)
    source: Any
    if ready.source_qdrant is not None:
        reader = (
            client if ready.source_qdrant == ready.qdrant else QdrantClient(ready.source_qdrant)
        )
        if reader is not client:
            closers.append(reader.close)
        if ready.target.live is None:
            raise ServiceError("The source collection is unknown.", code="source.not_found")
        source = QdrantSource(reader, ready.target.live, ready.text_field, None)
    else:
        from vecshift.connectors.pgvector.reader import PgSource

        if ready.source_pg is None or ready.pg is None:
            raise ServiceError("The source table is unknown.", code="source.not_found")
        try:
            conn = pgvector.connect_reader(ready.source_pg)
        except pgvector.ConnectError as exc:
            raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
        closers.append(conn.close)
        source = PgSource(
            conn,
            ready.pg.source.schema,
            ready.pg.source.table,
            ready.pg.primary_key[0],
            ready.text_field,
        )
    layout = Layout(
        collection=ready.target.collection,
        vector=ready.target.vector,
        dims=dims,
        distance=DISTANCES[ready.plan.metric],
        payload_indexes=ready.target.payload_indexes,
    )
    return QdrantWriter(
        client,
        source,
        layout,
        state.path.with_suffix(".lock"),
        owned=state.collections,
        on_create=lambda name: owns(state, name),
    )


async def run_engine(writer: Any, embedder: Any, state: Any, **kwargs: Any) -> ApplyResult:
    """Run the migration engine, and always close the embedder."""
    from vecshift.migrate import apply as run_apply

    try:
        return await run_apply(writer, embedder, state, **kwargs)
    finally:
        await embedder.aclose()
