"""The ``vecshift apply`` command."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import (
    DEFAULT_JOB,
    Prepared,
    QdrantPrepared,
    _fail,
    _money,
    _size,
    prepare,
    prepare_qdrant,
    read_job,
)
from vecshift.cli_style import finding_lines

if TYPE_CHECKING:
    from vecshift.jobs import JobSpec
    from vecshift.migrate import ApplyResult, Event, JobState
    from vecshift.planning import Plan

EXIT_STOPPED = 3
"""Stopped on purpose (Ctrl-C or budget): run apply again to continue."""


class _Stop:
    """First Ctrl-C finishes the current batch and stops cleanly; a second one aborts."""

    def __init__(self) -> None:
        self.requested = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        typer.secho(
            "\nStopping after the current batch. Press Ctrl-C again to abort now.",
            err=True,
            fg=typer.colors.YELLOW,
        )


class _Progress:
    def __init__(self, live: bool, noun: str = "column") -> None:
        self.live = live
        self.noun = noun
        self.started = time.monotonic()
        self.pending = 0
        self.last_print = 0.0

    def __call__(self, event: Event) -> None:
        d = event.data
        if event.kind == "start":
            self.pending = d["pending"]
            typer.echo(f"{self.pending:,} rows need a new vector.")
        elif event.kind == "column" and d["added"]:
            typer.echo(f"✔ {_added(self.noun)}.")
        elif event.kind == "trigger" and d["added"]:
            typer.echo("✔ Added the sync trigger.")
        elif event.kind == "batch":
            self._batch(d)
        elif event.kind == "pass" and d["number"] > 1 and self.live:
            typer.echo()
        elif event.kind == "index":
            if self.live:
                typer.echo()
            if d["state"] == "building":
                typer.echo(_building(self.noun, d["method"]))
            else:
                typer.echo(f"✔ {_indexed(self.noun, d['state'])}.")

    def _batch(self, d: dict[str, Any]) -> None:
        done = d["rows_written"]
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = done / elapsed
        left = max(self.pending - done, 0)
        eta = f"~{_eta(left / rate)} left" if rate > 0 and left else ""
        cost = f" · {_money(d['spent_usd'])}" if d["spent_usd"] is not None else ""
        line = f"  {done:,} / {self.pending:,} rows · {rate:,.0f} rows/s{cost} {eta}"
        if self.live:
            typer.echo("\r" + line.ljust(78), nl=False)
        elif time.monotonic() - self.last_print > 15 or not left:
            typer.echo(line)
            self.last_print = time.monotonic()


class _FancyProgress:
    """A live progress bar and spinners, for a person at a colour terminal."""

    def __init__(self, noun: str = "column") -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        from vecshift import ui

        self.ui = ui
        self.noun = noun
        self.started = time.monotonic()
        self.progress = Progress(
            SpinnerColumn(style=ui.ACCENT),
            TextColumn("{task.description}"),
            BarColumn(complete_style=ui.ACCENT, finished_style="green", bar_width=32),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[rate]}"),
            TextColumn("[dim]{task.fields[cost]}"),
            TimeRemainingColumn(compact=True),
            console=ui.console,
            transient=False,
        )
        self.task: Any = None
        self.status: Any = None
        self.done = 0

    def __call__(self, event: Event) -> None:
        ui, d = self.ui, event.data
        if event.kind == "column" and d["added"]:
            ui.success(_added(self.noun))
        elif event.kind == "trigger" and d["added"]:
            ui.success("Added the sync trigger")
        elif event.kind == "start":
            ui.note(f"  {d['pending']:,} rows need a new vector")
            self.progress.start()
            self.task = self.progress.add_task("Embedding", total=d["pending"], rate="", cost="")
        elif event.kind == "batch" and self.task is not None:
            self.done = d["rows_written"]
            rate = self.done / max(time.monotonic() - self.started, 1e-6)
            cost = _money(d["spent_usd"]) if d["spent_usd"] is not None else ""
            self.progress.update(
                self.task, completed=self.done, rate=f"{rate:,.0f} rows/s", cost=cost
            )
        elif event.kind == "pass" and d["number"] > 1 and self.task is not None:
            self.progress.update(
                self.task, description="Catching up", total=self.done + d["remaining"]
            )
        elif event.kind == "index":
            self.close()
            if d["state"] == "building":
                self.status = ui.console.status(
                    _building(self.noun, d["method"]),
                    spinner="dots",
                    spinner_style=ui.ACCENT,
                )
                self.status.start()
            else:
                ui.success(_indexed(self.noun, d["state"]))
        elif event.kind == "done":
            self.close()

    def close(self) -> None:
        if self.task is not None:
            self.progress.stop()
            self.task = None
        if self.status is not None:
            self.status.stop()
            self.status = None


def _added(noun: str) -> str:
    return "Created the new collection" if noun == "collection" else "Added the new column"


def _building(noun: str, method: str) -> str:
    if noun == "collection":
        return "Waiting for Qdrant to finish indexing the new collection…"
    return f"Building the {method} index concurrently. Writes continue…"


def _indexed(noun: str, state: str) -> str:
    if noun != "collection":
        return f"Index {state}"
    if state == "pending":
        return "Qdrant is still indexing; vecshift status shows when it's done"
    return "Qdrant has indexed the new collection"


def _eta(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


QDRANT_NEXT = (
    "check progress with vecshift status. Run apply again any time to copy what changed in "
    "the source since."
)


def _summary_fancy(result: ApplyResult, into_qdrant: bool = False) -> None:
    from vecshift import ui

    kind = {"complete": "ok", "failed": "error"}.get(result.status, "warn")
    ui.verdict(f"Apply {result.status}.", kind)
    if result.message:
        ui.detail(result.message)
    spent = _money(result.spent_usd) if result.spent_usd is not None else "unknown"
    total = _money(result.total_spent_usd) if result.total_spent_usd is not None else ""
    ui.console.print()
    ui.kv(
        [
            ("Rows written", f"{result.rows_written:,}", ""),
            ("Tokens", f"{result.tokens:,}", ""),
            ("Spent", spent, f"{total} across runs" if total else ""),
            ("Time", f"{result.seconds:,.0f} s", ""),
        ]
        + (
            [("Rejected", f"{result.rows_failed:,} rows", "retried on the next run")]
            if result.rows_failed
            else []
        )
    )
    if result.status == "complete" and into_qdrant:
        ui.next_step(QDRANT_NEXT)
    elif result.status == "complete":
        ui.next_step(
            "check it's safe to switch with vecshift cutover --check, then run vecshift "
            "cutover. It embeds rows added or edited since this run first."
        )
    elif result.status in {"stopped", "budget"}:
        ui.next_step("run vecshift apply again to continue where it stopped.")


def _summary(result: ApplyResult, into_qdrant: bool = False) -> None:
    from vecshift import ui

    if ui.fancy():
        _summary_fancy(result, into_qdrant)
        return
    color = {"complete": typer.colors.GREEN, "failed": typer.colors.RED}.get(
        result.status, typer.colors.YELLOW
    )
    typer.echo()
    typer.secho(f"Apply {result.status}.", fg=color, bold=True)
    if result.message:
        typer.echo(result.message)
    spent = _money(result.spent_usd) if result.spent_usd is not None else "unknown"
    total = (
        f" ({_money(result.total_spent_usd)} across runs)"
        if result.total_spent_usd is not None
        else ""
    )
    typer.echo(
        f"  Rows written {result.rows_written:,} · tokens {result.tokens:,} · spent {spent}"
        f"{total} · {result.seconds:,.0f} s"
    )
    if result.rows_failed:
        typer.echo(f"  Rejected by the provider: {result.rows_failed:,} rows (retried next run)")
    if result.status == "complete" and into_qdrant:
        typer.echo(f"\nNext: {QDRANT_NEXT}")
    elif result.status == "complete":
        typer.echo(
            "\nNext: check it's safe to switch with `vecshift cutover --check`, then run "
            "`vecshift cutover`.\nIt embeds any rows added or edited since this run first."
        )


def apply(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before starting.")] = False,
    no_index: Annotated[
        bool, typer.Option("--no-index", help="Backfill only; build the index on a later run.")
    ] = False,
    until: Annotated[
        int | None,
        typer.Option(
            "--until",
            min=1,
            max=99,
            metavar="PERCENT",
            help="Stop once this share of rows has a new vector, to check before continuing.",
        ),
    ] = None,
    output_json: Annotated[
        bool, typer.Option("--json", help="Print progress as JSON lines, for scripts and UIs.")
    ] = False,
) -> None:
    """Run (or resume) a migration: add the column, embed every row, and build the index."""
    import psycopg

    from vecshift.connectors import qdrant
    from vecshift.connectors.pgvector.writer import AlreadyRunning, Busy
    from vecshift.connectors.qdrant.writer import AlreadyRunning as QdrantRunning
    from vecshift.embeddings import EmbeddingError, create_embedder
    from vecshift.embeddings.providers import CONCURRENCY
    from vecshift.migrate import JobState

    into_qdrant = read_job(job_file).target.type == "qdrant"
    ready: Prepared | QdrantPrepared = (
        prepare_qdrant(job_file) if into_qdrant else prepare(job_file)
    )
    plan, job, spec = ready.plan, ready.job, ready.spec
    if not plan.ok:
        typer.secho("The plan has errors, so nothing was changed.", fg=typer.colors.RED, bold=True)
        finding_lines([f for f in plan.findings if f.severity.rank >= 2])
        raise typer.Exit(1)
    if plan.dimensions is None:
        raise _fail(
            "The new vector size is unknown.",
            "Add dims= to the model spec, or check it with: vecshift plan --probe",
        )
    if job.limits.budget_usd is not None and spec.price is None:
        raise _fail(
            "A budget is set, but the model has no price, so it can't be enforced.",
            "Add price= (USD per million tokens) to the model spec.",
        )
    if isinstance(ready, Prepared) and (
        len(ready.target.primary_key) != 1 or ready.profile.text_field is None
    ):
        raise _fail("Apply needs a single-column primary key and a text column.")

    state = JobState.for_job(job_file, job.name)
    noun = "collection" if into_qdrant else "column"
    changes = (
        f"This creates and fills the collection {plan.target_column.split(':')[0]}"
        if into_qdrant
        else f"This changes {plan.source.split()[0]}"
    )
    _announce(plan, job, state, output_json, noun)
    from vecshift import ui

    fancy = ui.fancy() and not output_json
    where = "your machine" if spec.is_local else spec.url
    if not yes and fancy:
        ui.console.print()
        ui.note(f"  {changes} and sends row text to {where}.")
        if not ui.confirm("Apply?", default=False):
            raise typer.Exit(1)
    elif not yes:
        typer.echo(f"\n{changes} and sends row text to {where}.", err=True)
        if not sys.stdin.isatty():
            raise _fail("Not changing anything without confirmation.", "Pass --yes to proceed.")
        if not typer.confirm("Apply?", err=True):
            raise typer.Exit(1)

    index = None
    if not no_index and job.target.index != "none":
        index = (str(getattr(job.target.index, "value", job.target.index)), plan.metric)

    closers: list[Any] = []
    try:
        writer = (
            _qdrant_writer(ready, job_file, plan.dimensions, closers)
            if isinstance(ready, QdrantPrepared)
            else _pg_writer(ready, plan.dimensions, closers)
        )
    except BaseException:
        for close in closers:
            close()
        raise

    stop = _Stop()
    previous = signal.signal(signal.SIGINT, stop)
    if output_json:

        def on_event(event: Event) -> None:
            typer.echo(json.dumps({"event": event.kind, **event.data}))

    elif fancy:
        ui.console.print()
        on_event = _FancyProgress(noun)
    else:
        on_event = _Progress(live=sys.stdout.isatty(), noun=noun)
    removed = 0
    try:
        embedder = create_embedder(spec)
        result = asyncio.run(
            _run(
                writer,
                embedder,
                state,
                dims=plan.dimensions,
                price_per_million=spec.price,
                budget_usd=job.limits.budget_usd,
                chunk_rows=spec.batch_size * CONCURRENCY,
                index=index,
                on_event=on_event,
                should_stop=lambda: stop.requested,
                until=until / 100 if until else None,
            )
        )
        if into_qdrant and result.status == "complete":
            removed = writer.remove_deleted()
    except (AlreadyRunning, QdrantRunning) as exc:
        raise _fail(str(exc), "Wait for it to finish, or stop it first.") from exc
    except (Busy, EmbeddingError) as exc:
        raise _fail(f"{exc}", "Progress is saved; run apply again to continue.") from exc
    except qdrant.QdrantError as exc:
        hint = exc.hint or "Every finished batch is saved; run apply again to continue."
        raise _fail(str(exc), hint) from exc
    except psycopg.Error as exc:
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        hint = "Every finished batch is saved; run apply again to continue."
        if "shared memory segment" in detail:
            hint = (
                "The parallel index build ran out of shared memory, sized by "
                "maintenance_work_mem. In Docker, /dev/shm is 64 MB unless the container "
                "runs with --shm-size (e.g. 2g); or lower maintenance_work_mem. " + hint
            )
        raise _fail(f"The database reported an error: {detail}", hint) from exc
    finally:
        if isinstance(on_event, _FancyProgress):
            on_event.close()
        signal.signal(signal.SIGINT, previous)
        for close in closers:
            close()

    copied = getattr(writer, "payloads_copied", 0)
    if output_json:
        extra = {"deleted": removed, "payloads_copied": copied} if into_qdrant else {}
        typer.echo(json.dumps({"event": "result", **_result_dict(result), **extra}))
    else:
        if removed or copied:
            parts = []
            if copied:
                parts.append(f"copied {copied:,} changed payloads")
            if removed:
                parts.append(f"removed {removed:,} points deleted from the source")
            message = "Also " + " and ".join(parts) + "."
            if fancy:
                ui.success(message)
            else:
                typer.echo(message)
        _summary(result, into_qdrant)
    if result.status == "failed":
        raise typer.Exit(1)
    if result.status in {"stopped", "budget"}:
        raise typer.Exit(EXIT_STOPPED)


def _announce(plan: Plan, job: JobSpec, state: JobState, output_json: bool, noun: str) -> None:
    """What apply is about to do, before it asks."""
    from vecshift import ui

    est = plan.estimates
    warnings = [f for f in plan.findings if f.severity.rank == 2]
    if output_json:
        return
    if ui.fancy():
        from rich.text import Text

        from vecshift.cli_plan import CHANGE_MARKS

        ui.title("apply", job.name)
        for change in plan.changes:
            if change.kind != "cutover":
                mark, colour = CHANGE_MARKS.get(change.kind, ("·", "dim"))
                ui.console.print(Text.assemble((f"  {mark} ", f"bold {colour}"), change.summary))
        ui.console.print()
        rows: list[tuple[str, str, str]] = [
            ("Estimated cost", _money(est.cost_usd), ""),
            ("New vectors" if noun == "collection" else "New column", _size(est.new_bytes), ""),
        ]
        if job.limits.budget_usd:
            rows.append(("Budget", f"${job.limits.budget_usd:,.2f}", "apply stops there"))
        if state.runs:
            rows.append(
                (
                    "Resuming",
                    f"{state.rows_written:,} rows",
                    f"written in earlier runs, {_money(state.spent_usd)} spent",
                )
            )
        ui.kv(rows)
        if warnings:
            ui.console.print()
            ui.findings(warnings)
        return
    typer.secho(f"vecshift apply · {job.name}", bold=True)
    for change in plan.changes:
        if change.kind != "cutover":
            typer.echo(f"  · {change.summary}")
    cost = _money(est.cost_usd)
    budget = f", budget ${job.limits.budget_usd:,.2f}" if job.limits.budget_usd else ""
    size = "new vectors" if noun == "collection" else "new column"
    typer.echo(f"  Estimated cost {cost}{budget}; {size} {_size(est.new_bytes)}.")
    if state.runs:
        typer.echo(
            f"  Resuming: {state.rows_written:,} rows written in earlier runs, "
            f"{_money(state.spent_usd)} spent."
        )
    if warnings:
        finding_lines(warnings)


def _pg_writer(ready: Prepared, dims: int, closers: list[Any]) -> Any:
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.writer import Layout, PgWriter

    job = ready.job
    if ready.profile.text_field is None:  # checked by the caller
        raise _fail("Apply needs a text column.")
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
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    closers.append(conn.close)
    return PgWriter(conn, layout)


def _qdrant_writer(ready: QdrantPrepared, job_file: Path, dims: int, closers: list[Any]) -> Any:
    from vecshift.connectors import pgvector
    from vecshift.connectors.qdrant import QdrantClient
    from vecshift.connectors.qdrant.writer import DISTANCES, Layout, QdrantSource, QdrantWriter
    from vecshift.migrate import JobState

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
            raise _fail("The source collection is unknown.")
        source = QdrantSource(reader, ready.target.live, ready.text_field, None)
    else:
        from vecshift.connectors.pgvector.reader import PgSource

        if ready.source_pg is None or ready.pg is None:
            raise _fail("The source table is unknown.")
        try:
            conn = pgvector.connect_reader(ready.source_pg)
        except pgvector.ConnectError as exc:
            raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
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
    lock = JobState.for_job(job_file, ready.job.name).path.with_suffix(".lock")
    return QdrantWriter(client, source, layout, lock)


async def _run(writer: Any, embedder: Any, state: Any, **kwargs: Any) -> ApplyResult:
    from vecshift.migrate import apply as run_apply

    try:
        return await run_apply(writer, embedder, state, **kwargs)
    finally:
        await embedder.aclose()


def _result_dict(result: ApplyResult) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(result)
