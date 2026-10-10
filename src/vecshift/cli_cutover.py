"""The ``vecshift cutover`` and ``vecshift rollback`` commands."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import DEFAULT_JOB, _fail, load
from vecshift.cli_style import finding_lines

if TYPE_CHECKING:
    import psycopg

    from vecshift.connectors.pgvector.switch import PgSwitch
    from vecshift.connectors.pgvector.writer import Layout
    from vecshift.doctor.findings import Finding
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.migrate import JobState

CATCH_UP_TRIES = 3


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

    job, settings, spec = load(job_file)
    try:
        conn = pgvector.connect_writer(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        live = resolve_source_column(
            conn, job.source.table, job.source.vector_column, job.target.column
        )
        target = inspect_target(conn, job.source.table, live, job.target.column)
        text = job.source.text_column or _pick(
            _columns(conn, target.source.relid), TEXT_COLUMNS, TEXT_TYPES
        )
    except pgvector.TargetSelectionError as exc:
        conn.close()
        raise _fail(str(exc)) from exc
    if len(target.primary_key) != 1 or text is None:
        conn.close()
        raise _fail("Cutover needs a single-column primary key and a text column.")
    dims = target.column_dimensions or target.source.dimensions or spec.dimensions or 0
    layout = Layout(
        schema=target.source.schema,
        table=target.source.table,
        pk=target.primary_key[0],
        text=text,
        target=job.target.column,
        vector_type=job.target.vector_type.value,
        dims=dims,
        extension_schema=target.extension_schema,
        live=live,
    )
    return _Located(job, spec, conn, layout, JobState.for_job(job_file, job.name))


def _confirm(message: str, yes: bool) -> None:
    if yes:
        return
    typer.echo(message, err=True)
    if not sys.stdin.isatty():
        raise _fail("Not changing the database without confirmation.", "Pass --yes to proceed.")
    if not typer.confirm("Continue?", err=True):
        raise typer.Exit(1)


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
    from vecshift.cli_apply import _run
    from vecshift.connectors.pgvector.writer import PgWriter
    from vecshift.embeddings import create_embedder
    from vecshift.embeddings.providers import CONCURRENCY

    spec = found.spec
    result = asyncio.run(
        _run(
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
        raise _fail(f"The final catch-up stopped: {result.message}", "Nothing was switched.")
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


def _show(findings: list[Finding], output_json: bool) -> None:
    if not output_json:
        finding_lines(findings)


def _guarded(run: Any) -> Any:
    """Turn database and lock problems into clean messages."""
    import psycopg

    from vecshift.connectors.pgvector.writer import AlreadyRunning, Busy
    from vecshift.embeddings import EmbeddingError

    try:
        return run()
    except AlreadyRunning as exc:
        raise _fail(str(exc), "Wait for it to finish, or stop it first.") from exc
    except (Busy, EmbeddingError) as exc:
        raise _fail(str(exc), "Nothing was switched; try again.") from exc
    except psycopg.Error as exc:
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise _fail(f"The database reported an error: {detail}", "Nothing was switched.") from exc


def cutover(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    check: Annotated[
        bool, typer.Option("--check", help="Only check that cutover is safe; change nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before switching.")] = False,
    allow_missing: Annotated[
        bool,
        typer.Option(
            "--allow-missing",
            help="Switch even though the provider rejected some rows (they get no vector).",
        ),
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Switch searches to the new vectors by giving them the column name your app uses."""
    from vecshift.connectors.pgvector.switch import PgSwitch, StillPending
    from vecshift.connectors.pgvector.writer import PgWriter

    found = _locate(job_file)
    lay, state = found.layout, found.state
    writer = PgWriter(found.conn, lay)
    switch = PgSwitch(writer)
    index = found.job.target.index.value
    method = None if index == "none" else index

    def run() -> dict[str, Any]:
        writer.acquire()
        try:
            ready = switch.check(method)
            blocked = any(
                f.severity.rank == 3 and f.id != "cutover.pending" for f in ready.findings
            )
            if check or blocked:
                return {"status": "ready" if ready.ok else "blocked", "readiness": ready}
            if not output_json:
                typer.secho(f"vecshift cutover · {found.job.name}", bold=True)
            _show([f for f in ready.findings if f.severity.rank >= 1], output_json)
            where = "your machine" if found.spec.is_local else found.spec.url
            sends = (
                f" First it embeds {ready.pending:,} rows that changed since the last apply, "
                f"sending their text to {where}."
                if ready.pending
                else ""
            )
            _confirm(
                f"\nCutover renames {lay.live} to {lay.previous} and {lay.target} to "
                f"{lay.live} on {lay.schema}.{lay.table}, so searches use {found.spec.name} "
                f"vectors.{sends} Switch your application to {found.spec.name} at the same time.",
                yes,
            )
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
                            raise _fail(
                                f"{exc.rows:,} rows still have no new vector.", hint
                            ) from exc
                        caught_up += _catch_up(found, switch)
            finally:
                switch.finish()
            _record(state, "cutover", lay)
            return {"status": "cut_over", "readiness": ready, "caught_up": caught_up}
        finally:
            writer.release()

    try:
        outcome = _guarded(run)
    finally:
        found.conn.close()

    ready = outcome["readiness"]
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": outcome["status"],
                    "live": lay.live,
                    "previous": lay.previous,
                    "new": lay.target,
                    "pending": ready.pending,
                    "caught_up": outcome.get("caught_up", 0),
                    "findings": [f.to_dict() for f in ready.findings],
                }
            )
        )
    elif outcome["status"] in {"ready", "blocked"}:
        typer.secho(f"vecshift cutover · {found.job.name}", bold=True)
        finding_lines(ready.findings)
        verdict = (
            ("Ready to cut over.", typer.colors.GREEN)
            if ready.ok
            else ("Not ready to cut over.", typer.colors.RED)
        )
        typer.secho(f"\n{verdict[0]}", fg=verdict[1], bold=True)
    else:
        typer.secho("\nCut over.", fg=typer.colors.GREEN, bold=True)
        typer.echo(
            f"  {lay.table}.{lay.live} now holds {found.spec.name} vectors; the old ones are "
            f"kept in {lay.previous}."
        )
        if outcome.get("caught_up"):
            typer.echo(f"  Embedded {outcome['caught_up']:,} late rows first.")
        typer.echo(
            f"\nNext: make sure your application embeds queries and new rows with "
            f"{found.spec.name}.\nUndo with: vecshift rollback. When you're sure, drop "
            f"{lay.previous} to free its space."
        )
    if outcome["status"] == "blocked":
        raise typer.Exit(1)


def rollback(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before switching.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Undo a cutover: give the old vectors their column name back."""
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
                raise _fail(
                    "There's no cutover to roll back.",
                    f"Rollback needs {lay.previous}, holding the old vectors, and no "
                    f"{lay.target} column.",
                )
            bound = switch.dependents(lay.live) + switch.dependents(lay.previous)
            if bound:
                raise _fail(
                    f"Views or functions are bound to the vector columns: {', '.join(bound)}.",
                    "Drop them before rollback and recreate them after it.",
                )
            _confirm(
                f"Rollback renames {lay.live} to {lay.target} and {lay.previous} to {lay.live} "
                f"on {lay.schema}.{lay.table}, so searches use the old vectors again. Switch "
                "your application back to the old model at the same time.",
                yes,
            )
            missing = switch.rollback()
            _record(state, "rollback", lay)
            return missing
        finally:
            writer.release()

    try:
        missing = _guarded(run)
    finally:
        found.conn.close()

    if output_json:
        typer.echo(
            json.dumps(
                {"status": "rolled_back", "live": lay.live, "new": lay.target, "missing": missing}
            )
        )
        return
    typer.secho("\nRolled back.", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        f"  {lay.table}.{lay.live} holds the old vectors again; the new ones are back in "
        f"{lay.target}, kept in sync for another cutover."
    )
    if missing:
        typer.echo(
            f"  {missing:,} rows were added or edited after cutover and have no old-model "
            "vector. Your application needs to embed them with the old model."
        )


def cleanup(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Don't ask for the column name first.")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """After cutover, drop the old vectors for good. This can't be undone."""
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
                raise _fail(
                    "There's no cutover to clean up after.",
                    f"Cleanup drops {lay.previous}, which cutover creates.",
                )
            bound = switch.dependents(lay.previous)
            if bound:
                raise _fail(
                    f"Views or functions use {lay.previous}: {', '.join(bound)}.",
                    "Drop or change them first.",
                )
            if not yes:
                typer.echo(
                    f"Cleanup drops {lay.schema}.{lay.table}.{lay.previous} and its index. The "
                    "old vectors are gone for good, and rollback is no longer possible.",
                    err=True,
                )
                if not sys.stdin.isatty():
                    raise _fail("Not dropping anything without confirmation.", "Pass --yes.")
                typed = typer.prompt(f"Type {lay.previous} to confirm", err=True, default="")
                if typed != lay.previous:
                    raise _fail("That didn't match, so nothing was dropped.")
            switch.cleanup()
            _record(state, "cleanup", lay)
        finally:
            writer.release()

    try:
        _guarded(run)
    finally:
        found.conn.close()
    if output_json:
        typer.echo(json.dumps({"status": "cleaned_up", "dropped": lay.previous}))
    else:
        typer.secho(f"\nDropped {lay.previous}. The migration is complete.", fg=typer.colors.GREEN)
