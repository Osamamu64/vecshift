"""The ``vecshift status`` command: where a migration stands, and what to run next.

Read-only: it connects like ``plan``, inside a read-only transaction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import DEFAULT_JOB, _fail, load

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

    @property
    def filled(self) -> float | None:
        if self.rows is None or self.missing is None:
            return None
        return 1.0 if self.rows == 0 else (self.rows - self.missing) / self.rows

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "filled": self.filled}


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


def collect(job_file: Path) -> Status:
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
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        live = resolve_source_column(
            conn, job.source.table, job.source.vector_column, job.target.column
        )
        source = select_column(pgvector.find_vector_columns(conn), job.source.table, live)
        columns = _columns(conn, source.relid)
        text = job.source.text_column or _pick(columns, TEXT_COLUMNS, TEXT_TYPES)
        layout = Layout(
            schema=source.schema,
            table=source.table,
            pk="",
            text=text or "",
            target=job.target.column,
            vector_type=job.target.vector_type.value,
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
        if stage == "filling" and missing == 0 and (index or job.target.index.value == "none"):
            stage = "ready"
        trigger = _trigger(conn, source.relid, layout.trigger_name())
    except pgvector.TargetSelectionError as exc:
        raise _fail(str(exc)) from exc
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


def _render(status: Status, job_name: str) -> None:
    typer.secho(f"vecshift status · {job_name}", bold=True)
    color = {
        "ready": typer.colors.GREEN,
        "cut_over": typer.colors.GREEN,
        "cleaned_up": typer.colors.GREEN,
        "conflict": typer.colors.RED,
    }.get(status.stage, typer.colors.BLUE)
    typer.echo("Stage    ", nl=False)
    typer.secho(STAGES[status.stage], fg=color, bold=True)
    typer.echo(f"Table    {status.table}")
    if status.stage == "cut_over":
        typer.echo(
            f"Columns  {status.live} holds the new vectors; the old are in {status.previous}"
        )
    elif status.stage == "cleaned_up":
        typer.echo(f"Columns  {status.live} holds the new vectors")
    else:
        typer.echo(f"Columns  {status.live} (live) → {status.new} (new)")
    if status.rows is not None and status.missing is not None and status.filled is not None:
        done = status.rows - status.missing
        width = 24
        bar = "█" * round(status.filled * width) + "░" * (width - round(status.filled * width))
        typer.echo(f"Vectors  {bar} {status.filled:.1%}  ({done:,} of {status.rows:,} rows)")
    if status.stage in {"filling", "ready", "cut_over", "conflict"}:
        typer.echo(f"Index    {status.index or 'none yet'}")
        typer.echo(f"Sync     {'trigger installed' if status.trigger else 'no trigger'}")
    if status.runs:
        failed = f", {status.failed:,} rejected by the provider" if status.failed else ""
        typer.echo(
            f"Apply    {status.runs} run{'s' if status.runs != 1 else ''}, "
            f"{status.rows_written:,} rows written, ${status.spent_usd:,.2f} spent{failed}"
        )
    if status.last_event:
        typer.echo(f"Last     {status.last_event.get('event')} at {status.last_event.get('at')}")
    typer.echo()
    typer.echo(f"Next: {status.next}")


def status(
    job_file: Annotated[
        Path, typer.Argument(help="The job file.", show_default=True)
    ] = DEFAULT_JOB,
    output_json: Annotated[bool, typer.Option("--json", help="Print the status as JSON.")] = False,
) -> None:
    """Show where a migration stands and what to run next. Read-only."""
    from vecshift.jobs import load_job

    result = collect(job_file)
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result, load_job(job_file).name)
