"""The ``vecshift plan`` command, and loading a job for the commands that use one."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_style import finding_lines

if TYPE_CHECKING:
    from vecshift.connectors.pgvector import ConnectionSettings
    from vecshift.connectors.pgvector.target import TargetState
    from vecshift.connectors.qdrant import QdrantSettings
    from vecshift.connectors.qdrant.target import QdrantTargetState
    from vecshift.doctor import IndexProfile
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.planning import Plan, ProbeResult, SampleStats

DEFAULT_JOB = Path("vecshift.yaml")
PROBE_DOCUMENTS = 16


def _fail(message: str, hint: str | None = None) -> typer.Exit:
    typer.secho(message, err=True, fg=typer.colors.RED)
    if hint:
        typer.echo(f"→ {hint}", err=True)
    return typer.Exit(2)


def _size(n: int | None) -> str:
    if n is None:
        return "—"
    for unit, size in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:,.1f} {unit}"
    return f"{n} B"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 90:
        return f"~{max(1, round(seconds))} s"
    if seconds < 90 * 60:
        return f"~{round(seconds / 60)} min"
    if seconds < 48 * 3600:
        return f"~{seconds / 3600:.1f} h"
    return f"~{seconds / 86400:.1f} days"


def _compact(n: int | None) -> str:
    if n is None:
        return "—"
    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= size * 10:
            return f"{n / size:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"{n:,}"


def _note(text: str) -> str:
    width = min(100, shutil.get_terminal_size((100, 24)).columns)
    return textwrap.fill(text, width=width, initial_indent=" " * 6, subsequent_indent=" " * 6)


def _money(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value == 0:
        return "$0.00"
    return "<$0.01" if value < 0.01 else f"~${value:,.2f}"


def _estimate_rows(plan: Plan) -> list[tuple[str, str, str]]:
    e = plan.estimates
    rows = "—" if e.rows is None else (f"{e.rows:,}" if e.rows_exact else f"~{e.rows:,}")
    lines = [
        ("Rows to embed", rows, "exact" if e.rows_exact else "from table statistics"),
        ("Tokens", "~" + _compact(e.tokens) if e.tokens else "—", e.tokens_method or ""),
        ("Cost", _money(e.cost_usd), ""),
        ("Requests", f"{e.requests:,}" if e.requests is not None else "—", ""),
        ("Duration", _duration(e.seconds), e.seconds_method or "add a rate limit or --probe"),
        (
            "New vectors" if plan.store == "qdrant" else "New column",
            _size(e.new_bytes),
            (
                f"old vectors {_size(e.old_bytes)}"
                if plan.store == "qdrant"
                else f"old column {_size(e.old_bytes)}"
            )
            if e.old_bytes
            else "",
        ),
    ]
    if e.index_memory_bytes:
        lines.append(
            (
                "Index build",
                f"~{_size(e.index_memory_bytes)}",
                f"maintenance_work_mem is {_size(e.maintenance_work_mem)}",
            )
        )
    return lines


CHANGE_MARKS = {
    "add_column": ("+", "green"),
    "create_collection": ("+", "green"),
    "trigger": ("+", "green"),
    "embed": ("~", "#3987e5"),
    "index": ("+", "green"),
    "cutover": ("⇄", "magenta"),
}


def _render_fancy(plan: Plan) -> None:
    from rich.padding import Padding
    from rich.text import Text

    from vecshift import ui

    dims = f"{plan.dimensions:,}" if plan.dimensions else "?"
    source_note = {
        "spec": "set in the spec",
        "known": "this model's size",
        "probe": "measured",
        None: "unknown",
    }[plan.dimensions_source]
    ui.title("plan", plan.job)
    ui.kv(
        [
            ("Source", plan.source, ""),
            ("Model", Text.assemble(plan.model, (" → ", "dim"), f"{dims} dimensions"), source_note),
        ]
    )
    ui.section("Changes")
    for change in plan.changes:
        mark, colour = CHANGE_MARKS.get(change.kind, ("·", "dim"))
        ui.console.print(Text.assemble((f"  {mark} ", f"bold {colour}"), change.summary))
        if change.sql:
            ui.sql(change.sql)
        if change.note:
            ui.console.print(Padding(Text(change.note, style="dim"), (0, 0, 0, 4)))
    ui.section("Estimates")
    ui.kv(_estimate_rows(plan))
    if plan.findings:
        ui.section("Checks")
        ui.findings(plan.findings)
    if plan.ok:
        ui.verdict("Ready to apply.")
        ui.next_step("vecshift apply")
    else:
        noun = "error" if plan.errors == 1 else "errors"
        ui.verdict(f"Fix {plan.errors} {noun} before applying.", "error")
        ui.console.print()


def _render(plan: Plan) -> None:
    from vecshift import ui

    if ui.fancy():
        _render_fancy(plan)
        return
    dims = f"{plan.dimensions:,}" if plan.dimensions else "?"
    source_note = {
        "spec": "set in the spec",
        "known": "this model's size",
        "probe": "measured",
        None: "unknown",
    }[plan.dimensions_source]
    typer.secho(f"vecshift plan · {plan.job}", bold=True)
    typer.echo(f"Source  {plan.source}")
    typer.echo(f"Model   {plan.model} → {dims} dimensions ({source_note})")

    typer.secho("\nChanges", bold=True)
    marks = {k: mark for k, (mark, _) in CHANGE_MARKS.items()}
    for change in plan.changes:
        typer.echo(f"  {marks.get(change.kind, '·')} {change.summary}")
        if change.sql:
            typer.secho(f"      {change.sql}", fg=typer.colors.CYAN)
        if change.note:
            typer.secho(_note(change.note), dim=True)

    typer.secho("\nEstimates", bold=True)
    lines = _estimate_rows(plan)
    width = max(len(k) for k, _, _ in lines)
    for key, value, note in lines:
        typer.echo(f"  {key:<{width}}  {value:<10}  " + typer.style(note, dim=True))

    if plan.findings:
        typer.secho("\nChecks", bold=True)
        finding_lines(plan.findings)
    typer.echo()
    if plan.ok:
        typer.secho("Ready to apply.", fg=typer.colors.GREEN, bold=True)
    else:
        noun = "error" if plan.errors == 1 else "errors"
        typer.secho(f"Fix {plan.errors} {noun} before applying.", fg=typer.colors.RED, bold=True)


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


@dataclass(slots=True)
class Prepared:
    """Everything known about a job before anything changes."""

    job: JobSpec
    settings: ConnectionSettings
    spec: ModelSpec
    profile: IndexProfile
    target: TargetState
    plan: Plan


def read_job(job_file: Path) -> JobSpec:
    from vecshift.jobs import JobError, load_job

    try:
        return load_job(job_file)
    except JobError as exc:
        hint = (
            "Create one with: vecshift init --table ... --model ..."
            if not job_file.exists()
            else None
        )
        raise _fail(str(exc), hint) from exc


def pg_settings(variable: str) -> ConnectionSettings:
    """Connection settings from the variable holding the connection string."""
    from vecshift.connectors import pgvector

    dsn = os.environ.get(variable)
    if not dsn:
        raise _fail(
            f"{variable} isn't set.",
            f"Export the database connection string as {variable}, put it in a .env file "
            "here, or run vecshift init to set it up.",
        )
    try:
        return pgvector.prepare(dsn)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc


def qdrant_settings(url_env: str, key_env: str) -> QdrantSettings:
    from vecshift.connectors import qdrant

    url = os.environ.get(url_env)
    if not url:
        raise _fail(
            f"{url_env} isn't set.",
            f"Export the Qdrant URL as {url_env} (and the API key as {key_env}, if the "
            "server needs one), or put them in a .env file here.",
        )
    try:
        return qdrant.prepare(url, key_env)
    except qdrant.QdrantError as exc:
        raise _fail(str(exc), exc.hint) from exc


def load(
    job_file: Path, dsn_env: str | None = None, *, command: str = "This command"
) -> tuple[JobSpec, ConnectionSettings, ModelSpec]:
    """The job, its connection settings (from the environment), and its model.

    ``dsn_env`` reads the connection string from another variable, such as a replica's.
    """
    from vecshift.embeddings import parse_spec

    job = read_job(job_file)
    if job.target.type == "qdrant":
        raise _fail(
            f"{command} can't handle migrations into Qdrant yet.",
            "vecshift plan, apply, and status do; the rest is coming in the next release.",
        )
    settings = pg_settings(dsn_env or job.pg_source.dsn_env)
    return job, settings, parse_spec(job.model)


def _measure(
    spec: ModelSpec, texts: list[str], *, probe: bool, yes: bool
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
        if not spec.is_local and not yes:
            typer.echo(
                f"The probe sends {len(chosen)} sample rows to {spec.url} to measure {spec.name}.",
                err=True,
            )
            if not sys.stdin.isatty():
                raise _fail("Not sending data without confirmation.", "Pass --yes to proceed.")
            if not typer.confirm("Continue?", err=True):
                raise typer.Exit(1)
        try:
            measured = asyncio.run(_probe(spec, chosen))
        except EmbeddingError as exc:
            raise _fail(f"The probe failed: {exc}") from exc

    return stats, measured


def prepare(
    job_file: Path, *, sample_size: int = 2000, probe: bool = False, yes: bool = False
) -> Prepared:
    """Load the job, inspect the database read-only, and build the plan."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.documents import sample_documents
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column
    from vecshift.planning import build_plan

    job, settings, spec = load(job_file)
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
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
        raise _fail(str(exc)) from exc
    finally:
        conn.rollback()
        conn.close()

    stats, measured = _measure(spec, texts, probe=probe, yes=yes)
    result = build_plan(job, profile, target, stats, measured)
    return Prepared(job, settings, spec, profile, target, result)


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


def prepare_qdrant(
    job_file: Path, *, sample_size: int = 2000, probe: bool = False, yes: bool = False
) -> QdrantPrepared:
    """Load a job whose target is Qdrant, inspect both sides read-only, and plan it."""
    from vecshift.connectors import pgvector, qdrant
    from vecshift.connectors.qdrant.target import inspect_target
    from vecshift.embeddings import parse_spec
    from vecshift.jobs.spec import QdrantSource
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
                    texts = _qdrant_texts(client, field.collection, text_field, sample_size)
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
            profile, pg, texts = _pg_source(job, source_pg, sample_size)
            text_field = profile.text_field or ""
            with qdrant.QdrantClient(settings) as client:
                target = inspect_target(
                    client, job.target_collection, target_spec.vector, target_spec.alias
                )
    except qdrant.QdrantError as exc:
        raise _fail(str(exc), exc.hint) from exc
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    except pgvector.TargetSelectionError as exc:
        raise _fail(str(exc)) from exc

    stats, measured = _measure(spec, texts, probe=probe, yes=yes)
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


def _qdrant_texts(client: Any, collection: str, text_field: str, size: int) -> list[str]:
    from vecshift.connectors.qdrant.inspect import _get, _points

    points, _ = _points(client, collection, False, min(size, 500), size + 1)
    texts = [_get(p.get("payload") or {}, text_field) for p in points]
    return [t for t in texts if isinstance(t, str) and t.strip()]


def _pg_source(
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


def plan(
    job_file: Annotated[
        Path, typer.Argument(help="The job file.", show_default=True)
    ] = DEFAULT_JOB,
    probe: Annotated[
        bool,
        typer.Option(
            help=f"Embed {PROBE_DOCUMENTS} sample rows with the real model to measure its "
            "size, token counts, and speed."
        ),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before probing.")] = False,
    sample_size: Annotated[
        int, typer.Option("--sample", min=50, max=100_000, help="Rows to sample.")
    ] = 2000,
    output_json: Annotated[bool, typer.Option("--json", help="Print the plan as JSON.")] = False,
) -> None:
    """Check a migration job and estimate it, without changing anything."""
    run = prepare_qdrant if read_job(job_file).target.type == "qdrant" else prepare
    result = run(job_file, sample_size=sample_size, probe=probe, yes=yes).plan
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
    if not result.ok:
        raise typer.Exit(1)
