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
from typing import TYPE_CHECKING, Annotated

import typer

from vecshift.cli_style import finding_lines

if TYPE_CHECKING:
    from vecshift.connectors.pgvector import ConnectionSettings
    from vecshift.connectors.pgvector.target import TargetState
    from vecshift.doctor import IndexProfile
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec
    from vecshift.planning import Plan, ProbeResult

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
            "New column",
            _size(e.new_bytes),
            f"old column {_size(e.old_bytes)}" if e.old_bytes else "",
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
    marks = {"add_column": "+", "trigger": "+", "embed": "~", "index": "+", "cutover": "⇄"}
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


def load(
    job_file: Path, dsn_env: str | None = None
) -> tuple[JobSpec, ConnectionSettings, ModelSpec]:
    """The job, its connection settings (from the environment), and its model.

    ``dsn_env`` reads the connection string from another variable, such as a replica's.
    """
    from vecshift.connectors import pgvector
    from vecshift.embeddings import parse_spec
    from vecshift.jobs import JobError, load_job

    try:
        job = load_job(job_file)
    except JobError as exc:
        hint = (
            "Create one with: vecshift init --table ... --model ..."
            if not job_file.exists()
            else None
        )
        raise _fail(str(exc), hint) from exc
    variable = dsn_env or job.source.dsn_env
    dsn = os.environ.get(variable)
    if not dsn:
        raise _fail(
            f"{variable} isn't set.",
            f"Export the database connection string as {variable}, put it in a .env file "
            "here, or run vecshift init to set it up.",
        )
    try:
        settings = pgvector.prepare(dsn)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    return job, settings, parse_spec(job.model)


def prepare(
    job_file: Path, *, sample_size: int = 2000, probe: bool = False, yes: bool = False
) -> Prepared:
    """Load the job, inspect the database read-only, and build the plan."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.documents import sample_documents
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column
    from vecshift.embeddings import EmbeddingError
    from vecshift.planning import SampleStats, build_plan

    job, settings, spec = load(job_file)
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        source_column = resolve_source_column(
            conn, job.source.table, job.source.vector_column, job.target.column
        )
        profile = pgvector.inspect(
            conn,
            table=job.source.table,
            column=source_column,
            text_column=job.source.text_column,
            sample_size=sample_size,
        )
        target = inspect_target(conn, job.source.table, source_column, job.target.column)
        texts: list[str] = []
        if profile.text_field:
            rows, _ = sample_documents(
                conn,
                job.source.table,
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

    result = build_plan(job, profile, target, stats, measured)
    return Prepared(job, settings, spec, profile, target, result)


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
    result = prepare(job_file, sample_size=sample_size, probe=probe, yes=yes).plan
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
    if not result.ok:
        raise typer.Exit(1)
