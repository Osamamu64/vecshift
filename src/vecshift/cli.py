"""Command-line interface."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from vecshift import __version__
from vecshift.cli_style import STYLE as _STYLE
from vecshift.cli_style import warn_if_password_on_command_line
from vecshift.cli_style import wrap as _wrap
from vecshift.core.fingerprint import EmbeddingFingerprint
from vecshift.doctor import Report, Severity, run_checks

app = typer.Typer(
    name="vecshift",
    # Tracebacks must never print local variables: they can hold connection strings and keys.
    pretty_exceptions_show_locals=False,
    help="Safe, observable embedding migrations for any vector store.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"vecshift {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Safe, observable embedding migrations for any vector store."""


@app.command()
def fingerprint(
    provider: Annotated[str, typer.Option(help="Embedding provider, e.g. openai.")],
    model: Annotated[str, typer.Option(help="Model name, e.g. text-embedding-3-small.")],
    dimensions: Annotated[int, typer.Option(help="Output dimensions.")],
    version: Annotated[str | None, typer.Option(help="Model version, if pinned.")] = None,
    task: Annotated[str | None, typer.Option(help="Task type, e.g. retrieval_document.")] = None,
    prefix: Annotated[str | None, typer.Option(help="Text prefix, e.g. 'passage: '.")] = None,
    normalized: Annotated[
        bool,
        typer.Option("--normalized/--not-normalized", help="Whether vectors are L2-normalized."),
    ] = True,
) -> None:
    """Print the model tag that identifies an embedding configuration's vector space."""
    try:
        fp = EmbeddingFingerprint(
            provider=provider,
            model=model,
            dimensions=dimensions,
            version=version,
            task=task,
            prefix=prefix,
            normalized=normalized,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(fp.model_tag)


class FailOn(StrEnum):
    NEVER = "never"
    WARNING = "warning"
    ERROR = "error"


def _render(report: Report, connection: str) -> None:
    dims = f"({report.declared_dimensions})" if report.declared_dimensions else ""
    rows = f"~{report.estimated_rows:,}" if report.estimated_rows is not None else "unknown"
    typer.secho(f"vecshift doctor · {report.store} via {connection}", bold=True)
    typer.echo(f"Target  {report.target}  {report.vector_type}{dims}")
    typer.echo(f"Rows    {rows} (inspected {report.sample_rows:,}, {report.sample_method})")
    typer.echo()
    for finding in report.sorted_findings():
        label, color, _ = _STYLE[finding.severity]
        typer.secho(f"{label}  ", fg=color, bold=True, nl=False)
        typer.secho(finding.title, bold=True)
        typer.echo(_wrap(finding.detail))
        if finding.hint:
            typer.secho(_wrap(f"→ {finding.hint}"), dim=True)
    typer.echo()
    counts = [(report.count(sev), name) for sev, (_, _, name) in _STYLE.items()]
    summary = ", ".join(
        f"{n} {name}{'s' if n != 1 and name in ('error', 'warning') else ''}" for n, name in counts
    )
    typer.echo(f"Summary: {summary}")


@app.command()
def doctor(
    ctx: typer.Context,
    dsn: Annotated[
        str,
        typer.Option(
            envvar=["VECSHIFT_DSN", "DATABASE_URL"],
            help="PostgreSQL connection string. Prefer the VECSHIFT_DSN environment variable "
            "so the password stays out of your shell history.",
            show_default=False,
        ),
    ],
    table: Annotated[
        str | None, typer.Option(help="Table to inspect, as table or schema.table.")
    ] = None,
    column: Annotated[
        str | None, typer.Option(help="Vector column, if there's more than one.")
    ] = None,
    text_column: Annotated[
        str | None, typer.Option(help="Column holding the source text, if not detected.")
    ] = None,
    sample_size: Annotated[
        int, typer.Option(min=1, max=100_000, help="Maximum rows to inspect.")
    ] = 2000,
    timeout: Annotated[int, typer.Option(min=1, help="Statement timeout in seconds.")] = 60,
    output_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    html: Annotated[
        Path | None,
        typer.Option(
            "--html",
            dir_okay=False,
            help="Also write a self-contained HTML report to this file.",
            show_default=False,
        ),
    ] = None,
    fail_on: Annotated[
        FailOn, typer.Option(help="Exit with status 1 if any finding is this severe or worse.")
    ] = FailOn.NEVER,
) -> None:
    """Inspect a pgvector index and report problems. Read-only."""
    warn_if_password_on_command_line(ctx, dsn)
    try:
        from vecshift.connectors import pgvector
    except ImportError as exc:  # pragma: no cover - a broken install
        typer.secho(
            "The PostgreSQL driver is missing. Reinstall: pip install --force-reinstall vecshift",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(2) from exc

    try:
        settings = pgvector.prepare(dsn)
        conn = pgvector.connect(settings, statement_timeout_s=timeout)
    except pgvector.ConnectError as exc:
        typer.secho(f"Couldn't connect: {exc}", err=True, fg=typer.colors.RED)
        if exc.hint:
            typer.echo(f"→ {exc.hint}", err=True)
        raise typer.Exit(2) from exc

    try:
        profile = pgvector.inspect(
            conn, table=table, column=column, text_column=text_column, sample_size=sample_size
        )
    except pgvector.TargetSelectionError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        if exc.candidates:
            typer.echo("Vector columns found:", err=True)
            for c in exc.candidates:
                dims = f"({c.dimensions})" if c.dimensions else ""
                typer.echo(f"  {c.qualified}  {c.type}{dims}", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.Error):
            typer.secho(f"Inspection failed: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(2) from exc
        raise
    finally:
        conn.rollback()
        conn.close()

    report = run_checks(profile)
    if html is not None:
        from vecshift.doctor.html import render_html

        page = render_html(
            report,
            connection=settings.display,
            connection_kind=settings.description,
            version=__version__,
        )
        try:
            html.write_text(page, encoding="utf-8")
        except OSError as exc:
            typer.secho(f"Couldn't write {html}: {exc.strerror}", err=True, fg=typer.colors.RED)
            raise typer.Exit(2) from exc

    if output_json:
        typer.echo(json.dumps({"connection": settings.display, **report.to_dict()}, indent=2))
    else:
        _render(report, settings.description)
    if html is not None:
        typer.echo(f"HTML report written to {html}", err=output_json)

    if fail_on is not FailOn.NEVER and report.worst.rank >= Severity(fail_on.value).rank:
        raise typer.Exit(1)


from vecshift.cli_bench import bench  # noqa: E402

app.command()(bench)

from vecshift.cli_plan import init, plan  # noqa: E402

app.command()(init)
app.command()(plan)

from vecshift.cli_apply import apply  # noqa: E402
from vecshift.cli_cutover import cleanup, cutover, rollback  # noqa: E402

app.command()(apply)
app.command()(cutover)
app.command()(rollback)
app.command()(cleanup)

from vecshift.cli_eval import eval_  # noqa: E402

app.command(name="eval")(eval_)
