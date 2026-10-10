"""Command-line interface."""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from vecshift import __version__
from vecshift.cli_style import STYLE as _STYLE
from vecshift.cli_style import warn_if_password_on_command_line
from vecshift.cli_style import wrap as _wrap
from vecshift.core.fingerprint import EmbeddingFingerprint
from vecshift.doctor import Report, Severity, run_checks
from vecshift.envfile import DEFAULT as DEFAULT_ENV_FILE

TAGLINE = "Safe, observable embedding migrations for any vector store."

app = typer.Typer(
    name="vecshift",
    # Tracebacks must never print local variables: they can hold connection strings and keys.
    pretty_exceptions_show_locals=False,
    help=TAGLINE,
    invoke_without_command=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"vecshift {__version__}")
        raise typer.Exit()


def _load_env_file(path: Path | None) -> None:
    from vecshift import envfile

    if path is None:
        return
    try:
        envfile.load(path)
    except envfile.EnvFileError as exc:
        typer.secho(f"Couldn't read {path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc
    if envfile.readable_by_others(path):
        typer.secho(
            f"Warning: other users of this machine can read {path}. Run: chmod 600 {path}",
            err=True,
            fg=typer.colors.YELLOW,
        )


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            dir_okay=False,
            help="Read settings such as VECSHIFT_DSN from this file, if it exists. Variables "
            "already set in the environment take precedence.",
        ),
    ] = DEFAULT_ENV_FILE,
    no_env_file: Annotated[
        bool, typer.Option("--no-env-file", help="Don't read a .env file.")
    ] = False,
) -> None:
    """Safe, observable embedding migrations for any vector store."""
    _load_env_file(None if no_env_file else env_file)
    if ctx.invoked_subcommand is None:
        from vecshift import ui

        ui.banner(__version__, TAGLINE)
        typer.echo(ctx.get_help())


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


def _render_fancy(report: Report, connection: str) -> None:
    from rich.text import Text

    from vecshift import ui

    dims = f"({report.declared_dimensions})" if report.declared_dimensions else ""
    rows = f"~{report.estimated_rows:,}" if report.estimated_rows is not None else "unknown"
    ui.title("doctor", f"{report.store} via {connection}")
    ui.kv(
        [
            ("Target", Text.assemble(report.target, (f"  {report.vector_type}{dims}", "dim")), ""),
            ("Rows", rows, f"inspected {report.sample_rows:,}, {report.sample_method}"),
        ]
    )
    ui.section("Findings")
    ui.findings(report.sorted_findings())
    summary = Text("  ")
    for severity, (_, _, name) in _STYLE.items():
        n = report.count(severity)
        icon, colour, _ = ui.SEVERITY_STYLE[severity.value]
        plural = "s" if n != 1 and name in ("error", "warning") else ""
        summary.append(f"{icon} {n} {name}{plural}   ", style=colour if n else "dim")
    ui.console.print()
    ui.console.print(summary)
    ui.console.print()


def _render(report: Report, connection: str) -> None:
    from vecshift import ui

    if ui.fancy():
        _render_fancy(report, connection)
        return
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
        str | None,
        typer.Option(
            envvar=["VECSHIFT_DSN", "DATABASE_URL"],
            help="PostgreSQL connection string. Prefer the VECSHIFT_DSN environment variable "
            "so the password stays out of your shell history.",
            show_default=False,
        ),
    ] = None,
    table: Annotated[
        str | None, typer.Option(help="Table to inspect, as table or schema.table.")
    ] = None,
    column: Annotated[
        str | None, typer.Option(help="Vector column, if there's more than one.")
    ] = None,
    qdrant_url: Annotated[
        str | None,
        typer.Option(
            "--qdrant",
            envvar="VECSHIFT_QDRANT_URL",
            help="Inspect Qdrant at this URL instead, e.g. http://localhost:6333. The API key "
            "is read from QDRANT_API_KEY.",
            show_default=False,
        ),
    ] = None,
    collection: Annotated[
        str | None, typer.Option(help="Qdrant collection or alias to inspect.")
    ] = None,
    vector: Annotated[
        str | None, typer.Option(help="Qdrant named vector, if the collection has several.")
    ] = None,
    text_column: Annotated[
        str | None,
        typer.Option(
            help="Column (or Qdrant payload field, such as metadata.text) holding the source "
            "text, if not detected."
        ),
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
    """Inspect a pgvector or Qdrant index and report problems. Read-only."""
    warn_if_password_on_command_line(ctx, dsn)
    source = ctx.get_parameter_source("qdrant_url")
    explicit_qdrant = source is not None and source.name == "COMMANDLINE"
    if collection or vector or explicit_qdrant or (qdrant_url and not dsn):
        if not qdrant_url:
            _doctor_fail("Missing option '--qdrant' (or VECSHIFT_QDRANT_URL).")
        profile, display, kind = _qdrant_profile(
            str(qdrant_url), collection, vector, text_column, sample_size, timeout
        )
    elif dsn:
        profile, display, kind = _pg_profile(dsn, table, column, text_column, sample_size, timeout)
    else:
        _doctor_fail(
            "Missing option '--dsn' (or VECSHIFT_DSN).",
            "Pass a PostgreSQL connection string, or --qdrant URL for Qdrant.",
        )

    report = run_checks(profile)
    if html is not None:
        from vecshift.doctor.html import render_html

        page = render_html(
            report,
            connection=display,
            connection_kind=kind,
            version=__version__,
        )
        try:
            html.write_text(page, encoding="utf-8")
        except OSError as exc:
            typer.secho(f"Couldn't write {html}: {exc.strerror}", err=True, fg=typer.colors.RED)
            raise typer.Exit(2) from exc

    if output_json:
        typer.echo(json.dumps({"connection": display, **report.to_dict()}, indent=2))
    else:
        _render(report, kind)
    if html is not None:
        typer.echo(f"HTML report written to {html}", err=output_json)

    if fail_on is not FailOn.NEVER and report.worst.rank >= Severity(fail_on.value).rank:
        raise typer.Exit(1)


def _doctor_fail(message: str, hint: str | None = None) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    if hint:
        typer.echo(f"→ {hint}", err=True)
    raise typer.Exit(2)


def _candidates(candidates: Sequence[Any], heading: str) -> None:
    if candidates:
        typer.echo(heading, err=True)
        for c in candidates:
            dims = f"({c.dimensions})" if c.dimensions else ""
            typer.echo(f"  {c.qualified}  {c.type}{dims}", err=True)


def _pg_profile(
    dsn: str,
    table: str | None,
    column: str | None,
    text_column: str | None,
    sample_size: int,
    timeout: int,
) -> tuple[Any, str, str]:
    """Profile a pgvector column: (profile, connection shown, connection kind)."""
    try:
        from vecshift.connectors import pgvector
    except ImportError:  # pragma: no cover - a broken install
        _doctor_fail(
            "The PostgreSQL driver is missing. Reinstall: pip install --force-reinstall vecshift"
        )

    try:
        settings = pgvector.prepare(dsn)
        conn = pgvector.connect(settings, statement_timeout_s=timeout)
    except pgvector.ConnectError as exc:
        _doctor_fail(f"Couldn't connect: {exc}", exc.hint)

    try:
        profile = pgvector.inspect(
            conn, table=table, column=column, text_column=text_column, sample_size=sample_size
        )
    except pgvector.TargetSelectionError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        _candidates(exc.candidates or [], "Vector columns found:")
        raise typer.Exit(2) from exc
    except Exception as exc:
        import psycopg

        if isinstance(exc, psycopg.Error):
            _doctor_fail(f"Inspection failed: {exc}")
        raise
    finally:
        conn.rollback()
        conn.close()
    return profile, settings.display, settings.description


def _qdrant_profile(
    url: str,
    collection: str | None,
    vector: str | None,
    text_field: str | None,
    sample_size: int,
    timeout: int,
) -> tuple[Any, str, str]:
    """Profile a Qdrant vector: (profile, URL shown, connection kind)."""
    from vecshift.connectors import qdrant

    try:
        settings = qdrant.prepare(url)
        with qdrant.QdrantClient(settings, timeout=float(timeout)) as client:
            profile = qdrant.inspect(client, collection, vector, text_field, sample_size)
    except qdrant.SelectionError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        _candidates(exc.candidates, "Vectors found:")
        raise typer.Exit(2) from exc
    except qdrant.QdrantError as exc:
        _doctor_fail(str(exc), exc.hint)
    return profile, settings.display, settings.description


from vecshift.cli_bench import bench  # noqa: E402

app.command()(bench)

from vecshift.cli_init import init  # noqa: E402
from vecshift.cli_plan import plan  # noqa: E402

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

from vecshift.cli_status import status  # noqa: E402

app.command()(status)
