"""The ``vecshift status`` command: where a migration stands, and what to run next."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vecshift.cli_plan import DEFAULT_JOB
from vecshift.cli_style import reports_errors
from vecshift.service import Status
from vecshift.service import status as collect
from vecshift.service.progress import STAGES

JOURNEY = ("Plan", "Fill", "Index", "Cut over", "Clean up")


def _progress(status: Status) -> int:
    """How many steps of the journey are done."""
    if status.stage == "cleaned_up":
        return 5
    if status.stage == "cut_over":
        return 4
    if status.stage == "ready":
        return 3
    if status.stage in {"filling", "conflict"}:
        return 2 if status.missing == 0 else 1
    return 0


def _render_rich(status: Status, job_name: str) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from vecshift import ui

    colour = {"conflict": "red", "not_started": ui.MUTED}.get(status.stage, "green")
    if status.stage in {"filling"}:
        colour = ui.ACCENT

    done = _progress(status)
    journey = Text()
    for i, name in enumerate(JOURNEY):
        if i:
            journey.append(" ── ", style=ui.ACCENT if i <= done else ui.MUTED)
        if i < done:
            journey.append(f"✔ {name}", style="green")
        elif i == done and status.stage != "cleaned_up":
            journey.append(f"● {name}", style=f"bold {ui.ACCENT}")
        else:
            journey.append(f"○ {name}", style=ui.MUTED)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    qd = status.store == "qdrant"
    grid.add_row("Source" if qd else "Table", status.table)
    if qd:
        grid.add_row("Collection", _qdrant_where(status))
    elif status.stage == "cut_over":
        grid.add_row(
            "Columns", f"{status.live} holds the new vectors · old kept in {status.previous}"
        )
    elif status.stage == "cleaned_up":
        grid.add_row("Columns", f"{status.live} holds the new vectors")
    else:
        grid.add_row("Columns", Text.assemble(status.live, (" → ", "dim"), status.new))
    if status.rows is not None and status.missing is not None and status.filled is not None:
        width = 28
        full = round(status.filled * width)
        bar = Text.assemble(
            ("━" * full, ui.ACCENT if status.filled < 1 else "green"),
            ("━" * (width - full), ui.MUTED),
            (f"  {status.filled:.1%}", "bold"),
            (f"  {status.rows - status.missing:,} of {status.rows:,} {_unit(status)}", "dim"),
        )
        grid.add_row("Vectors", bar)
    if status.stage in {"filling", "ready", "cut_over", "conflict"}:
        if qd:
            grid.add_row("Index", "ready" if status.index else Text("indexing", style="yellow"))
        else:
            grid.add_row("Index", status.index or Text("none yet", style="yellow"))
            grid.add_row("Sync", "trigger installed" if status.trigger else "no trigger")
    if status.runs:
        failed = f" · {status.failed:,} rejected" if status.failed else ""
        grid.add_row(
            "Apply",
            f"{status.runs} run{'s' if status.runs != 1 else ''} · "
            f"{status.rows_written:,} rows · ${status.spent_usd:,.2f}{failed}",
        )
    if status.last_event:
        grid.add_row("Last", f"{status.last_event.get('event')} at {status.last_event.get('at')}")

    body = Table.grid()
    body.add_row(Text(f"● {STAGES[status.stage]}", style=f"bold {colour}"))
    body.add_row("")
    body.add_row(journey)
    body.add_row("")
    body.add_row(grid)
    ui.console.print()
    ui.console.print(
        Panel(
            body,
            title=Text.assemble(("vecshift status", "bold"), (f" · {job_name}", "dim")),
            title_align="left",
            border_style=ui.MUTED,
            padding=(1, 2),
            expand=False,
        )
    )
    ui.console.print(Text.assemble(("  ➜ Next  ", f"bold {ui.ACCENT}"), status.next))
    ui.console.print()


def _unit(status: Status) -> str:
    return "points" if status.source_store == "qdrant" else "rows"


def _qdrant_where(status: Status) -> str:
    alias = status.live
    if status.stage == "cut_over":
        return f"{status.new}, which {alias} now points at · old kept in {status.previous}"
    if status.stage == "cleaned_up":
        return f"{status.new}, which {alias} points at"
    if alias:
        return f"{status.new} · {alias} still points at the old vectors"
    return f"{status.new} · no alias set"


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
    qd = status.store == "qdrant"
    typer.echo(f"{'Source' if qd else 'Table':<9}{status.table}")
    if qd:
        typer.echo(f"Target   {_qdrant_where(status)}")
    elif status.stage == "cut_over":
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
        typer.echo(
            f"Vectors  {bar} {status.filled:.1%}  ({done:,} of {status.rows:,} {_unit(status)})"
        )
    if status.stage in {"filling", "ready", "cut_over", "conflict"} and qd:
        typer.echo(f"Index    {'ready' if status.index else 'indexing'}")
    elif status.stage in {"filling", "ready", "cut_over", "conflict"}:
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


@reports_errors
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
        from vecshift import ui

        name = load_job(job_file).name
        if ui.fancy():
            _render_rich(result, name)
        else:
            _render(result, name)
