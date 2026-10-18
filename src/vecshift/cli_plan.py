"""The ``vecshift plan`` command."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from vecshift.cli_style import finding_lines, reports_errors
from vecshift.service import Consent, Sending, ServiceError, prepare_plan
from vecshift.service.plan import PROBE_DOCUMENTS

if TYPE_CHECKING:
    from vecshift.planning import Plan

DEFAULT_JOB = Path("vecshift.yaml")


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


def _consent(yes: bool) -> Consent:
    """Ask at the terminal before the probe sends sample rows to a remote model."""

    def consent(sending: Sending) -> bool:
        if yes:
            return True
        for name, url in sending.remote:
            typer.echo(
                f"The probe sends {sending.count} sample rows to {url} to measure {name}.",
                err=True,
            )
        if not sys.stdin.isatty():
            raise ServiceError(
                "Not sending data without confirmation.",
                "Pass --yes to proceed.",
                "consent.required",
            )
        return typer.confirm("Continue?", err=True)

    return consent


@reports_errors
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
    result = prepare_plan(
        job_file, sample_size=sample_size, probe=probe, consent=_consent(yes)
    ).plan
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
    if not result.ok:
        raise typer.Exit(1)
