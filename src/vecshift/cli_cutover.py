"""The ``vecshift cutover``, ``rollback``, and ``cleanup`` commands, for every store."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from vecshift import ui
from vecshift.cli_plan import DEFAULT_JOB
from vecshift.cli_style import finding_lines, reports_errors
from vecshift.service import ServiceError

if TYPE_CHECKING:
    from vecshift.service.switching import (
        CleanupNotice,
        CutoverOutcome,
        RollbackOutcome,
        SwitchNotice,
    )


def _confirm(message: str, yes: bool) -> bool:
    """Show what's about to change, and ask unless ``yes``. Whether to go ahead."""
    if yes:
        return True
    if ui.fancy():
        ui.console.print()
        ui.note("  " + " ".join(message.split()))
        return ui.confirm("Continue?", default=False)
    typer.echo(message, err=True)
    if not sys.stdin.isatty():
        raise ServiceError(
            "Not changing the database without confirmation.",
            "Pass --yes to proceed.",
            "confirm.required",
        )
    return typer.confirm("Continue?", err=True)


def _type_name(notice: CleanupNotice, yes: bool) -> str:
    """Ask for the name of what cleanup deletes, unless ``yes``."""
    if yes:
        return notice.name
    if ui.fancy():
        ui.warn(notice.message)
        return ui.text(f"Type {notice.name} to confirm")
    typer.echo(notice.message, err=True)
    if not sys.stdin.isatty():
        doing = "deleting" if notice.store == "qdrant" else "dropping"
        raise ServiceError(
            f"Not {doing} anything without confirmation.", "Pass --yes.", "confirm.required"
        )
    return str(typer.prompt(f"Type {notice.name} to confirm", err=True, default=""))


@reports_errors
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
    """Switch searches to the new vectors: the column name your app uses, or its alias."""
    from vecshift.service.switching import cutover as run

    def confirm(notice: SwitchNotice) -> bool:
        if not output_json:
            ui.title("cutover", notice.job)
            if notice.readiness is not None:
                finding_lines([f for f in notice.readiness.findings if f.severity.rank >= 1])
        return _confirm("\n" + notice.message, yes)

    outcome = run(job_file, check=check, allow_missing=allow_missing, confirm=confirm)
    ready = outcome.readiness
    if output_json:
        typer.echo(json.dumps(outcome.to_dict()))
    elif outcome.status in {"ready", "blocked"}:
        ui.title("cutover", outcome.job)
        finding_lines(ready.findings)
        if ready.ok:
            ui.verdict("Ready to cut over.")
            if ui.fancy():
                ui.next_step("vecshift cutover")
        else:
            ui.verdict("Not ready to cut over.", "error")
    elif outcome.store == "qdrant":
        _cut_over_qdrant(outcome)
    else:
        ui.verdict("Cut over.")
        ui.detail(
            f"{outcome.table}.{outcome.live} now holds {outcome.model} vectors; the old ones "
            f"are kept in {outcome.previous}."
        )
        if outcome.caught_up:
            ui.detail(f"Embedded {outcome.caught_up:,} late rows first.")
        ui.next_step(
            f"make sure your application embeds queries and new rows with "
            f"{outcome.model}.\nUndo with: vecshift rollback. When you're sure, drop "
            f"{outcome.previous} to free its space."
        )
    if outcome.status == "blocked":
        raise typer.Exit(1)


def _cut_over_qdrant(outcome: CutoverOutcome) -> None:
    ui.verdict("Cut over.")
    ui.detail(f"The alias {outcome.live} now points at {outcome.new}.")
    if outcome.caught_up:
        ui.detail(f"Embedded {outcome.caught_up:,} late points first.")
    if outcome.late:
        ui.detail(
            f"{outcome.late:,} points of {outcome.previous} differ from {outcome.new}: "
            "edits made in the moments before the switch, or written to the old "
            "collection by name. Check them."
        )
    undo = (
        f"Undo with: vecshift rollback. When you're sure, delete {outcome.previous} with "
        "vecshift cleanup."
        if outcome.previous
        else "The source is unchanged."
    )
    ui.next_step(
        f"make sure your application embeds queries and new points with {outcome.model}.\n{undo}"
    )


@reports_errors
def rollback(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before switching.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Undo a cutover: give the old vectors their column name (or alias) back."""
    from vecshift.service.switching import rollback as run

    outcome = run(job_file, confirm=lambda notice: _confirm(notice.message, yes))
    if output_json:
        typer.echo(json.dumps(outcome.to_dict()))
        return
    if ui.fancy():
        ui.title("rollback", outcome.job)
    ui.verdict("Rolled back.")
    _rolled_back(outcome)
    if ui.fancy():
        ui.console.print()


def _rolled_back(outcome: RollbackOutcome) -> None:
    if outcome.store == "qdrant":
        ui.detail(
            f"The alias {outcome.live} points at {outcome.previous} again; {outcome.new} is "
            "kept for another cutover."
        )
        if outcome.missing:
            ui.detail(
                f"{outcome.missing:,} points were added or edited in {outcome.new} after "
                f"cutover and are missing or different in {outcome.previous}. Your "
                "application needs to write them again, embedded with the old model."
            )
        return
    ui.detail(
        f"{outcome.table}.{outcome.live} holds the old vectors again; the new ones are back "
        f"in {outcome.new}, kept in sync for another cutover."
    )
    if outcome.missing:
        ui.detail(
            f"{outcome.missing:,} rows were added or edited after cutover and have no "
            "old-model vector. Your application needs to embed them with the old model."
        )


@reports_errors
def cleanup(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Don't ask for the column (or collection) name.")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """After cutover, drop the old vectors for good. This can't be undone."""
    from vecshift.service.switching import cleanup as run

    outcome = run(job_file, confirm=lambda notice: _type_name(notice, yes))
    if output_json:
        typer.echo(json.dumps(outcome.to_dict()))
        return
    if ui.fancy():
        ui.title("cleanup", outcome.job)
    done = "Deleted" if outcome.store == "qdrant" else "Dropped"
    ui.verdict(f"{done} {outcome.deleted}. The migration is complete.")
    if ui.fancy():
        ui.console.print()
