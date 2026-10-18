"""Terminal styling shared by the CLI commands."""

from __future__ import annotations

import functools
import shutil
import textwrap
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeVar

import typer

from vecshift.doctor import Finding, Severity

P = ParamSpec("P")
R = TypeVar("R")

STYLE = {
    Severity.ERROR: ("✖ ERROR  ", typer.colors.RED, "error"),
    Severity.WARNING: ("⚠ WARNING", typer.colors.YELLOW, "warning"),
    Severity.INFO: ("● INFO   ", typer.colors.BLUE, "info"),
    Severity.OK: ("✔ OK     ", typer.colors.GREEN, "ok"),
}


INDENT = " " * 11


def wrap(text: str) -> str:
    width = min(100, shutil.get_terminal_size((100, 24)).columns)
    return textwrap.fill(text, width=width, initial_indent=INDENT, subsequent_indent=INDENT)


def finding_lines(findings: Sequence[Finding]) -> None:
    """Print findings, most severe first, in the doctor style."""
    from vecshift import ui

    if ui.fancy():
        ui.findings(findings)
        return
    for finding in sorted(findings, key=lambda f: -f.severity.rank):
        label, color, _ = STYLE[finding.severity]
        typer.secho(f"{label}  ", fg=color, bold=True, nl=False)
        typer.secho(finding.title.replace("`", ""), bold=True)
        typer.echo(wrap(finding.detail.replace("`", "")))
        if finding.hint:
            typer.secho(wrap(f"→ {finding.hint.replace('`', '')}"), dim=True)


def warn_if_password_on_command_line(ctx: typer.Context, dsn: str | None) -> None:
    """Passwords given as arguments are visible to other users through ps and shell history."""
    source = ctx.get_parameter_source("dsn")
    # Compared by name: typer may bundle its own copy of click's ParameterSource enum.
    if dsn and source is not None and source.name == "COMMANDLINE":
        from psycopg.conninfo import conninfo_to_dict

        try:
            has_password = bool(conninfo_to_dict(dsn).get("password"))
        except Exception:  # an invalid string is reported later, where it's parsed
            return
        if has_password:
            typer.secho(
                "Warning: the --dsn password is visible to other users of this machine and "
                "saved in shell history. Prefer the VECSHIFT_DSN environment variable.",
                err=True,
                fg=typer.colors.YELLOW,
            )


def fail(message: str, hint: str | None = None) -> typer.Exit:
    """Print a problem (and what to do about it) to stderr; raise the result to exit 2."""
    typer.secho(message, err=True, fg=typer.colors.RED)
    if hint:
        typer.echo(f"→ {hint}", err=True)
    return typer.Exit(2)


def reports_errors(command: Callable[P, R]) -> Callable[P, R]:
    """Show the service layer's errors as the CLI always has: exit 2 with the message and
    hint, or exit 1 when the person said no."""
    from vecshift.service import Declined, ServiceError

    @functools.wraps(command)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except Declined as exc:
            raise typer.Exit(1) from exc
        except ServiceError as exc:
            raise fail(exc.message, exc.hint) from exc

    return run
