"""Terminal styling shared by the CLI commands."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from collections.abc import Sequence

import typer

from vecshift.doctor import Finding, Severity

STYLE = {
    Severity.ERROR: ("✖ ERROR  ", typer.colors.RED, "error"),
    Severity.WARNING: ("⚠ WARNING", typer.colors.YELLOW, "warning"),
    Severity.INFO: ("● INFO   ", typer.colors.BLUE, "info"),
    Severity.OK: ("✔ OK     ", typer.colors.GREEN, "ok"),
}


INDENT = " " * 11

# The logo's mark: points moving from the old embedding space (grey) into the new (blue).
MARK = (
    ("·", (169, 168, 161)),
    ("•", (131, 154, 181)),
    ("●", (91, 139, 199)),
    ("●", (42, 120, 214)),
)


def fancy() -> bool:
    """Whether to draw decoration: only for a person at a terminal that wants colour."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def banner(version: str, tagline: str | None = None) -> None:
    """The logo as text, for the start of interactive commands. Plain output stays plain."""
    if not fancy():
        return
    for glyph, rgb in MARK:
        typer.secho(glyph, fg=rgb, bold=True, nl=False)
        typer.echo(" ", nl=False)
    typer.secho(" vecshift", bold=True, nl=False)
    typer.secho(f" {version}", dim=True)
    if tagline:
        typer.secho(f"{' ' * 9}{tagline}", dim=True)
    typer.echo()


def wrap(text: str) -> str:
    width = min(100, shutil.get_terminal_size((100, 24)).columns)
    return textwrap.fill(text, width=width, initial_indent=INDENT, subsequent_indent=INDENT)


def finding_lines(findings: Sequence[Finding]) -> None:
    """Print findings, most severe first, in the doctor style."""
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
