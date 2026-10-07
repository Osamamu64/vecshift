"""Terminal styling shared by the CLI commands."""

from __future__ import annotations

import shutil
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
