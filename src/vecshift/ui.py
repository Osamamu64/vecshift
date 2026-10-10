"""Terminal presentation: the banner, step headers, and interactive prompts.

Decoration and arrow-key menus appear only for a person at a colour terminal. Anywhere
else (pipes, CI, tests, ``NO_COLOR``) the same calls fall back to plain text and numbered
prompts, so scripts and logs stay readable.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

# The logo's two colours: the old embedding space (grey) and the new one (blue).
OLD = (169, 168, 161)
NEW = (57, 135, 229)
ACCENT = "#3987e5"
MUTED = "#8a8983"

console = Console(highlight=False)

# "vecshift" in a block font, one row per line. Box-drawing characters are the shadow.
WORDMARK = (
    "██╗   ██╗███████╗ ██████╗███████╗██╗  ██╗██╗███████╗████████╗",
    "██║   ██║██╔════╝██╔════╝██╔════╝██║  ██║██║██╔════╝╚══██╔══╝",
    "██║   ██║█████╗  ██║     ███████╗███████║██║█████╗     ██║   ",
    "╚██╗ ██╔╝██╔══╝  ██║     ╚════██║██╔══██║██║██╔══╝     ██║   ",
    " ╚████╔╝ ███████╗╚██████╗███████║██║  ██║██║██║        ██║   ",
    "  ╚═══╝  ╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   ",
)
MARK = (("·", 0.0), ("•", 0.33), ("●", 0.66), ("●", 1.0))


def fancy() -> bool:
    """Whether a person is at a colour terminal."""
    return (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


def _blend(t: float) -> str:
    r, g, b = (round(o + (n - o) * t) for o, n in zip(OLD, NEW, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def _mark() -> Text:
    text = Text()
    for glyph, t in MARK:
        text.append(glyph + " ", style=f"bold {_blend(t)}")
    return text


def banner(version: str, tagline: str, details: Sequence[tuple[str, str]] = ()) -> None:
    """The wordmark, shading from the old grey to the new blue, with the version below."""
    if not fancy():
        return
    width = shutil.get_terminal_size((100, 24)).columns
    console.print()
    if width >= len(WORDMARK[0]) + 4:
        span = len(WORDMARK[0]) - 1
        for row in WORDMARK:
            line = Text("  ")
            for i, char in enumerate(row):
                if char == "█":
                    line.append(char, style=_blend(i / span))
                elif char != " ":
                    line.append(char, style=MUTED)
                else:
                    line.append(char)
            console.print(line)
        console.print()
    line = Text("  ")
    line.append_text(_mark())
    line.append(" vecshift", style="bold")
    line.append(f" {version}", style=ACCENT)
    line.append(f"  {tagline}", style="dim")
    console.print(line)
    for label, value in details:
        entry = Text(f"  {label:<10}", style="dim")
        entry.append(value)
        console.print(entry)
    console.print()


def step(number: int, total: int, title: str) -> None:
    """A section header: ``◆ Database ─────── 1/3``."""
    if fancy():
        console.print()
        console.rule(
            Text.assemble(
                ("◆ ", f"bold {ACCENT}"), (title, "bold"), (f"  {number}/{total}", "dim")
            ),
            align="left",
            style=MUTED,
        )
    else:
        typer.echo()
        typer.secho(title, bold=True)


def success(message: str) -> None:
    if fancy():
        console.print(Text.assemble(("✔ ", "bold green"), message))
    else:
        typer.secho(f"✔ {message}", fg=typer.colors.GREEN)


def warn(message: str) -> None:
    if fancy():
        console.print(Text.assemble(("! ", "bold yellow"), (message, "yellow")))
    else:
        typer.secho(message, fg=typer.colors.YELLOW)


def error(message: str, hint: str | None = None) -> None:
    if fancy():
        console.print(Text.assemble(("✖ ", "bold red"), (message, "red")))
        if hint:
            console.print(Text(f"  → {hint}", style="dim"))
    else:
        typer.secho(message, fg=typer.colors.RED)
        if hint:
            typer.echo(f"→ {hint}")


def note(message: str) -> None:
    if fancy():
        console.print(Text(message, style="dim"))
    else:
        typer.echo(message)


@contextmanager
def working(message: str) -> Iterator[None]:
    """A spinner while something slow happens, such as connecting."""
    if fancy():
        with console.status(Text(message, style="dim"), spinner="dots", spinner_style=ACCENT):
            yield
    else:
        yield


# --- prompts


def _style() -> Any:
    from questionary import Style

    return Style(
        [
            ("qmark", f"fg:{ACCENT} bold"),
            ("question", "bold"),
            ("answer", f"fg:{ACCENT} bold"),
            ("pointer", f"fg:{ACCENT} bold"),
            ("highlighted", f"fg:{ACCENT} bold"),
            ("selected", f"fg:{ACCENT}"),
            ("instruction", f"fg:{MUTED} italic"),
            ("text", ""),
            ("disabled", f"fg:{MUTED} italic"),
        ]
    )


def _answer(value: Any) -> Any:
    """questionary returns None when the person presses Ctrl-C."""
    if value is None:
        typer.echo("Cancelled.")
        raise typer.Exit(130)
    return value


def select(message: str, choices: Sequence[tuple[str, str]], default: int = 0) -> int:
    """Pick one of ``(label, description)`` choices; returns its index."""
    if fancy():
        import questionary

        options = [
            questionary.Choice(
                title=[("class:text", label), ("class:instruction", f"  {hint}" if hint else "")],
                value=i,
            )
            for i, (label, hint) in enumerate(choices)
        ]
        return int(
            _answer(
                questionary.select(
                    message,
                    choices=options,
                    default=options[default],
                    qmark="?",
                    pointer="❯",  # noqa: RUF001 - the menu pointer
                    instruction="(↑↓ to move, enter to pick)",
                    style=_style(),
                ).ask()
            )
        )
    typer.echo(message)
    for number, (label, hint) in enumerate(choices, start=1):
        typer.echo(f"  {number}) {label}" + (f"  {hint}" if hint else ""))
    while True:
        picked = str(typer.prompt("Choice", default=str(default + 1))).strip()
        if picked.isdigit() and 1 <= int(picked) <= len(choices):
            return int(picked) - 1
        typer.secho(f"Pick a number from 1 to {len(choices)}.", fg=typer.colors.YELLOW)


def text(
    message: str,
    default: str = "",
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """Free text. ``validate`` returns an error message, or None when the value is fine."""
    if fancy():
        import questionary

        def check(value: str) -> bool | str:
            problem = validate(value.strip()) if validate else None
            return problem or True

        return str(
            _answer(
                questionary.text(
                    message, default=default, validate=check, qmark="?", style=_style()
                ).ask()
            )
        ).strip()
    while True:
        value = str(typer.prompt(message, default=default, show_default=bool(default))).strip()
        problem = validate(value) if validate else None
        if not problem:
            return value
        typer.secho(problem, fg=typer.colors.YELLOW)


def secret(message: str, allow_empty: bool = False) -> str:
    """Hidden input, for passwords and keys. Nothing is echoed."""
    if fancy():
        import questionary

        def check(value: str) -> bool | str:
            return True if value.strip() or allow_empty else "Paste a value, or press Ctrl-C."

        return str(
            _answer(questionary.password(message, validate=check, qmark="?", style=_style()).ask())
        ).strip()
    default = "" if allow_empty else None
    value = typer.prompt(message, hide_input=True, default=default, show_default=False)
    return str(value).strip()


def confirm(message: str, default: bool = True) -> bool:
    if fancy():
        import questionary

        return bool(
            _answer(questionary.confirm(message, default=default, qmark="?", style=_style()).ask())
        )
    return typer.confirm(message, default=default)
