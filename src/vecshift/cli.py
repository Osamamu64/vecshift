"""Command-line interface."""

from __future__ import annotations

from typing import Annotated

import typer

from vecshift import __version__
from vecshift.core.fingerprint import EmbeddingFingerprint

app = typer.Typer(
    name="vecshift",
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
