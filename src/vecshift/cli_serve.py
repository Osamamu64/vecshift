"""The ``vecshift serve`` command: the REST API, on this machine by default."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from vecshift.cli_style import fail


def serve(
    host: Annotated[
        str,
        typer.Option(help="Where to listen. Anything but localhost needs VECSHIFT_API_TOKEN set."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="The port.")] = 8765,
    jobs: Annotated[
        Path,
        typer.Option(file_okay=False, help="The folder of job files (*.yaml) to serve."),
    ] = Path("."),
) -> None:
    """Serve the REST API for the job files in a folder. Read docs/api.md first."""
    try:
        import uvicorn

        from vecshift.server import ServerSettings, create_app
        from vecshift.server.settings import TOKEN_ENV
    except ImportError as exc:
        raise fail(
            "The API server isn't installed.", 'Install it with: pip install "vecshift[server]"'
        ) from exc

    settings = ServerSettings(
        jobs=jobs, host=host, port=port, token=os.environ.get(TOKEN_ENV) or None
    )
    problem = settings.problem()
    if problem:
        raise fail(problem)
    shown = f"[{host}]" if ":" in host else host
    typer.echo(f"vecshift API on http://{shown}:{port}/api/v1 · jobs in {jobs.resolve()}")
    if settings.token:
        typer.echo(f"Requests need Authorization: Bearer <{TOKEN_ENV}>.")
    uvicorn.run(create_app(settings), host=host, port=port, server_header=False, log_level="info")
