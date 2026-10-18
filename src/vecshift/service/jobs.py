"""Loading a job, and the connection settings it names in the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from vecshift.service.errors import ServiceError

if TYPE_CHECKING:
    from vecshift.connectors.pgvector import ConnectionSettings
    from vecshift.connectors.qdrant import QdrantSettings
    from vecshift.embeddings import ModelSpec
    from vecshift.jobs import JobSpec


def read_job(job_file: Path) -> JobSpec:
    from vecshift.jobs import JobError, load_job

    try:
        return load_job(job_file)
    except JobError as exc:
        if not job_file.exists():
            raise ServiceError(
                str(exc), "Create one with: vecshift init --table ... --model ...", "job.missing"
            ) from exc
        raise ServiceError(str(exc), code="job.invalid") from exc


def into_qdrant(job_file: Path) -> bool:
    """Whether the job's new vectors go to Qdrant."""
    return read_job(job_file).target.type == "qdrant"


def pg_settings(variable: str) -> ConnectionSettings:
    """Connection settings from the variable holding the connection string."""
    from vecshift.connectors import pgvector

    dsn = os.environ.get(variable)
    if not dsn:
        raise ServiceError(
            f"{variable} isn't set.",
            f"Export the database connection string as {variable}, put it in a .env file "
            "here, or run vecshift init to set it up.",
            "env.missing",
        )
    try:
        return pgvector.prepare(dsn)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc


def qdrant_settings(url_env: str, key_env: str) -> QdrantSettings:
    from vecshift.connectors import qdrant

    url = os.environ.get(url_env)
    if not url:
        raise ServiceError(
            f"{url_env} isn't set.",
            f"Export the Qdrant URL as {url_env} (and the API key as {key_env}, if the "
            "server needs one), or put them in a .env file here.",
            "env.missing",
        )
    try:
        return qdrant.prepare(url, key_env)
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint, "connect.failed") from exc


def load(
    job_file: Path, dsn_env: str | None = None, *, command: str = "This command"
) -> tuple[JobSpec, ConnectionSettings, ModelSpec]:
    """A pgvector job, its connection settings (from the environment), and its model.

    ``dsn_env`` reads the connection string from another variable, such as a replica's.
    """
    from vecshift.embeddings import parse_spec

    job = read_job(job_file)
    if job.target.type == "qdrant":
        raise ServiceError(
            f"{command} reads a pgvector target, and this job's target is Qdrant.",
            code="job.wrong_store",
        )
    settings = pg_settings(dsn_env or job.pg_source.dsn_env)
    return job, settings, parse_spec(job.model)
