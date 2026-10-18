"""The REST API against a real PostgreSQL: the same answers as the CLI's JSON."""

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.integration.test_pgvector_doctor import create_healthy
from tests.integration.test_pgvector_plan import job_file
from vecshift.cli import app as cli
from vecshift.server import ServerSettings, create_app

runner = CliRunner()


def cli_json(*args: str) -> Any:
    result = runner.invoke(cli, [*args, "--json"])
    assert result.exit_code in (0, 1), result.output
    return json.loads(result.stdout)


def test_read_endpoints_match_the_cli(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_healthy(setup, rows=200)
    path = job_file(tmp_path, "hash/8")
    monkeypatch.setenv("VECSHIFT_DSN", db_dsn)
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(ServerSettings(jobs=tmp_path)), base_url="http://localhost")

    assert client.get("/api/v1/jobs/j/plan").json() == cli_json("plan", str(path))
    assert client.get("/api/v1/jobs/j/status?full=true").json() == cli_json("status", str(path))
    quick = client.get("/api/v1/jobs/j/status").json()
    assert quick["stage"] == "not_started" and quick["rows"] is None

    runner.invoke(cli, ["apply", str(path), "--yes"])
    full = client.get("/api/v1/jobs/j/status?full=true").json()
    assert full == cli_json("status", str(path)) and full["stage"] == "ready"
    assert client.get("/api/v1/jobs/j/status").json()["stage"] == "filling", "not counted"
    listed = client.get("/api/v1/jobs").json()["jobs"]
    assert [(j["name"], j["stage"]) for j in listed] == [("j", "filling")]

    body = {"store": "pgvector", "table": "public.documents", "column": "embedding"}
    doctor = client.post("/api/v1/doctor", json=body, headers={"X-Vecshift": "1"})
    assert doctor.status_code == 200
    expected = cli_json(
        "doctor", "--dsn", db_dsn, "--table", "public.documents", "--column", "embedding"
    )
    assert doctor.json() == expected
