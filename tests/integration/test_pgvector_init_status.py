"""The guided init and the status command, against a real database."""

import json
import os
import stat
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from typer.testing import CliRunner

from tests.integration.test_pgvector_doctor import create_healthy
from vecshift.cli import app

runner = CliRunner()


@pytest.fixture
def no_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start without a connection string, and drop any the command sets for itself."""
    for name in ("VECSHIFT_DSN", "DATABASE_URL"):
        monkeypatch.setenv(name, "placeholder")  # recorded, so teardown removes it
        monkeypatch.delenv(name)


def test_guided_init(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_dsn: None,
) -> None:
    create_healthy(setup)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vecshift.cli_init.interactive", lambda: True)
    params = conninfo_to_dict(db_dsn)
    wrong = make_conninfo("", **{**params, "password": "not-the-password"})
    answers = [
        wrong,  # rejected, asked again
        db_dsn,
        "",  # the model that made the current vectors: skipped
        "5",  # hash/256
        "y",  # save the connection string to .env
        "y",  # check the plan
    ]
    result = runner.invoke(app, ["init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert "Couldn't connect" in result.output and "✔ Connected to" in result.output
    assert "Found public.documents.embedding" in result.output
    assert "Text comes from content" in result.output
    assert "vecshift plan · documents-reembed" in result.output, "the plan ran"

    secret = str(params["password"])
    assert "not-the-password" not in result.output
    assert db_dsn not in result.output and f"password={secret}" not in result.output
    job = (tmp_path / "vecshift.yaml").read_text()
    assert "password" not in job and "dsn_env: VECSHIFT_DSN" in job
    assert 'vector_column: "embedding"' in job and 'text_column: "content"' in job
    assert 'model: "hash/256"' in job

    env = tmp_path / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert secret in env.read_text()

    # Later commands find the connection string in .env, with nothing exported.
    os.environ.pop("VECSHIFT_DSN", None)
    later = runner.invoke(app, ["status"])
    assert later.exit_code == 0, later.output
    assert "Not started" in later.output


def test_guided_init_can_leave_secrets_unsaved(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_dsn: None,
) -> None:
    create_healthy(setup)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vecshift.cli_init.interactive", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("OPENAI_API_KEY")
    answers = [
        db_dsn,
        "hash/64",  # the old model, for eval
        "1",  # openai/text-embedding-3-small
        "",  # no budget
        "n",  # don't save the connection string
        "",  # no API key
        "n",  # don't run the plan
    ]
    result = runner.invoke(app, ["init"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert not (tmp_path / ".env").exists()
    assert "export VECSHIFT_DSN='…'" in result.output
    assert "Set OPENAI_API_KEY before running vecshift apply" in result.output
    job = (tmp_path / "vecshift.yaml").read_text()
    assert 'model: "hash/64"' in job and "# budget_usd" in job


def status(path: Path, dsn: str) -> dict:  # type: ignore[type-arg]
    result = runner.invoke(app, ["status", str(path), "--json"], env={"VECSHIFT_DSN": dsn})
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)  # type: ignore[no-any-return]


def test_status_follows_a_migration(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=50)
    path = tmp_path / "j.yaml"
    made = runner.invoke(
        app, ["init", "--table", "documents", "--model", "hash/16", "-o", str(path)]
    )
    assert made.exit_code == 0, made.output
    env = {"VECSHIFT_DSN": db_dsn}

    first = status(path, db_dsn)
    assert first["stage"] == "not_started" and first["rows"] is None
    assert "vecshift plan" in first["next"]

    runner.invoke(app, ["apply", str(path), "--yes", "--until", "50"], env=env)
    partial = status(path, db_dsn)
    assert partial["stage"] == "filling" and partial["trigger"] is True
    assert partial["rows"] == 50 and 0 < partial["missing"] <= 25 and partial["index"] is None
    assert "vecshift apply" in partial["next"]

    runner.invoke(app, ["apply", str(path), "--yes"], env=env)
    ready = status(path, db_dsn)
    assert ready["stage"] == "ready" and ready["missing"] == 0 and ready["filled"] == 1.0
    assert ready["index"] == "hnsw" and ready["runs"] == 2 and ready["rows_written"] == 50

    assert runner.invoke(app, ["cutover", str(path), "--yes"], env=env).exit_code == 0
    switched = status(path, db_dsn)
    assert switched["stage"] == "cut_over" and switched["missing"] == 0
    assert switched["last_event"]["event"] == "cutover" and "rollback" in switched["next"]

    assert runner.invoke(app, ["cleanup", str(path), "--yes"], env=env).exit_code == 0
    done = status(path, db_dsn)
    assert done["stage"] == "cleaned_up" and "Nothing left" in done["next"]

    text = runner.invoke(app, ["status", str(path)], env=env)
    assert "Stage    Done" in text.output


def test_status_changes_nothing(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=5)
    path = tmp_path / "j.yaml"
    runner.invoke(app, ["init", "--table", "documents", "--model", "hash/16", "-o", str(path)])
    before = setup.execute(
        "SELECT count(*) FROM pg_attribute WHERE attrelid = 'documents'::regclass"
    ).fetchone()
    status(path, db_dsn)
    after = setup.execute(
        "SELECT count(*) FROM pg_attribute WHERE attrelid = 'documents'::regclass"
    ).fetchone()
    assert before == after
    assert not (tmp_path / ".vecshift").exists(), "status writes no state"
