import json
from pathlib import Path

import httpx
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from tests.integration.test_pgvector_doctor import create_healthy
from vecshift.cli import app

runner = CliRunner()


def job_file(tmp_path: Path, model: str = "ollama/nomic-embed-text", **extra: str) -> Path:
    result = runner.invoke(
        app,
        ["init", "--table", "public.documents", "--model", model, "-o", str(tmp_path / "j.yaml")],
    )
    assert result.exit_code == 0, result.output
    path = tmp_path / "j.yaml"
    text = path.read_text()
    for old, new in extra.items():
        text = text.replace(old, new)
    path.write_text(text)
    return path


def plan(path: Path, dsn: str, *args: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    result = runner.invoke(app, ["plan", str(path), "--json", *args], env={"VECSHIFT_DSN": dsn})
    assert result.exit_code in (0, 1), result.output
    return result.exit_code, json.loads(result.stdout)


def finding_ids(data: dict) -> set[str]:  # type: ignore[type-arg]
    return {f["id"] for f in data["findings"]}


def test_init_refuses_to_overwrite(tmp_path: Path) -> None:
    path = job_file(tmp_path)
    again = runner.invoke(app, ["init", "--table", "t", "--model", "ollama/m", "-o", str(path)])
    assert again.exit_code == 2 and "already exists" in again.output
    bad = runner.invoke(
        app, ["init", "--table", "t;drop", "--model", "ollama/m", "-o", str(tmp_path / "x.yaml")]
    )
    assert bad.exit_code == 2 and "plain identifier" in bad.output


def test_ready_plan(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=500)
    setup.execute("ANALYZE public.documents")
    code, data = plan(job_file(tmp_path), db_dsn)
    assert code == 0 and data["ok"], data["findings"]
    assert data["dimensions"] == 768 and data["estimates"]["rows"] == 500
    sql = [c["sql"] for c in data["changes"] if c["sql"]]
    assert sql[0] == (
        "ALTER TABLE public.documents ADD COLUMN embedding_v2 extensions.vector(768);"
    )
    assert "extensions.vector_cosine_ops" in sql[1]


def test_generated_sql_runs(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    """The DDL a plan prints must be valid against a real database."""
    create_healthy(setup, rows=50)
    _, data = plan(job_file(tmp_path), db_dsn)
    for change in data["changes"]:
        if change["sql"]:
            setup.execute(change["sql"])
    _, again = plan(tmp_path / "j.yaml", db_dsn)
    assert "plan.column_exists" in finding_ids(again)


def test_conflicting_column(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=50)
    setup.execute("ALTER TABLE public.documents ADD COLUMN embedding_v2 extensions.vector(3)")
    code, data = plan(job_file(tmp_path), db_dsn)
    assert code == 1 and "plan.column_conflict" in finding_ids(data)


def test_low_index_memory(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=2000)
    setup.execute("ANALYZE public.documents")
    dbname = conninfo_to_dict(db_dsn)["dbname"]
    setup.execute(f"ALTER DATABASE \"{dbname}\" SET maintenance_work_mem = '1MB'")
    _, data = plan(job_file(tmp_path), db_dsn)
    assert "plan.index_memory" in finding_ids(data)
    assert data["estimates"]["maintenance_work_mem"] == 1024 * 1024


def test_not_owner(reader: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=50)
    role = conninfo_to_dict(reader)["user"]
    setup.execute(f"GRANT SELECT ON public.documents TO {role}")
    code, data = plan(job_file(tmp_path), reader)
    assert code == 1 and "plan.not_owner" in finding_ids(data)


def test_probe(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_healthy(setup, rows=200)
    sent: list[int] = []

    async def post(self: httpx.AsyncClient, url: str, json: dict) -> httpx.Response:  # type: ignore[type-arg]
        sent.append(len(json["input"]))
        data = [{"index": i, "embedding": [0.1] * 5} for i in range(len(json["input"]))]
        return httpx.Response(
            200,
            json={"data": data, "usage": {"prompt_tokens": 100}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    path = job_file(tmp_path, "openai/text-embedding-3-small")

    refused = runner.invoke(app, ["plan", str(path), "--probe"], env={"VECSHIFT_DSN": db_dsn})
    assert refused.exit_code == 2 and "Pass --yes" in refused.output and not sent

    _, data = plan(path, db_dsn, "--probe", "--yes")
    assert sent == [16]
    assert data["dimensions"] == 5 and data["dimensions_source"] == "probe"
    assert data["estimates"]["tokens_method"] == "measured on 16 documents"


def test_missing_dsn_and_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECSHIFT_DSN", raising=False)
    missing = runner.invoke(app, ["plan", str(tmp_path / "none.yaml")])
    assert missing.exit_code == 2 and "vecshift init" in missing.output
    no_dsn = runner.invoke(app, ["plan", str(job_file(tmp_path))])
    assert no_dsn.exit_code == 2 and "VECSHIFT_DSN isn't set" in no_dsn.output
