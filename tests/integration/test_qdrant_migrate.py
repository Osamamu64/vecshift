"""plan, apply, and status for migrations into a real Qdrant: from a Qdrant collection and
from a pgvector table.

Needs VECSHIFT_TEST_QDRANT_URL (and VECSHIFT_TEST_PG_DSN, like every integration test).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from typer.testing import CliRunner

from vecshift.cli import app

URL = os.environ.get("VECSHIFT_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="VECSHIFT_TEST_QDRANT_URL is not set")
runner = CliRunner()


@pytest.fixture
def http() -> Iterator[httpx.Client]:
    assert URL
    with httpx.Client(base_url=URL) as client:
        yield client


@pytest.fixture
def names(http: httpx.Client) -> Iterator[dict[str, str]]:
    base = f"vecshift_mig_{uuid.uuid4().hex[:8]}"
    found = {"old": f"{base}_v1", "new": f"{base}_v2", "alias": base}
    try:
        yield found
    finally:
        http.post(
            "/collections/aliases",
            json={"actions": [{"delete_alias": {"alias_name": found["alias"]}}]},
        )
        for name in (found["old"], found["new"]):
            http.delete(f"/collections/{name}")


def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    assert URL
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECSHIFT_QDRANT_URL", URL)
    result = runner.invoke(app, ["--no-env-file", *args])
    return result


def last_json(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.startswith("{")]
    return dict(json.loads(lines[-1]))


def points(http: httpx.Client, name: str) -> dict[Any, dict[str, Any]]:
    body = {"limit": 1000, "with_payload": True, "with_vector": True}
    result = http.post(f"/collections/{name}/points/scroll", json=body).json()["result"]
    return {p["id"]: p for p in result["points"]}


def test_qdrant_to_qdrant(
    http: httpx.Client, names: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, alias = names["old"], names["new"], names["alias"]
    http.put(
        f"/collections/{old}", json={"vectors": {"dense": {"size": 4, "distance": "Dot"}}}
    ).raise_for_status()
    http.put(
        f"/collections/{old}/index?wait=true",
        json={"field_name": "lang", "field_schema": "keyword"},
    ).raise_for_status()
    seeded = [
        {
            "id": i,
            "vector": {"dense": [0.1, 0.2, 0.3, float(i)]},
            "payload": {"text": f"Article {i} about deliveries", "lang": "en"}
            if i % 20
            else {"lang": "en"},
        }
        for i in range(1, 101)
    ]
    http.put(f"/collections/{old}/points?wait=true", json={"points": seeded}).raise_for_status()
    http.post(
        "/collections/aliases",
        json={"actions": [{"create_alias": {"alias_name": alias, "collection_name": old}}]},
    ).raise_for_status()
    (tmp_path / "vecshift.yaml").write_text(
        f"name: q2q\nsource: {{type: qdrant, collection: {alias}}}\n"
        f"target: {{collection: {new}}}\nmodel: hash/16\n"
    )

    plan = cli(tmp_path, monkeypatch, "plan", "--json")
    assert plan.exit_code == 0, plan.output
    found = json.loads(plan.output)
    assert found["store"] == "qdrant" and found["target_column"] == f"{new}:dense"
    assert found["metric"] == "inner_product" and found["estimates"]["rows"] == 95
    assert {c["kind"] for c in found["changes"]} >= {"create_collection", "cutover"}

    applied = cli(tmp_path, monkeypatch, "apply", "--yes", "--json")
    assert applied.exit_code == 0, applied.output
    assert last_json(applied.output)["rows_written"] == 95
    info = http.get(f"/collections/{new}").json()["result"]
    assert info["config"]["params"]["vectors"] == {"dense": {"size": 16, "distance": "Dot"}}
    assert "lang" in info["payload_schema"]
    copied = points(http, new)
    assert len(copied) == 95 and len(copied[1]["vector"]["dense"]) == 16
    assert copied[1]["payload"] == {"text": "Article 1 about deliveries", "lang": "en"}
    assert len(points(http, old)) == 100, "the source is untouched"

    # The application keeps writing to the old collection meanwhile.
    http.post(
        f"/collections/{old}/points/payload?wait=true",
        json={"payload": {"text": "Article 2, rewritten"}, "points": [2]},
    ).raise_for_status()
    http.post(
        f"/collections/{old}/points/payload?wait=true",
        json={"payload": {"lang": "de"}, "points": [3]},
    ).raise_for_status()
    http.post(f"/collections/{old}/points/delete?wait=true", json={"points": [4]})
    status = cli(tmp_path, monkeypatch, "status", "--json")
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["missing"] == 1, "the rewritten point"

    again = cli(tmp_path, monkeypatch, "apply", "--yes", "--json")
    result = last_json(again.output)
    assert result["rows_written"] == 1 and result["payloads_copied"] == 1
    assert result["deleted"] == 1
    copied = points(http, new)
    assert copied[2]["payload"]["text"] == "Article 2, rewritten"
    assert copied[3]["payload"]["lang"] == "de" and 4 not in copied

    status = json.loads(cli(tmp_path, monkeypatch, "status", "--json").output)
    assert status["stage"] == "ready" and status["store"] == "qdrant"
    assert status["live"] == alias and status["previous"] == old and status["new"] == new
    assert status["rows"] == 94 and status["missing"] == 0

    evaluated = cli(
        tmp_path, monkeypatch, "eval", "--json", "-y", "--num-queries", "10", "--neighbours", "10"
    )
    assert evaluated.exit_code in {0, 1, 3}, evaluated.output
    report = json.loads(evaluated.output)
    assert report["new"]["column"] == new and report["old"]["column"] == old
    assert report["new"]["index_method"] == "qdrant-hnsw"

    ready = cli(tmp_path, monkeypatch, "cutover", "--check")
    assert ready.exit_code == 0 and "Ready to cut over" in ready.output
    http.post(
        f"/collections/{old}/points/payload?wait=true",
        json={"payload": {"text": "Article 5, last-minute edit"}, "points": [5]},
    ).raise_for_status()
    switched = cli(tmp_path, monkeypatch, "cutover", "--yes", "--json")
    assert switched.exit_code == 0, switched.output
    outcome = json.loads(switched.output)
    assert outcome["status"] == "cut_over" and outcome["caught_up"] == 1
    assert outcome["previous"] == old and outcome["late"] == 0

    def alias_target() -> str:
        found = http.get("/aliases").json()["result"]["aliases"]
        return str(next(a["collection_name"] for a in found if a["alias_name"] == alias))

    assert alias_target() == new
    assert points(http, new)[5]["payload"]["text"] == "Article 5, last-minute edit"
    status = json.loads(cli(tmp_path, monkeypatch, "status", "--json").output)
    assert status["stage"] == "cut_over" and status["previous"] == old
    again = cli(tmp_path, monkeypatch, "plan")
    assert again.exit_code == 1 and "Already cut over" in again.output

    # The application writes through the alias, now into the new collection.
    http.put(
        f"/collections/{alias}/points?wait=true",
        json={"points": [{"id": 500, "vector": {"dense": [0.0] * 16}, "payload": {"text": "n"}}]},
    ).raise_for_status()
    back = cli(tmp_path, monkeypatch, "rollback", "--yes", "--json")
    assert back.exit_code == 0, back.output
    assert json.loads(back.output)["changed"] == 1 and alias_target() == old

    assert cli(tmp_path, monkeypatch, "cutover", "--yes").exit_code == 0
    assert alias_target() == new
    refused = cli(tmp_path, monkeypatch, "cleanup")
    assert refused.exit_code == 2 and http.get(f"/collections/{old}").status_code == 200
    done = cli(tmp_path, monkeypatch, "cleanup", "--yes", "--json")
    assert done.exit_code == 0 and json.loads(done.output)["deleted"] == old
    assert http.get(f"/collections/{old}").status_code == 404
    status = json.loads(cli(tmp_path, monkeypatch, "status", "--json").output)
    assert status["stage"] == "cleaned_up"


def test_pgvector_to_qdrant(
    http: httpx.Client,
    names: dict[str, str],
    setup: psycopg.Connection,
    db_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    new, alias = names["new"], names["alias"]
    setup.execute(
        "CREATE TABLE notes (slug text PRIMARY KEY, body text, tags jsonb, "
        "embedding extensions.vector(3))"
    )
    setup.execute(
        "INSERT INTO notes SELECT 'note-' || g, CASE WHEN g = 7 THEN NULL ELSE "
        "'Note ' || g || ' on returns' END, '{\"a\": 1}', '[1,0,0]' "
        "FROM generate_series(1, 40) g"
    )
    monkeypatch.setenv("VECSHIFT_DSN", db_dsn)
    (tmp_path / "vecshift.yaml").write_text(
        "name: p2q\nsource: {table: notes}\n"
        f"target: {{type: qdrant, collection: {new}, alias: {alias}}}\nmodel: hash/16\n"
    )

    plan = cli(tmp_path, monkeypatch, "plan", "--json")
    assert plan.exit_code == 0, plan.output
    assert json.loads(plan.output)["estimates"]["rows"] == 39

    applied = cli(tmp_path, monkeypatch, "apply", "--yes", "--json")
    assert applied.exit_code == 0, applied.output
    copied = points(http, new)
    assert len(copied) == 39
    by_slug = {p["payload"]["slug"]: p for p in copied.values()}
    note = by_slug["note-1"]
    assert uuid.UUID(note["id"]).version == 5, "text keys become stable UUIDs"
    assert note["payload"] == {"slug": "note-1", "body": "Note 1 on returns", "tags": {"a": 1}}
    assert "embedding" not in note["payload"], "old vectors aren't copied"

    setup.execute("UPDATE notes SET body = 'Note 1, revised' WHERE slug = 'note-1'")
    setup.execute("UPDATE notes SET tags = '{\"a\": 2}' WHERE slug = 'note-2'")
    setup.execute("DELETE FROM notes WHERE slug = 'note-3'")
    assert json.loads(cli(tmp_path, monkeypatch, "status", "--json").output)["missing"] == 1

    again = last_json(cli(tmp_path, monkeypatch, "apply", "--yes", "--json").output)
    assert again["rows_written"] == 1 and again["payloads_copied"] == 1
    assert again["deleted"] == 1
    by_slug = {p["payload"]["slug"]: p for p in points(http, new).values()}
    assert by_slug["note-1"]["payload"]["body"] == "Note 1, revised"
    assert by_slug["note-2"]["payload"]["tags"] == {"a": 2} and "note-3" not in by_slug
    assert setup.execute("SELECT count(*) FROM notes").fetchone() == (39,)

    status = json.loads(cli(tmp_path, monkeypatch, "status", "--json").output)
    assert status["stage"] == "ready" and status["table"] == "notes"
    assert status["rows"] == 38 and status["missing"] == 0

    evaluated = cli(
        tmp_path, monkeypatch, "eval", "--json", "-y", "--num-queries", "10", "--neighbours", "10"
    )
    assert evaluated.exit_code in {0, 1, 3}, evaluated.output
    report = json.loads(evaluated.output)
    assert report["old"]["column"] == "notes.embedding" and report["new"]["column"] == new

    switched = cli(tmp_path, monkeypatch, "cutover", "--yes", "--json")
    assert switched.exit_code == 0, switched.output
    assert json.loads(switched.output)["previous"] is None
    aliases = http.get("/aliases").json()["result"]["aliases"]
    assert {"alias_name": alias, "collection_name": new} in aliases
    back = cli(tmp_path, monkeypatch, "rollback", "--yes")
    assert back.exit_code == 2 and "nothing to point it back at" in back.output
    cleaned = cli(tmp_path, monkeypatch, "cleanup", "--yes")
    assert cleaned.exit_code == 2 and "untouched" in cleaned.output
    assert setup.execute("SELECT count(*) FROM notes").fetchone() == (39,)


def test_guided_init_writes_a_job_plan_accepts(
    http: httpx.Client, names: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, alias = names["old"], names["new"], names["alias"]
    http.put(
        f"/collections/{old}", json={"vectors": {"size": 4, "distance": "Cosine"}}
    ).raise_for_status()
    seeded = [
        {"id": i, "vector": [1.0, 0.0, 0.0, float(i)], "payload": {"page_content": f"doc {i}"}}
        for i in range(1, 6)
    ]
    http.put(f"/collections/{old}/points?wait=true", json={"points": seeded}).raise_for_status()
    http.post(
        "/collections/aliases",
        json={"actions": [{"create_alias": {"alias_name": alias, "collection_name": old}}]},
    ).raise_for_status()
    monkeypatch.setattr("vecshift.cli_init.interactive", lambda: True)
    monkeypatch.setenv("VECSHIFT_DSN", "placeholder")
    monkeypatch.delenv("VECSHIFT_DSN")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECSHIFT_QDRANT_URL", URL or "")
    answers = ["hash/4", "5", "y"]  # the old model, hash/256, and run the plan
    result = runner.invoke(
        app, ["--no-env-file", "init", "--collection", alias], input="\n".join(answers) + "\n"
    )
    assert result.exit_code == 0, result.output
    assert f"Searches go through the alias {alias}" in result.output
    assert f"vecshift plan · {alias}-reembed" in result.output, "the plan ran"
    job = (tmp_path / "vecshift.yaml").read_text(encoding="utf-8")
    assert f'collection: "{alias}"' in job and 'text_field: "page_content"' in job
    planned = cli(tmp_path, monkeypatch, "plan", "--json")
    assert planned.exit_code == 0, planned.output
    assert json.loads(planned.stdout)["store"] == "qdrant"
    assert http.get(f"/collections/{new}").status_code == 404, "init and plan only read"
