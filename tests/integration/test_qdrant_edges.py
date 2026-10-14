"""Edge cases of migrations into Qdrant that once lost or corrupted data, against real
services. Needs VECSHIFT_TEST_QDRANT_URL and VECSHIFT_TEST_PG_DSN."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest

from tests.integration.test_qdrant_migrate import URL, cli, last_json, points

pytestmark = pytest.mark.skipif(not URL, reason="VECSHIFT_TEST_QDRANT_URL is not set")


@pytest.fixture
def http() -> Iterator[httpx.Client]:
    assert URL
    made: list[str] = []
    with httpx.Client(base_url=URL) as client:
        client.made = made  # type: ignore[attr-defined]
        yield client
        aliases = client.get("/aliases").json()["result"]["aliases"]
        for a in aliases:
            if a["alias_name"].startswith("vsedge_"):
                client.post(
                    "/collections/aliases",
                    json={"actions": [{"delete_alias": {"alias_name": a["alias_name"]}}]},
                )
        for c in client.get("/collections").json()["result"]["collections"]:
            if c["name"].startswith("vsedge_"):
                client.delete(f"/collections/{c['name']}")


def name() -> str:
    return f"vsedge_{uuid.uuid4().hex[:8]}"


def collection(http: httpx.Client, coll: str, points_: list[dict[str, Any]], size: int = 4) -> None:
    http.put(f"/collections/{coll}", json={"vectors": {"size": size, "distance": "Cosine"}})
    http.put(f"/collections/{coll}/points?wait=true", json={"points": points_}).raise_for_status()


def alias(http: httpx.Client, alias_name: str, coll: str) -> None:
    http.post(
        "/collections/aliases",
        json={"actions": [{"create_alias": {"alias_name": alias_name, "collection_name": coll}}]},
    ).raise_for_status()


def alias_target(http: httpx.Client, alias_name: str) -> str | None:
    for a in http.get("/aliases").json()["result"]["aliases"]:
        if a["alias_name"] == alias_name:
            return str(a["collection_name"])
    return None


def docs(n: int) -> list[dict[str, Any]]:
    return [
        {"id": i, "vector": [1.0, 0.0, 0.0, float(i)], "payload": {"text": f"Doc {i} body"}}
        for i in range(1, n + 1)
    ]


@pytest.mark.parametrize(
    ("kind", "values"),
    [
        ("timestamptz", "'2024-01-01'::timestamptz + g * interval '1 hour'"),
        ("numeric(6,2)", "g + 0.5"),
    ],
)
def test_keys_whose_json_differs_from_their_text(
    http: httpx.Client,
    setup: psycopg.Connection,
    db_dsn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    values: str,
) -> None:
    """These keys once never matched their points, so every apply deleted everything."""
    new, live = name(), name()
    setup.execute(
        f"CREATE TABLE t (k {kind} PRIMARY KEY, body text, big numeric, "
        "embedding extensions.vector(3))"
    )
    setup.execute(
        f"INSERT INTO t SELECT {values}, 'Row ' || g || ' text', 1e20, '[1,0,0]' "
        "FROM generate_series(1, 10) g"
    )
    monkeypatch.setenv("VECSHIFT_DSN", db_dsn)
    (tmp_path / "vecshift.yaml").write_text(
        f"source: {{table: t}}\ntarget: {{type: qdrant, collection: {new}, alias: {live}}}\n"
        "model: hash/8\n"
    )
    out = cli(tmp_path, monkeypatch, "apply", "--yes", "--json")
    assert out.exit_code == 0, out.output
    first = last_json(out.output)
    assert first["rows_written"] == 10 and first["deleted"] == 0
    second = last_json(cli(tmp_path, monkeypatch, "apply", "--yes", "--json").output)
    assert second["rows_written"] == 0 and second["deleted"] == 0
    assert second["payloads_copied"] == 0, "big numbers come back as floats; still the same"
    assert len(points(http, new)) == 10
    setup.execute("DELETE FROM t WHERE body = 'Row 3 text'")
    third = last_json(cli(tmp_path, monkeypatch, "apply", "--yes", "--json").output)
    assert third["deleted"] == 1 and len(points(http, new)) == 9
    switched = cli(tmp_path, monkeypatch, "cutover", "--yes", "--json")
    assert switched.exit_code == 0, switched.output
    assert alias_target(http, live) == new and len(points(http, new)) == 9


def test_never_fills_a_collection_it_didnt_create(
    http: httpx.Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, other = name(), name()
    collection(http, src, docs(5))
    unrelated = [{"id": i, "vector": [0.0, 1.0] + [0.0] * 6, "payload": {"x": i}} for i in (1, 100)]
    collection(http, other, unrelated, size=8)
    (tmp_path / "vecshift.yaml").write_text(
        f"source: {{type: qdrant, collection: {src}}}\ntarget: {{collection: {other}}}\n"
        "model: hash/8\n"
    )
    plan = cli(tmp_path, monkeypatch, "plan")
    assert plan.exit_code == 1 and "this job didn't create it" in plan.output
    applied = cli(tmp_path, monkeypatch, "apply", "--yes")
    assert applied.exit_code == 1
    assert {p: v["payload"] for p, v in points(http, other).items()} == {
        1: {"x": 1},
        100: {"x": 100},
    }


def test_jobs_sharing_a_state_file_keep_their_own_history(
    http: httpx.Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two job files without a name share .vecshift/migration.state.json."""
    jobs = {}
    for job in ("a", "b"):
        old, new, live = name(), name(), name()
        collection(http, old, docs(4))
        alias(http, live, old)
        (tmp_path / f"{job}.yaml").write_text(
            f"source: {{type: qdrant, collection: {live}}}\ntarget: {{collection: {new}}}\n"
            "model: hash/8\n"
        )
        jobs[job] = (old, new, live)
        assert cli(tmp_path, monkeypatch, "apply", f"{job}.yaml", "--yes").exit_code == 0
        assert cli(tmp_path, monkeypatch, "cutover", f"{job}.yaml", "--yes").exit_code == 0
    (a_old, a_new, a_live), (b_old, _, _) = jobs["a"], jobs["b"]
    back = cli(tmp_path, monkeypatch, "rollback", "a.yaml", "--yes")
    assert back.exit_code == 0, back.output
    assert alias_target(http, a_live) == a_old
    assert cli(tmp_path, monkeypatch, "cutover", "a.yaml", "--yes").exit_code == 0
    done = cli(tmp_path, monkeypatch, "cleanup", "a.yaml", "--yes", "--json")
    assert json.loads(done.output)["deleted"] == a_old
    assert http.get(f"/collections/{b_old}").status_code == 200, "b's rollback target stays"
    assert alias_target(http, a_live) == a_new


def test_rollback_needs_the_cutover_record(
    http: httpx.Client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, live = name(), name(), name()
    collection(http, old, docs(4))
    alias(http, live, old)
    (tmp_path / "vecshift.yaml").write_text(
        f"source: {{type: qdrant, collection: {live}}}\ntarget: {{collection: {new}}}\n"
        "model: hash/8\n"
    )
    assert cli(tmp_path, monkeypatch, "apply", "--yes").exit_code == 0
    # --check only reads, even when a payload changed in the source.
    http.post(
        f"/collections/{old}/points/payload?wait=true",
        json={"payload": {"text": "Doc 1 body", "tag": "x"}, "points": [1]},
    ).raise_for_status()
    assert cli(tmp_path, monkeypatch, "cutover", "--check").exit_code == 0
    assert "tag" not in points(http, new)[1]["payload"]
    assert cli(tmp_path, monkeypatch, "cutover", "--yes").exit_code == 0
    assert points(http, new)[1]["payload"]["tag"] == "x", "cutover copies it"

    shutil.move(tmp_path / ".vecshift", tmp_path / "elsewhere")
    back = cli(tmp_path, monkeypatch, "rollback", "--yes")
    assert back.exit_code == 2 and "no record" in back.output
    assert alias_target(http, live) == new
    cleaned = cli(tmp_path, monkeypatch, "cleanup", "--yes")
    assert cleaned.exit_code == 2 and http.get(f"/collections/{old}").status_code == 200
