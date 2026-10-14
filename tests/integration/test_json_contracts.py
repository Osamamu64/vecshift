"""The ``--json`` output of every command, pinned as a contract.

Scripts, and soon the REST API, read these shapes, so they mustn't change by accident. Each
test runs a whole migration against real services and records the *shape* of every JSON
output: keys and value types, not values. The shapes live in ``tests/contracts/``.

Adding a field is fine, but it fails here until the contract is updated, so the change is
deliberate and shows up in review. Removing or renaming one is a breaking change: note it
in the changelog. To update the contracts after a deliberate change:

    VECSHIFT_UPDATE_CONTRACTS=1 uv run pytest tests/integration/test_json_contracts.py

See docs/json.md for what the fields mean.
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

from tests.integration.test_pgvector_doctor import create_healthy
from vecshift.cli import app

CONTRACTS = Path(__file__).parent.parent / "contracts"
UPDATE = bool(os.environ.get("VECSHIFT_UPDATE_CONTRACTS"))
QDRANT = os.environ.get("VECSHIFT_TEST_QDRANT_URL")
runner = CliRunner()


# Objects keyed by data, such as dimensions or language pairs, rather than by field names:
# their entries are merged under one "<key>".
MAPS = {
    "facts.dimensions",
    "facts.models",
    "old.scores",
    "new.scores",
}


def shape(value: Any, path: str = "") -> Any:
    """Keys and value types, without the values. Lists show the merged shape of their items."""
    if isinstance(value, dict):
        if path in MAPS:
            merged: Any = None
            for key in value:
                merged = merge(merged, shape(value[key], f"{path}.<key>"))
            return {"<key>": merged} if value else {}
        return {key: shape(value[key], f"{path}.{key}".lstrip(".")) for key in sorted(value)}
    if isinstance(value, list):
        merged = None
        for item in value:
            merged = merge(merged, shape(item, path))
        return [] if merged is None else [merged]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    raise TypeError(f"not JSON: {value!r}")


def merge(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if isinstance(a, dict) and isinstance(b, dict):
        return {
            k: merge(a.get(k), b.get(k)) if k in a and k in b else a.get(k, b.get(k))
            for k in sorted(a.keys() | b.keys())
        }
    if isinstance(a, list) and isinstance(b, list):
        return [merge(a[0] if a else None, b[0] if b else None)] if a or b else []
    if a == b:
        return a
    kinds = set(str(a).split("|")) | set(str(b).split("|"))
    return "|".join(sorted(kinds))


class Recorder:
    """Collects each command's JSON shape, then checks them against the contract file."""

    def __init__(self, name: str) -> None:
        self.path = CONTRACTS / f"{name}.json"
        self.shapes: dict[str, Any] = {}

    def output(self, step: str, result: Any, codes: tuple[int, ...] = (0,)) -> dict[str, Any]:
        assert result.exit_code in codes, result.output
        data = json.loads(result.stdout)
        self.shapes[step] = shape(data)
        return dict(data)

    def lines(self, step: str, result: Any) -> list[dict[str, Any]]:
        """JSON lines (apply): one shape per event kind."""
        assert result.exit_code == 0, result.output
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        assert events[-1]["event"] == "result", "the result comes last"
        by_kind: dict[str, Any] = {}
        for event in events:
            by_kind[event["event"]] = merge(by_kind.get(event["event"]), shape(event))
        self.shapes[step] = by_kind
        return events

    def check(self) -> None:
        text = json.dumps(self.shapes, indent=2, ensure_ascii=False) + "\n"
        if UPDATE or not self.path.exists():
            self.path.parent.mkdir(exist_ok=True)
            self.path.write_text(text, encoding="utf-8")
            if not UPDATE:
                pytest.fail(f"Wrote a new contract, {self.path.name}: review and commit it.")
            return
        expected = json.loads(self.path.read_text(encoding="utf-8"))
        assert self.shapes == expected, (
            f"--json output no longer matches {self.path.name}. If the change is deliberate, "
            "rerun with VECSHIFT_UPDATE_CONTRACTS=1 and note it in the changelog."
        )


def long_text(i: int) -> str:
    """A document long enough for bench and eval to make queries from."""
    return (
        f"Article {i} explains how deliveries, returns, and refunds work for orders placed "
        f"in region {i % 7}. Customers can track every parcel online, change the address "
        f"before dispatch, and ask for a refund within {i % 30 + 1} days of delivery."
    )


LONG_SQL = (
    "'Article ' || id || ' explains how deliveries, returns, and refunds work for orders "
    "placed in region ' || id % 7 || '. Customers can track every parcel online, change the "
    "address before dispatch, and ask for a refund within ' || id % 30 + 1 || ' days.'"
)


def test_shape_merges_list_items_and_maps() -> None:
    assert shape({"b": [1, 2.5], "a": None, "c": [{"x": 1}, {"x": None, "y": "s"}]}) == {
        "a": "null",
        "b": ["number"],
        "c": [{"x": "null|number", "y": "string"}],
    }
    assert shape({"facts": {"dimensions": {"3": 1, "1536": 2}}}) == {
        "facts": {"dimensions": {"<key>": "number"}}
    }


# --- pgvector


def test_pgvector_contract(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=60)
    setup.execute("UPDATE public.documents SET content = " + LONG_SQL)
    setup.execute("ANALYZE public.documents")
    job = tmp_path / "vecshift.yaml"
    job.write_text(
        "name: contract\nsource:\n  table: public.documents\n  model: hash/3\n"
        "target:\n  column: embedding_v2\nmodel: hash/16\n",
        encoding="utf-8",
    )
    env = {"VECSHIFT_DSN": db_dsn}
    rec = Recorder("pgvector")

    def run(*args: str) -> Any:
        return runner.invoke(app, ["--no-env-file", *args], env=env)

    rec.output("doctor", run("doctor", "--table", "public.documents", "--json"))
    rec.output("bench", run("bench", "--table", "public.documents", "-m", "hash/8", "--json"))
    rec.output("plan", run("plan", str(job), "--json"))
    rec.output("status (not started)", run("status", str(job), "--json"))
    rec.lines("apply", run("apply", str(job), "--yes", "--json"))
    rec.output("status (ready)", run("status", str(job), "--json"))
    rec.output(
        "eval",
        run("eval", str(job), "--yes", "--json", "--num-queries", "20"),
        codes=(0, 1, 3),
    )
    rec.output("cutover --check", run("cutover", str(job), "--check", "--json"), codes=(0, 1))
    rec.output("cutover", run("cutover", str(job), "--yes", "--json"))
    rec.output("status (cut over)", run("status", str(job), "--json"))
    rec.output("rollback", run("rollback", str(job), "--yes", "--json"))
    assert run("cutover", str(job), "--yes").exit_code == 0
    rec.output("cleanup", run("cleanup", str(job), "--yes", "--json"))
    rec.output("status (cleaned up)", run("status", str(job), "--json"))
    rec.check()


# --- Qdrant

qdrant_only = pytest.mark.skipif(not QDRANT, reason="VECSHIFT_TEST_QDRANT_URL is not set")


@pytest.fixture
def http() -> Iterator[httpx.Client]:
    assert QDRANT
    with httpx.Client(base_url=QDRANT) as client:
        yield client


@pytest.fixture
def names(http: httpx.Client) -> Iterator[dict[str, str]]:
    base = f"vecshift_contract_{uuid.uuid4().hex[:8]}"
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


def qdrant_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    assert QDRANT
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VECSHIFT_QDRANT_URL", QDRANT)
    return runner.invoke(app, ["--no-env-file", *args])


@qdrant_only
def test_qdrant_contract(
    http: httpx.Client, names: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, alias = names["old"], names["new"], names["alias"]
    http.put(
        f"/collections/{old}", json={"vectors": {"size": 4, "distance": "Cosine"}}
    ).raise_for_status()
    seeded = [
        {
            "id": i,
            "vector": [1.0, 0.1 * i, 0.0, 0.5],
            "payload": {"page_content": long_text(i)},
        }
        for i in range(1, 61)
    ]
    http.put(f"/collections/{old}/points?wait=true", json={"points": seeded}).raise_for_status()
    http.post(
        "/collections/aliases",
        json={"actions": [{"create_alias": {"alias_name": alias, "collection_name": old}}]},
    ).raise_for_status()
    (tmp_path / "vecshift.yaml").write_text(
        f"name: contract\nsource: {{type: qdrant, collection: {alias}}}\nmodel: hash/16\n",
        encoding="utf-8",
    )
    rec = Recorder("qdrant")

    def run(*args: str) -> Any:
        return qdrant_cli(tmp_path, monkeypatch, *args)

    rec.output("doctor", run("doctor", "--qdrant", QDRANT or "", "--collection", alias, "--json"))
    rec.output(
        "bench",
        run("bench", "--qdrant", QDRANT or "", "--collection", alias, "-m", "hash/8", "--json"),
    )
    rec.output("plan", run("plan", "--json"))
    rec.output("status (not started)", run("status", "--json"))
    rec.lines("apply", run("apply", "--yes", "--json"))
    rec.output("status (ready)", run("status", "--json"))
    rec.output(
        "eval",
        run("eval", "--yes", "--json", "--num-queries", "10", "--neighbours", "10"),
        codes=(0, 1, 3),
    )
    rec.output("cutover --check", run("cutover", "--check", "--json"), codes=(0, 1))
    rec.output("cutover", run("cutover", "--yes", "--json"))
    rec.output("status (cut over)", run("status", "--json"))
    rec.output("rollback", run("rollback", "--yes", "--json"))
    assert run("cutover", "--yes").exit_code == 0
    rec.output("cleanup", run("cleanup", "--yes", "--json"))
    rec.output("status (cleaned up)", run("status", "--json"))
    rec.check()
    assert http.get(f"/collections/{new}").status_code == 200


@qdrant_only
def test_pgvector_to_qdrant_contract(
    http: httpx.Client,
    names: dict[str, str],
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_healthy(setup, rows=40)
    alias = names["alias"]
    (tmp_path / "vecshift.yaml").write_text(
        "name: contract\nsource:\n  table: public.documents\n"
        f"target:\n  type: qdrant\n  collection: {names['new']}\n  alias: {alias}\n"
        "model: hash/16\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VECSHIFT_DSN", db_dsn)
    rec = Recorder("pgvector-to-qdrant")

    def run(*args: str) -> Any:
        return qdrant_cli(tmp_path, monkeypatch, *args)

    rec.output("plan", run("plan", "--json"))
    rec.lines("apply", run("apply", "--yes", "--json"))
    rec.output("status (ready)", run("status", "--json"))
    rec.output("cutover", run("cutover", "--yes", "--json"))
    rec.output("status (cut over)", run("status", "--json"))
    rec.check()
