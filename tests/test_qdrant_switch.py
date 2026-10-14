"""Cutover for Qdrant (moving the alias) and eval's Qdrant searches, against a fake Qdrant."""

from pathlib import Path
from typing import Any

import pytest

from tests.test_qdrant import FakeQdrant, client
from tests.test_qdrant_migrate import run, setup
from vecshift.connectors.qdrant.search import PgSide, QdrantRows, QdrantSide, SplitSearcher
from vecshift.connectors.qdrant.switch import QdrantSwitch
from vecshift.connectors.qdrant.writer import QdrantWriter


def ids(readiness: Any) -> set[str]:
    return {f.id for f in readiness.findings}


def test_cutover_readiness_and_switch(tmp_path: Path) -> None:
    fake, writer = setup()
    fake.aliases = [("docs", "docs_v1")]
    switch = QdrantSwitch(writer.client, writer, "docs", "docs_v1")
    assert switch.stage() == "not_applied" and "cutover.not_applied" in ids(switch.check())

    run(writer, tmp_path)
    ready = switch.check()
    assert switch.stage() == "applied" and ready.ok and ready.pending == 0
    assert "cutover.model_switch" in ids(ready)

    fake.points["docs_v1"].append(
        {"id": 77, "vector": [1.0, 0.0], "payload": {"page_content": "x"}}
    )
    ready = switch.check()
    assert ready.pending == 1 and "cutover.pending" in ids(ready)

    switch.point("docs_v2")
    assert fake.aliases == [("docs", "docs_v2")]
    assert switch.stage() == "cut_over" and "cutover.done" in ids(switch.check())
    switch.point("docs_v1")
    assert fake.aliases == [("docs", "docs_v1")], "rollback moves it back"


def test_cutover_problems(tmp_path: Path) -> None:
    fake, writer = setup()
    run(writer, tmp_path)
    assert "cutover.no_alias" in ids(QdrantSwitch(writer.client, writer, None, "docs_v1").check())
    taken = QdrantSwitch(writer.client, writer, "docs_v1", None)
    assert "cutover.alias_conflict" in ids(taken.check())
    fake.aliases = [("docs", "elsewhere")]
    moved = QdrantSwitch(writer.client, writer, "docs", "docs_v1")
    assert "cutover.alias_moved" in ids(moved.check())
    fake.collections["docs_v2"]["status"] = "yellow"
    assert "cutover.indexing" in ids(moved.check())


def test_new_alias_for_a_pgvector_source(tmp_path: Path) -> None:
    fake, writer = setup()
    run(writer, tmp_path)
    switch = QdrantSwitch(writer.client, writer, "articles", None)
    assert switch.check().ok
    switch.point("docs_v2")
    assert fake.aliases == [("articles", "docs_v2")]
    switch.remove_alias()
    assert fake.aliases == []


def test_lock_nests_within_one_writer(tmp_path: Path) -> None:
    _, writer = setup()
    lock = tmp_path / "job.lock"
    held = QdrantWriter(writer.client, writer.source, writer.layout, lock)
    held.acquire()
    held.acquire()  # cutover holds it while its catch-up runs apply
    held.release()
    other = QdrantWriter(writer.client, writer.source, writer.layout, lock)
    with pytest.raises(Exception, match="already"):
        other.acquire()
    held.release()
    other.acquire()
    other.release()


# --- eval


def eval_fake() -> FakeQdrant:
    fake = FakeQdrant()
    points = [
        {
            "id": i,
            "vector": {"dense": [1.0, float(i)]},
            "payload": {"text": f"text {i}", "slug": f"s{i}", "metadata": {"source": f"d{i % 3}"}},
        }
        for i in range(1, 11)
    ]
    fake.add("c", {"dense": {"size": 2, "distance": "Dot"}}, points)
    return fake


def test_qdrant_side_searches() -> None:
    fake = eval_fake()
    side = QdrantSide(client(fake), "c", "dense", "Dot", group_field="metadata.source")
    hits = side.search([0.0, 1.0], 3)
    assert [k for k, _ in hits] == ["10", "9", "8"] and hits[0][1] == -10.0, "distance = -score"
    assert side.exact([0.0, 1.0], 2) == hits[:2]
    assert fake.requests[-1].content and b'"exact":true' in fake.requests[-1].content
    side.search([0.0, 1.0], 3, setting=64)
    assert b'"hnsw_ef":64' in fake.requests[-1].content
    assert side.index() == ("qdrant-hnsw", 100, [32, 64, 100, 128, 256])
    group, near = side.neighbours("4", 2)
    assert group == "d1" and [k for k, _ in near] == ["10", "9"] and "4" not in dict(near)
    assert side.vectors(["2", "3"]) == [[1.0, 2.0], [1.0, 3.0]]

    rows = QdrantRows(side, "text")
    assert ("3", "text 3") in rows.sample("text", 20)
    assert rows.texts("text", ["5"]) == {"5": "text 5"}


def test_keys_from_payload_and_distances() -> None:
    fake = eval_fake()
    side = QdrantSide(
        client(fake),
        "c",
        "dense",
        "Cosine",
        key_field="slug",
        to_points=lambda ks: [int(k[1:]) for k in ks],
    )
    hits = side.search([0.0, 1.0], 1)
    assert hits[0][0] == "s10" and hits[0][1] == pytest.approx(1 - 10.0)
    assert side.vectors(["s2"]) == [[1.0, 2.0]]
    assert QdrantSide(client(fake), "c", "dense", "Euclid")._distance(3.0) == 3.0
    fake.collections["c"]["config"]["hnsw_config"] = {"m": 0}
    assert QdrantSide(client(fake), "c", "dense", "Dot").index() == (None, None, [])


def test_split_searcher_routes_by_side() -> None:
    fake = eval_fake()
    new = QdrantSide(client(fake), "c", "dense", "Dot")

    class OldPg:
        def search(self, side: str, vector: Any, k: int, setting: Any = None) -> list[Any]:
            return [(f"{side}-pg", 0.0)]

        def index(self, side: str) -> tuple[str, int, list[int]]:
            return ("hnsw", 40, [40])

    searcher = SplitSearcher(QdrantRows(new, "text"), {"old": PgSide(OldPg()), "new": new})
    assert searcher.search("old", [0.0, 1.0], 1) == [("old-pg", 0.0)]
    assert searcher.search("new", [0.0, 1.0], 1)[0][0] == "10"
    assert searcher.index("old") == ("hnsw", 40, [40])
    assert searcher.texts("text", ["1"]) == {"1": "text 1"}
