"""Migrations into Qdrant: the writer, against a fake Qdrant, and the plan."""

import asyncio
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.test_doctor_checks import make_profile
from tests.test_qdrant import FakeQdrant, client
from vecshift.connectors.pgvector.reader import point_id
from vecshift.connectors.qdrant import QdrantError
from vecshift.connectors.qdrant.target import QdrantTargetState, inspect_target, payload_indexes
from vecshift.connectors.qdrant.writer import (
    AlreadyRunning,
    Layout,
    QdrantSource,
    QdrantWriter,
)
from vecshift.doctor import AnnIndex, Severity
from vecshift.embeddings import create_embedder, parse_spec
from vecshift.jobs import JobSpec
from vecshift.migrate import ApplyResult, JobState, apply
from vecshift.planning import SampleStats
from vecshift.planning.qdrant import build_qdrant_plan


def points(n: int, start: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "vector": [1.0, 0.0],
            "payload": {"page_content": f"text {i}", "lang": "en"} if i % 10 else {"lang": "en"},
        }
        for i in range(start, start + n)
    ]


def setup(n: int = 25, page: int = 4) -> tuple[FakeQdrant, QdrantWriter]:
    fake = FakeQdrant()
    fake.add(
        "docs_v1",
        {"size": 2, "distance": "Cosine"},
        points(n),
        payload_schema={"lang": {"data_type": "keyword", "points": n}},
    )
    c = client(fake)
    layout = Layout("docs_v2", None, 8, "Cosine", payload_indexes(fake.collections["docs_v1"]))
    writer = QdrantWriter(
        c, QdrantSource(c, "docs_v1", "page_content", None), layout, page_size=page
    )
    return fake, writer


def run(writer: QdrantWriter, tmp_path: Path, **kwargs: Any) -> ApplyResult:
    async def go() -> ApplyResult:
        embedder = create_embedder(parse_spec("hash/8"))
        try:
            return await apply(
                writer,
                embedder,
                JobState(tmp_path / "state.json"),
                dims=8,
                price_per_million=None,
                budget_usd=None,
                chunk_rows=3,
                index=("hnsw", "cosine"),
                **kwargs,
            )
        finally:
            await embedder.aclose()

    return asyncio.run(go())


def copy(fake: FakeQdrant) -> dict[Any, dict[str, Any]]:
    return {p["id"]: p for p in fake.points["docs_v2"]}


def test_fills_a_new_collection(tmp_path: Path) -> None:
    fake, writer = setup()
    result = run(writer, tmp_path)
    assert result.status == "complete" and result.rows_written == 23, "two points have no text"
    target = copy(fake)
    assert set(target) == {i for i in range(1, 26) if i % 10}
    assert target[3]["payload"] == {"page_content": "text 3", "lang": "en"}
    assert len(target[3]["vector"]) == 8
    assert fake.collections["docs_v2"]["config"]["params"]["vectors"] == {
        "size": 8,
        "distance": "Cosine",
    }
    assert fake.collections["docs_v2"]["payload_schema"] == {"lang": {"data_type": "keyword"}}
    assert len(fake.points["docs_v1"]) == 25, "the source is only read"
    assert writer.pending_count() == 0 and writer.total_count() == 23
    assert writer.dimensions_in_use() == {8}


def test_catches_up_with_changes(tmp_path: Path) -> None:
    fake, writer = setup()
    run(writer, tmp_path)
    source = {p["id"]: p for p in fake.points["docs_v1"]}
    source[3]["payload"] = {"page_content": "edited", "lang": "en"}  # re-embed
    source[4]["payload"] = {"page_content": "text 4", "lang": "fr"}  # copy payload only
    source[5]["payload"] = {"lang": "en"}  # text removed: drop
    fake.points["docs_v1"] = [p for p in fake.points["docs_v1"] if p["id"] != 6]  # deleted
    fake.points["docs_v1"].append(
        {"id": 99, "vector": [0.0, 1.0], "payload": {"page_content": "new"}}
    )
    before = copy(fake)[4]["vector"]

    assert writer.pending_count() == 2
    result = run(writer, tmp_path)
    assert result.rows_written == 2 and writer.payloads_copied == 1
    assert writer.remove_deleted() == 2
    target = copy(fake)
    assert target[3]["payload"]["page_content"] == "edited"
    assert target[4]["payload"]["lang"] == "fr" and target[4]["vector"] == before
    assert 5 not in target and 6 not in target and 99 in target
    assert writer.pending_count() == 0


def test_status_counts_without_writing(tmp_path: Path) -> None:
    fake, writer = setup()
    run(writer, tmp_path)
    fake.points["docs_v1"][0]["payload"] = {"page_content": "text 1", "lang": "de"}
    reader = QdrantWriter(writer.client, writer.source, writer.layout, read_only=True)
    assert reader.pending_count() == 0
    assert copy(fake)[1]["payload"]["lang"] == "en", "status must not copy payloads"


def test_resumes_and_skips_rejected(tmp_path: Path) -> None:
    fake, writer = setup(n=30, page=7)
    result = run(writer, tmp_path, until=0.5)
    assert result.status == "stopped" and 0 < len(fake.points["docs_v2"]) < 27
    assert writer.pending_count(("28", "29")) == writer.pending_count() - 2
    assert "29" not in {i.id for i in writer.fetch(None, 100, ("29",))}
    assert run(writer, tmp_path).status == "complete"
    assert len(fake.points["docs_v2"]) == 27


def test_uuid_ids_page_in_order() -> None:
    fake = FakeQdrant()
    ids = sorted(str(uuid.uuid4()) for _ in range(9))
    fake.add(
        "u",
        {"size": 2},
        [{"id": i, "vector": [1.0, 0.0], "payload": {"text": i}} for i in ids] + points(3),
    )
    source = QdrantSource(client(fake), "u", "text", None)
    seen: list[Any] = []
    after: Any = None
    while True:
        items, after = source.page(after, 2)
        seen.extend(i.key for i in items)
        if after is None:
            break
    assert seen == [i for i in ids], "integer points have no text field here"
    assert source.count() == 9


def test_existing_collection_must_match(tmp_path: Path) -> None:
    fake, writer = setup()
    fake.add("docs_v2", {"size": 3, "distance": "Cosine"}, [])
    with pytest.raises(QdrantError, match="different vector size"):
        writer.ensure_column()
    fake.add("docs_v2", {"size": 8, "distance": "Cosine"}, [])
    assert writer.ensure_column() is False


def test_named_vector(tmp_path: Path) -> None:
    fake, writer = setup()
    writer.layout = Layout("docs_v2", "dense", 8, "Dot")
    run(writer, tmp_path)
    assert fake.collections["docs_v2"]["config"]["params"]["vectors"] == {
        "dense": {"size": 8, "distance": "Dot"}
    }
    assert set(copy(fake)[1]["vector"]) == {"dense"}


def test_one_apply_at_a_time(tmp_path: Path) -> None:
    _, writer = setup()
    lock = tmp_path / ".vecshift" / "job.lock"
    first = QdrantWriter(writer.client, writer.source, writer.layout, lock)
    second = QdrantWriter(writer.client, writer.source, writer.layout, lock)
    first.acquire()
    with pytest.raises(AlreadyRunning, match="docs_v2"):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


def test_waits_for_indexing() -> None:
    fake, writer = setup()
    writer.ensure_column()
    states = iter(["yellow", "yellow", "green"])
    naps: list[float] = []

    def nap(seconds: float) -> None:
        naps.append(seconds)
        fake.collections["docs_v2"]["status"] = next(states)

    writer._sleep = nap
    fake.collections["docs_v2"]["status"] = "yellow"
    assert writer.build_index("hnsw", "cosine", 10) == "built" and len(naps) == 3
    assert writer.build_index("hnsw", "cosine", 10) == "exists"
    fake.collections["docs_v2"]["status"] = "red"
    with pytest.raises(QdrantError, match="red"):
        writer.build_index("hnsw", "cosine", 10)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (7, 7),
        (uuid.UUID(int=5), str(uuid.UUID(int=5))),
        ("6F1F4B38-7C4C-4F62-9B39-5C8D5E5D6A01", "6f1f4b38-7c4c-4f62-9b39-5c8d5e5d6a01"),
    ],
)
def test_point_ids(key: Any, expected: Any) -> None:
    assert point_id(key) == expected


def test_other_keys_get_stable_uuids() -> None:
    for key in ("doc-1", -3, True):
        made = point_id(key)
        assert isinstance(made, str) and uuid.UUID(made).version == 5
        assert point_id(key) == made
    assert point_id("a") != point_id("b")


# --- inspecting the target


def test_inspect_target() -> None:
    from vecshift.connectors.qdrant import select_field

    fake = FakeQdrant()
    fake.add(
        "docs_v1",
        {"dense": {"size": 2, "distance": "Cosine"}, "title": {"size": 2}},
        points(3),
        payload_schema={
            "lang": {"data_type": "keyword"},
            "n": {"params": {"type": "integer", "range": True}},
        },
    )
    fake.aliases = [("docs", "docs_v1")]
    c = client(fake)
    field, info = select_field(c, "docs", "dense")
    state = inspect_target(c, "docs_v2", "dense", None, ("docs", field, info))
    assert state.alias == "docs" and state.alias_target == "docs_v1" and state.live == "docs_v1"
    assert state.source_is_alias and not state.exists
    assert state.other_vectors == ("title",)
    assert state.payload_indexes == {"lang": "keyword", "n": {"type": "integer", "range": True}}
    fake.add("docs_v2", {"dense": {"size": 8, "distance": "Cosine"}}, [])
    state = inspect_target(c, "docs_v2", "dense", None, ("docs", field, info))
    assert state.exists and state.dimensions == 8 and state.distance == "Cosine"


# --- planning


SAMPLE = SampleStats(documents=100, average_chars=400.0)


def qjob(**target: Any) -> JobSpec:
    return JobSpec.model_validate(
        {
            "source": {"type": "qdrant", "collection": "docs"},
            "target": target,
            "model": "ollama/nomic-embed-text",
            "limits": {"budget_usd": 10},
        }
    )


def state(**overrides: Any) -> QdrantTargetState:
    values: dict[str, Any] = {
        "collection": "docs_v2",
        "vector": None,
        "alias": "docs",
        "alias_target": "docs_v1",
        "live": "docs_v1",
        "source_is_alias": True,
        "collections": frozenset({"docs_v1"}),
    }
    values.update(overrides)
    return QdrantTargetState(**values)


def qplan(j: JobSpec | None = None, s: QdrantTargetState | None = None, **profile: Any) -> Any:
    defaults: dict[str, Any] = {
        "store": "qdrant",
        "target": "docs_v1",
        "estimated_rows": 10_000,
        "rows": 100,
        "models": Counter({"text-embedding-ada-002": 100}),
        "ann_indexes": [AnnIndex("docs_v1", "hnsw", "inner_product")],
    }
    defaults.update(profile)
    return build_qdrant_plan(j or qjob(), make_profile(**defaults), s or state(), SAMPLE)


def found(p: Any) -> dict[str, Severity]:
    return {f.id: f.severity for f in p.findings}


def test_qdrant_plan_happy_path() -> None:
    p = qplan()
    assert p.ok and p.store == "qdrant" and p.target_column == "docs_v2"
    assert p.metric == "inner_product", "inherits the source's distance"
    assert [c.kind for c in p.changes] == ["create_collection", "embed", "index", "cutover"]
    assert "Dot" in p.changes[0].summary
    assert "alias docs (now docs_v1) → docs_v2" in p.changes[-1].summary
    assert p.estimates.new_bytes == p.estimates.rows * 768 * 4
    assert p.to_dict()["store"] == "qdrant"


def test_qdrant_plan_problems() -> None:
    assert found(qplan(s=state(other_vectors=("title",))))["plan.other_vectors"] is Severity.ERROR
    assert found(qplan(s=state(collection="docs_v1")))["plan.target_is_source"] is Severity.ERROR
    assert found(qplan(s=state(alias_target="docs_v2")))["plan.cut_over"] is Severity.ERROR
    conflict = state(exists=True, dimensions=3, distance="Dot")
    assert found(qplan(s=conflict))["plan.collection_conflict"] is Severity.ERROR
    same = state(exists=True, dimensions=768, distance="Dot")
    assert found(qplan(s=same))["plan.collection_exists"] is Severity.INFO
    assert "create_collection" not in [c.kind for c in qplan(s=same).changes]
    alone = qplan(s=state(alias=None, alias_target=None, source_is_alias=False))
    assert found(alone)["plan.no_alias"] is Severity.WARNING and alone.ok
    assert "cutover" not in [c.kind for c in alone.changes]
    taken = state(alias="docs_v1", alias_target=None)
    assert found(qplan(s=taken))["plan.alias_conflict"] is Severity.ERROR


def test_qdrant_plan_text_and_metric() -> None:
    p = qplan(qjob(metric="l2"), texts_present=90)
    assert p.metric == "l2" and "Euclid" in p.changes[0].summary
    assert found(p)["plan.partial_text"] is Severity.WARNING
    assert "gone after cutover" in next(f.detail for f in p.findings if f.id == "plan.partial_text")
    assert found(qplan(text_field=None))["plan.no_text"] is Severity.ERROR


def test_pgvector_source_needs_a_single_key() -> None:
    from tests.test_planner import target

    j = JobSpec.model_validate(
        {
            "source": {"table": "docs"},
            "target": {"type": "qdrant"},
            "model": "ollama/nomic-embed-text",
        }
    )
    s = state(alias=None, alias_target=None, live=None, source_is_alias=False)
    profile = make_profile(estimated_rows=100, rows=100)
    p = build_qdrant_plan(j, profile, s, SAMPLE, None, target(primary_key=("a", "b")))
    assert found(p)["plan.no_primary_key"] is Severity.ERROR
    p = build_qdrant_plan(
        j, profile, s, SAMPLE, None, target(primary_key=("id",), row_security=True)
    )
    assert found(p)["plan.row_security"] is Severity.ERROR
    assert "plan.no_primary_key" not in found(p)
