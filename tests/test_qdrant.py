"""The Qdrant connector, against a fake Qdrant over httpx.MockTransport."""

import json
from typing import Any

import httpx
import pytest

from vecshift.connectors import qdrant
from vecshift.connectors.qdrant.documents import sample_documents
from vecshift.doctor import run_checks

KEY = "qdrant-key-never-print-me"


class FakeQdrant:
    """Answers the handful of REST calls the connector makes."""

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.points: dict[str, list[dict[str, Any]]] = {}
        self.aliases: list[tuple[str, str]] = []
        self.random_sampling = True
        self.requests: list[httpx.Request] = []

    def add(self, name: str, vectors: dict[str, Any], points: list[dict[str, Any]]) -> None:
        self.collections[name] = {
            "points_count": len(points),
            "config": {"params": {"vectors": vectors}, "hnsw_config": {"m": 16}},
        }
        self.points[name] = points

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        def ok(result: Any) -> httpx.Response:
            return httpx.Response(200, json={"result": result, "status": "ok"})

        if path == "/collections":
            return ok({"collections": [{"name": n} for n in self.collections]})
        if path == "/aliases":
            return ok(
                {"aliases": [{"alias_name": a, "collection_name": c} for a, c in self.aliases]}
            )
        parts = path.strip("/").split("/")
        name = parts[1] if len(parts) > 1 else ""
        if name not in self.collections:
            return httpx.Response(404, json={"status": {"error": "Not found"}})
        if len(parts) == 2:
            return ok(self.collections[name])
        points = self.points[name]
        if parts[-1] == "query":
            if not self.random_sampling:
                return httpx.Response(400, json={"status": {"error": "unknown variant"}})
            return ok({"points": points[: body["limit"]]})
        if parts[-1] == "scroll":
            start = body.get("offset") or 0
            page = points[start : start + body["limit"]]
            nxt = start + body["limit"] if start + body["limit"] < len(points) else None
            return ok({"points": page, "next_page_offset": nxt})
        if parts[-1] == "points":
            wanted = set(body["ids"])
            return ok([p for p in points if p["id"] in wanted])
        return httpx.Response(404)


def client(fake: FakeQdrant, url: str = "http://localhost:6333") -> qdrant.QdrantClient:
    return qdrant.QdrantClient(qdrant.prepare(url), transport=httpx.MockTransport(fake))


def docs(n: int, **extra: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "vector": [1.0, 0.0, 0.0] if i % 2 else [0.0, 1.0, 0.0],
            "payload": {"page_content": f"text {i}", "metadata": {"model_tag": "openai/x"}},
            **extra,
        }
        for i in range(1, n + 1)
    ]


# --- settings


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("localhost:6333", "doesn't look like a Qdrant URL"),
        ("ftp://host", "doesn't look like a Qdrant URL"),
        ("https://user:pass@host", "credentials"),
        ("https://host/?api-key=1", "query string"),
    ],
)
def test_bad_urls(url: str, message: str) -> None:
    with pytest.raises(qdrant.QdrantError, match=message):
        qdrant.prepare(url)


def test_key_is_never_sent_in_the_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", KEY)
    with pytest.raises(qdrant.QdrantError, match="plain http") as caught:
        qdrant.prepare("http://qdrant.example.com:6333")
    assert KEY not in str(caught.value)
    assert qdrant.prepare("http://localhost:6333").key == KEY, "fine on this machine"
    settings = qdrant.prepare("https://xyz.cloud.qdrant.io/")
    assert settings.url == "https://xyz.cloud.qdrant.io" and settings.key == KEY
    assert KEY not in settings.display


def test_key_header_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeQdrant()
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    client(fake).collections()
    assert "api-key" not in fake.requests[-1].headers
    monkeypatch.setenv("QDRANT_API_KEY", KEY)
    client(fake).collections()
    assert fake.requests[-1].headers["api-key"] == KEY


def test_errors_are_explained_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", KEY)

    def respond(status: int, **kwargs: Any) -> qdrant.QdrantClient:
        handler = httpx.MockTransport(lambda r: httpx.Response(status, **kwargs))
        return qdrant.QdrantClient(qdrant.prepare("http://localhost:6333"), transport=handler)

    with pytest.raises(qdrant.QdrantError, match="API key is missing or wrong") as caught:
        respond(403).collections()
    assert caught.value.hint and "QDRANT_API_KEY" in caught.value.hint
    with pytest.raises(qdrant.QdrantError, match="redirect"):
        respond(302, headers={"location": "https://elsewhere"}).collections()
    with pytest.raises(qdrant.QdrantError, match="returned 500: boom"):
        respond(500, json={"status": {"error": "boom"}}).collections()

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    broken = qdrant.QdrantClient(
        qdrant.prepare("http://localhost:6333"), transport=httpx.MockTransport(unreachable)
    )
    with pytest.raises(qdrant.QdrantError, match="Couldn't reach Qdrant") as caught:
        broken.collections()
    assert KEY not in str(caught.value) and KEY not in (caught.value.hint or "")


# --- inspection


def test_inspect_a_langchain_style_collection() -> None:
    fake = FakeQdrant()
    fake.add("docs", {"size": 3, "distance": "Cosine"}, docs(10))
    fake.aliases = [("docs_live", "docs")]
    profile = qdrant.inspect(client(fake), "docs_live")
    assert profile.store == "qdrant" and profile.target == "docs"
    assert profile.declared_dimensions == 3 and profile.estimated_rows == 10
    assert profile.text_field == "payload.page_content"
    assert profile.model_field == "payload.metadata.model_tag"
    assert profile.aliases == ("docs_live",)
    assert profile.sample.method == "full collection" and profile.sample.rows == 10
    assert profile.sample.duplicate_vectors == 8, "two distinct vectors among ten"
    assert profile.sample.norms == pytest.approx([1.0] * 10)
    assert profile.ann_indexes[0].metric == "cosine"
    found = {f.id: f.severity.value for f in run_checks(profile).findings}
    assert found["alias.present"] == "ok" and found["text.ok"] == "ok"


def test_named_vectors_and_missing_ones() -> None:
    fake = FakeQdrant()
    points = [
        {"id": 1, "vector": {"dense": [3.0, 4.0]}, "payload": {"text": "a"}},
        {"id": 2, "vector": {}, "payload": {"text": "b"}},
    ]
    fake.add("multi", {"dense": {"size": 2, "distance": "Dot"}, "other": {"size": 5}}, points)
    c = client(fake)
    with pytest.raises(qdrant.SelectionError, match="Choose one with --vector") as caught:
        qdrant.inspect(c, "multi")
    assert [f.qualified for f in caught.value.candidates] == ["multi:dense", "multi:other"]
    profile = qdrant.inspect(c, "multi", "dense")
    assert profile.target == "multi:dense" and profile.text_field == "payload.text"
    assert profile.sample.null_vectors == 1 and profile.sample.norms == [5.0]
    assert profile.ann_indexes[0].metric == "inner_product"
    found = {f.id for f in run_checks(profile).findings}
    assert "alias.none" in found and "vectors.null" in found


def test_selection_errors() -> None:
    fake = FakeQdrant()
    with pytest.raises(qdrant.SelectionError, match="No collections"):
        qdrant.inspect(client(fake))
    fake.add("a", {"size": 2}, [])
    fake.add("b", {"size": 2}, [])
    with pytest.raises(qdrant.SelectionError, match="more than one") as caught:
        qdrant.inspect(client(fake))
    assert len(caught.value.candidates) == 2
    with pytest.raises(qdrant.SelectionError, match="No collection or alias named 'c'"):
        qdrant.inspect(client(fake), "c")
    with pytest.raises(qdrant.SelectionError, match="no vector named 'x'"):
        qdrant.inspect(client(fake), "a", "x")


def test_large_collections_are_sampled() -> None:
    fake = FakeQdrant()
    fake.add("big", {"size": 3, "distance": "Cosine"}, docs(50))
    profile = qdrant.inspect(client(fake), "big", sample_size=20)
    assert profile.sample.method == "random sample" and profile.sample.rows == 20
    fake.random_sampling = False  # Qdrant before 1.11
    profile = qdrant.inspect(client(fake), "big", sample_size=20)
    assert profile.sample.method == "first points" and profile.sample.rows == 20


def test_text_field_by_path() -> None:
    fake = FakeQdrant()
    points = [{"id": 1, "vector": [1.0, 0.0], "payload": {"meta": {"body": "x"}}}]
    fake.add("c", {"size": 2, "distance": "Euclid"}, points)
    assert qdrant.inspect(client(fake), "c").text_field is None
    assert qdrant.inspect(client(fake), "c", text_field="meta.body").text_field == (
        "payload.meta.body"
    )
    with pytest.raises(qdrant.SelectionError, match="no_such"):
        qdrant.inspect(client(fake), "c", text_field="no_such")


def test_sample_documents_includes_labeled_ids() -> None:
    fake = FakeQdrant()
    fake.add("docs", {"size": 3, "distance": "Cosine"}, docs(30))
    rows, name = sample_documents(client(fake), "docs", None, None, 5, {"25"})
    ids = [i for i, _ in rows]
    assert name == "docs" and len(rows) == 6 and "25" in ids
    assert dict(rows)["25"] == "text 25"
