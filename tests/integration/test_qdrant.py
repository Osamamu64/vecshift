"""doctor and bench against a real Qdrant.

Set VECSHIFT_TEST_QDRANT_URL to run them, for example:

    docker run -d -p 6333:6333 qdrant/qdrant
    VECSHIFT_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/integration
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import httpx
import pytest
from typer.testing import CliRunner

from vecshift.cli import app

URL = os.environ.get("VECSHIFT_TEST_QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="VECSHIFT_TEST_QDRANT_URL is not set")
runner = CliRunner()


@pytest.fixture
def collection() -> Iterator[str]:
    """A collection of 120 documents, half recording their model, reachable by alias."""
    assert URL
    name = f"vecshift_test_{uuid.uuid4().hex[:8]}"
    with httpx.Client(base_url=URL) as http:
        http.put(
            f"/collections/{name}", json={"vectors": {"size": 4, "distance": "Cosine"}}
        ).raise_for_status()
        points = [
            {
                "id": i,
                "vector": [float(i % 4 == k) + 0.1 for k in range(4)],
                "payload": {
                    "page_content": (
                        f"Document number {i} explains topic {i % 7} for the support team. "
                        f"It covers refunds, deliveries and account settings in case {i * 13}."
                    ),
                    "metadata": {"model_tag": "openai/text-embedding-3-small"} if i % 2 else {},
                },
            }
            for i in range(1, 121)
        ]
        http.put(
            f"/collections/{name}/points?wait=true", json={"points": points}
        ).raise_for_status()
        alias = {"create_alias": {"collection_name": name, "alias_name": f"{name}_live"}}
        http.post("/collections/aliases", json={"actions": [alias]}).raise_for_status()
        try:
            yield name
        finally:
            http.delete(f"/collections/{name}")


def doctor(*args: str) -> tuple[int, dict]:  # type: ignore[type-arg]
    assert URL
    result = runner.invoke(app, ["doctor", "--qdrant", URL, "--json", *args])
    assert result.exit_code in (0, 1), result.output
    return result.exit_code, json.loads(result.stdout)


def test_doctor_through_an_alias(collection: str) -> None:
    _, report = doctor("--collection", f"{collection}_live")
    ids = {f["id"]: f["severity"] for f in report["findings"]}
    assert report["store"] == "qdrant" and report["target"] == collection
    assert ids["alias.present"] == "ok" and ids["text.ok"] == "ok"
    assert ids["models.partial"] == "warning", "half the points record their model"
    assert ids["index.ok"] == "ok"
    assert report["connection"] == URL


def test_doctor_samples_large_collections(collection: str) -> None:
    _, report = doctor("--collection", collection, "--sample-size", "50")
    assert report["sample"] == {"rows": 50, "method": "random sample"}


def test_doctor_explains_bad_targets(collection: str) -> None:
    assert URL
    result = runner.invoke(app, ["doctor", "--qdrant", URL, "--collection", "no_such_thing"])
    assert result.exit_code == 2 and "No collection or alias named" in result.output
    assert collection in result.output, "candidates are listed"


def test_bench_samples_documents(collection: str) -> None:
    assert URL
    result = runner.invoke(
        app,
        [
            "bench", "--qdrant", URL, "--collection", collection,
            "-m", "hash/64", "-m", "hash/256", "--sample", "100", "--no-cache", "--json",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["documents"] == 100 and data["source"] == collection
    assert {m["name"] for m in data["models"]} == {"hash/64", "hash/256"}
