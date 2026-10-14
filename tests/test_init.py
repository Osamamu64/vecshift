"""``vecshift init`` for migrations into Qdrant, from flags and guided, against a fake Qdrant."""

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from tests.test_qdrant import FakeQdrant
from vecshift.cli import app
from vecshift.connectors import qdrant
from vecshift.jobs import JobSpec

runner = CliRunner()


def job(path: Path) -> JobSpec:
    return JobSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_flags_write_a_qdrant_job(tmp_path: Path) -> None:
    path = tmp_path / "j.yaml"
    result = runner.invoke(
        app, ["init", "--collection", "docs", "--model", "hash/8", "-o", str(path)]
    )
    assert result.exit_code == 0, result.output
    spec = job(path)
    assert spec.source.type == "qdrant" and spec.source.collection == "docs"
    assert spec.target.type == "qdrant" and spec.target_collection == "docs_v2"
    assert "url_env: VECSHIFT_QDRANT_URL" in path.read_text(encoding="utf-8")


def test_flags_write_a_pgvector_to_qdrant_job(tmp_path: Path) -> None:
    path = tmp_path / "j.yaml"
    args = ["init", "--table", "public.docs", "--to", "qdrant", "--alias", "docs_live"]
    result = runner.invoke(app, [*args, "--model", "hash/8", "-o", str(path)])
    assert result.exit_code == 0, result.output
    spec = job(path)
    assert spec.source.type == "pgvector" and spec.target.type == "qdrant"
    assert spec.qdrant_target.alias == "docs_live" and spec.target_collection == "docs_live_v2"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--table", "t", "--collection", "c"], "not both"),
        (["--collection", "c", "--to", "pgvector"], "only be migrated into Qdrant"),
        (["--table", "t", "--alias", "a"], "--alias is for a migration into Qdrant"),
        (["--table", "t", "--to", "qdrant", "--column", "e2"], "--column is for a pgvector"),
        (["--collection", "bad name!"], "aren't valid"),
    ],
)
def test_flags_that_dont_fit(tmp_path: Path, args: list[str], message: str) -> None:
    path = tmp_path / "j.yaml"
    result = runner.invoke(app, ["init", *args, "--model", "hash/8", "-o", str(path)])
    assert result.exit_code == 2 and message in result.output, result.output
    assert not path.exists()


def test_pgvector_flags_are_unchanged(tmp_path: Path) -> None:
    from vecshift.jobs import template

    path = tmp_path / "j.yaml"
    result = runner.invoke(app, ["init", "--table", "docs", "--model", "hash/8", "-o", str(path)])
    assert result.exit_code == 0, result.output
    assert path.read_text(encoding="utf-8") == template("docs", "hash/8")


# --- guided


def points(n: int, text_key: str = "page_content") -> list[dict[str, Any]]:
    return [
        {"id": i, "vector": [1.0, 0.0], "payload": {text_key: f"text {i}", "lang": "en"}}
        for i in range(1, n + 1)
    ]


class KeyedQdrant(FakeQdrant):
    """A fake Qdrant that, when ``key`` is set, refuses requests without it."""

    key: str | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.key is not None and request.headers.get("api-key") != self.key:
            return httpx.Response(401, json={"status": {"error": "unauthorized"}})
        return self(request)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> KeyedQdrant:
    """A fake Qdrant behind every client init makes, at a terminal, with nothing set."""
    server = KeyedQdrant()
    real = qdrant.QdrantClient

    def make(settings: qdrant.QdrantSettings, **kwargs: Any) -> qdrant.QdrantClient:
        return real(settings, transport=httpx.MockTransport(server.handle))

    monkeypatch.setattr(qdrant, "QdrantClient", make)
    monkeypatch.setattr("vecshift.cli_init.interactive", lambda: True)
    for name in ("VECSHIFT_DSN", "DATABASE_URL", "VECSHIFT_QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.setenv(name, "placeholder")  # recorded, so teardown removes what init sets
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return server


def guided(answers: list[str], *args: str) -> Any:
    return runner.invoke(app, ["--no-env-file", "init", *args], input="\n".join(answers) + "\n")


def test_guided_qdrant_through_an_alias(fake: KeyedQdrant, tmp_path: Path) -> None:
    fake.add("docs_v1", {"size": 2, "distance": "Cosine"}, points(5))
    fake.aliases.append(("docs", "docs_v1"))
    result = guided(
        [
            "2",  # a Qdrant collection
            "",  # the URL: the default, http://localhost:6333
            "",  # the text field: page_content, detected
            "",  # the old model: skipped
            "5",  # hash/256
            "n",  # don't check the plan
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Connected to Qdrant at http://localhost:6333" in result.output
    assert "Found docs_v1" in result.output and "alias docs" in result.output
    assert "Searches go through the alias docs" in result.output
    assert "export VECSHIFT_QDRANT_URL='…'" in result.output, "the URL isn't saved unasked"
    spec = job(tmp_path / "vecshift.yaml")
    assert spec.source.type == "qdrant" and spec.source.collection == "docs"
    assert spec.source.text_field == "page_content" and spec.model == "hash/256"
    assert spec.qdrant_target.alias is None and spec.target_collection == "docs_v2"


def test_guided_qdrant_without_an_alias(fake: KeyedQdrant, tmp_path: Path) -> None:
    fake.add("docs", {"size": 2, "distance": "Cosine"}, points(5, "body"))
    fake.add("docs_live_v2", {"size": 2, "distance": "Cosine"}, [])
    result = guided(
        [
            "",  # --to qdrant leaves two routes: Enter picks a Qdrant collection
            "",  # the URL
            "",  # which vectors: docs, the first
            "docs",  # an alias can't take a collection's name: asked again
            "docs_live",  # the alias cutover creates
            "",  # docs_live_v2 is taken: the suggested docs_live_v2_2
            "",  # the text field: body, detected
            "",  # the old model
            "5",
            "n",
        ],
        "--to",
        "qdrant",
    )
    assert result.exit_code == 0, result.output
    assert "searches docs by name" in result.output
    assert "docs is already a collection's name" in result.output
    assert "docs_live_v2 already exists" in result.output
    spec = job(tmp_path / "vecshift.yaml")
    assert spec.source.type == "qdrant" and spec.source.collection == "docs"
    assert spec.qdrant_target.alias == "docs_live" and spec.target_collection == "docs_live_v2_2"
    assert spec.source.text_field == "body"


def test_guided_qdrant_asks_for_a_key_and_never_shows_it(fake: KeyedQdrant, tmp_path: Path) -> None:
    fake.add("docs_v1", {"size": 2, "distance": "Cosine"}, points(3))
    fake.aliases.append(("docs", "docs_v1"))
    fake.key = "qdrant-key-0123456789"
    result = guided(["", "wrong-key", fake.key, "", "", "5", "n"], "--collection", "docs")
    assert result.exit_code == 0, result.output
    assert "This Qdrant needs an API key" in result.output
    assert "Qdrant refused that key" in result.output
    assert fake.key not in result.output and "wrong-key" not in result.output
    assert fake.key not in (tmp_path / "vecshift.yaml").read_text(encoding="utf-8")
    assert "export QDRANT_API_KEY='…'" in result.output


def test_guided_offers_every_route(fake: KeyedQdrant) -> None:
    result = guided(["3"])  # pgvector into Qdrant: asks for the database first
    assert "What do you want to migrate?" in result.output
    assert "A pgvector table, into Qdrant" in result.output
    assert "Connection string" in result.output
