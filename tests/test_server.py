"""The REST API against temp job folders and a fake Qdrant: endpoints, parity with the CLI's
JSON, errors as problem+json, and the rules every request passes."""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.test_qdrant import FakeQdrant
from vecshift.cli import app as cli
from vecshift.connectors import qdrant
from vecshift.server import ServerSettings, create_app, openapi

runner = CliRunner()
OPENAPI = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
TOKEN = "t0ken-for-tests-0123456789"

QDRANT_JOB = """\
name: docs
source: {type: qdrant, collection: docs}
target: {type: qdrant}
model: hash/8
"""
PG_JOB = """\
name: articles
source: {type: pgvector, table: public.articles}
target: {type: pgvector, column: embedding_v2}
model: compat/m,url=https://models.example.com/v1?token=sk-in-the-url,dims=8
"""


def points(n: int) -> list[dict[str, Any]]:
    return [
        {"id": i, "vector": [1.0, 0.0], "payload": {"page_content": f"text {i}"}}
        for i in range(1, n + 1)
    ]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeQdrant:
    server = FakeQdrant()
    server.add("docs_v1", {"size": 2, "distance": "Cosine"}, points(5))
    server.aliases.append(("docs", "docs_v1"))
    real = qdrant.QdrantClient

    def make(settings: qdrant.QdrantSettings, **kwargs: Any) -> qdrant.QdrantClient:
        return real(settings, transport=httpx.MockTransport(server))

    monkeypatch.setattr(qdrant, "QdrantClient", make)
    for name in ("VECSHIFT_DSN", "DATABASE_URL", "QDRANT_API_KEY", "VECSHIFT_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VECSHIFT_QDRANT_URL", "http://localhost:6333")
    return server


@pytest.fixture
def jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "docs.yaml").write_text(QDRANT_JOB, encoding="utf-8")
    (tmp_path / "articles.yml").write_text(PG_JOB, encoding="utf-8")
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a job", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def api(jobs: Path, **settings: Any) -> TestClient:
    base = settings.pop("base_url", "http://localhost")
    return TestClient(create_app(ServerSettings(jobs=jobs, **settings)), base_url=base)


def cli_json(*args: str) -> Any:
    result = runner.invoke(cli, ["--no-env-file", *args, "--json"])
    assert result.exit_code in (0, 1), result.output
    return json.loads(result.stdout)


# --- endpoints


def test_meta(jobs: Path) -> None:
    from vecshift import __version__

    client = api(jobs)
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/version").json() == {"version": __version__, "api": "v1"}


def test_jobs_list_every_file_with_a_quick_stage(jobs: Path, fake: FakeQdrant) -> None:
    listed = {j["name"]: j for j in api(jobs).get("/api/v1/jobs").json()["jobs"]}
    assert set(listed) == {"articles", "compose", "docs"}, "notes.txt isn't a job file"
    docs = listed["docs"]
    assert docs["valid"] and docs["stage"] == "not_started"
    assert docs["source"] == {"type": "qdrant", "name": "docs"}
    assert docs["target"] == {"type": "qdrant", "name": "docs_v2"}
    assert listed["compose"]["valid"] is False
    assert listed["compose"]["problem"]["code"] == "job.invalid"
    assert str(jobs) not in json.dumps(listed), "the server's own paths stay private"
    articles = listed["articles"]
    assert articles["stage"] is None and articles["problem"]["code"] == "env.missing"
    assert "sk-in-the-url" not in json.dumps(articles)


def test_job_detail(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    docs = client.get("/api/v1/jobs/docs").json()
    assert docs["spec"]["source"]["collection"] == "docs"
    assert docs["yaml"] == QDRANT_JOB
    assert docs["state"] == {
        "runs": 0,
        "rows_written": 0,
        "tokens": 0,
        "spent_usd": 0.0,
        "failed": 0,
        "history": [],
        "collections": [],
        "updated_at": None,
    }
    articles = client.get("/api/v1/jobs/articles").json()
    assert "sk-in-the-url" not in json.dumps(articles), "URL query strings can carry keys"
    assert "?…" in articles["spec"]["model"]


@pytest.mark.parametrize("name", ["nope", "..%2Fdocs", "docs.yaml", ".hidden", "a b"])
def test_unknown_names_are_404(jobs: Path, name: str) -> None:
    response = api(jobs).get(f"/api/v1/jobs/{name}")
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] in {"job.missing", "http.404"}


def test_a_link_out_of_the_folder_isnt_a_job(jobs: Path, tmp_path_factory: Any) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.yaml"
    outside.write_text(QDRANT_JOB, encoding="utf-8")
    try:
        (jobs / "linked.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("can't create symlinks here")
    assert api(jobs).get("/api/v1/jobs/linked").status_code == 404


def test_status_and_plan_match_the_cli(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    assert client.get("/api/v1/jobs/docs/status?full=true").json() == cli_json(
        "status", "docs.yaml"
    )
    assert client.get("/api/v1/jobs/docs/plan").json() == cli_json("plan", "docs.yaml")


def test_doctor_matches_the_cli(jobs: Path, fake: FakeQdrant) -> None:
    body = {"store": "qdrant", "collection": "docs"}
    response = api(jobs).post("/api/v1/doctor", json=body, headers={"X-Vecshift": "1"})
    assert response.status_code == 200, response.json()
    assert response.json() == cli_json(
        "doctor", "--qdrant", "http://localhost:6333", "--collection", "docs"
    )


def test_doctor_reads_only_variables_vecshift_uses(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs, base_url="http://127.0.0.1")
    for body in (
        {"store": "pgvector", "dsn_env": "HOME"},
        {"store": "pgvector", "dsn_env": "OPENAI_API_KEY"},  # a key, not a connection
        {"store": "qdrant", "url_env": "PATH"},
    ):
        response = client.post("/api/v1/doctor", json=body, headers={"X-Vecshift": "1"})
        assert response.status_code == 422, body
        assert response.json()["code"] == "options.invalid"


def test_doctor_lists_the_choices(jobs: Path, fake: FakeQdrant) -> None:
    fake.add("other", {"size": 2, "distance": "Cosine"}, points(2))
    response = api(jobs).post(
        "/api/v1/doctor", json={"store": "qdrant"}, headers={"X-Vecshift": "1"}
    )
    assert response.status_code == 409
    problem = response.json()
    assert problem["code"] == "source.ambiguous"
    assert {c["name"] for c in problem["candidates"]} == {"docs_v1", "other"}


def test_probing_a_remote_model_needs_allow_remote(
    jobs: Path, fake: FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[int] = []

    async def post(self: httpx.AsyncClient, url: str, json: dict[str, Any]) -> httpx.Response:
        sent.append(len(json["input"]))
        data = [{"index": i, "embedding": [0.1] * 4} for i in range(len(json["input"]))]
        return httpx.Response(
            200,
            json={"data": data, "usage": {"prompt_tokens": 9}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789")
    remote = QDRANT_JOB.replace("hash/8", "openai/text-embedding-3-small")
    (jobs / "docs.yaml").write_text(remote, encoding="utf-8")
    client = api(jobs)

    refused = client.get("/api/v1/jobs/docs/plan?probe=true")
    assert refused.status_code == 422 and not sent
    assert refused.json()["code"] == "consent.required"
    assert "https://api.openai.com/v1" in refused.json()["detail"]
    allowed = client.get("/api/v1/jobs/docs/plan?probe=true&allow_remote=true")
    assert allowed.status_code == 200 and sent == [5]
    assert allowed.json()["dimensions_source"] == "probe"


def test_environment_names_without_values(jobs: Path, fake: FakeQdrant) -> None:
    os.environ["VECSHIFT_DSN"] = "postgresql://app:hunter2-secret@db.example.com:5432/shop"
    try:
        found = {v["name"]: v for v in api(jobs).get("/api/v1/environment").json()["variables"]}
    finally:
        del os.environ["VECSHIFT_DSN"]
    assert found["VECSHIFT_DSN"]["display"] == "postgresql://app@db.example.com:5432/shop"
    assert found["VECSHIFT_DSN"]["used_by"] == ["articles"]
    assert found["VECSHIFT_QDRANT_URL"] == {
        "name": "VECSHIFT_QDRANT_URL",
        "kind": "url",
        "set": True,
        "display": "http://localhost:6333",
        "problem": None,
        "used_by": ["docs"],
    }
    assert found["OPENAI_API_KEY"]["set"] is False and found["OPENAI_API_KEY"]["display"] is None
    assert "hunter2" not in json.dumps(found)


# --- the rules every request passes


def test_only_localhost_names_without_a_token(jobs: Path) -> None:
    for host in ("http://localhost:8765", "http://127.0.0.1", "http://[::1]:8765"):
        assert api(jobs, base_url=host).get("/api/v1/health").status_code == 200, host
    refused = api(jobs, base_url="http://attacker.example").get("/api/v1/jobs")
    assert refused.status_code == 421 and refused.json()["code"] == "request.host"


def test_a_token_is_needed_once_set(jobs: Path) -> None:
    client = api(jobs, token=TOKEN, base_url="http://api.internal")
    assert client.get("/api/v1/health").status_code == 200, "health checks stay open"
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}):
        refused = client.get("/api/v1/jobs", headers=headers)
        assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Bearer"
    ok = client.get("/api/v1/jobs", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200


def test_changes_need_the_custom_header(jobs: Path) -> None:
    client = api(jobs)
    refused = client.post("/api/v1/doctor", json={"store": "qdrant"})
    assert refused.status_code == 403 and refused.json()["code"] == "request.header"


def test_bodies_are_capped(jobs: Path) -> None:
    client = api(jobs, max_body=100)
    response = client.post(
        "/api/v1/doctor", content=b"{" + b" " * 200 + b"}", headers={"X-Vecshift": "1"}
    )
    assert response.status_code == 413


def test_every_response_is_locked_down(jobs: Path) -> None:
    headers = api(jobs).get("/api/v1/health").headers
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in headers["content-security-policy"]
    assert "access-control-allow-origin" not in headers
    assert "server" not in headers


def test_invalid_requests_dont_echo_their_input(jobs: Path) -> None:
    body = {"store": "qdrant", "sample": "sk-planted-secret-value"}
    response = api(jobs).post("/api/v1/doctor", json=body, headers={"X-Vecshift": "1"})
    assert response.status_code == 422 and response.json()["code"] == "request.invalid"
    assert "sk-planted-secret-value" not in response.text


def test_unexpected_errors_say_nothing(jobs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vecshift.service

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("postgresql://app:hunter2@db/shop")

    monkeypatch.setattr(vecshift.service, "status", broken)
    client = TestClient(
        create_app(ServerSettings(jobs=jobs)),
        base_url="http://localhost",
        raise_server_exceptions=False,
    )
    response = client.get("/api/v1/jobs/docs/status")
    assert response.status_code == 500 and response.json()["code"] == "internal"
    assert "hunter2" not in response.text and "Traceback" not in response.text


@pytest.mark.parametrize(
    ("host", "token", "problem"),
    [
        ("127.0.0.1", None, None),
        ("0.0.0.0", None, "needs a token"),  # noqa: S104
        ("0.0.0.0", TOKEN, None),  # noqa: S104
        ("127.0.0.1", "short", "too short"),
    ],
)
def test_where_the_server_may_listen(
    tmp_path: Path, host: str, token: str | None, problem: str | None
) -> None:
    found = ServerSettings(jobs=tmp_path, host=host, token=token).problem()
    assert (found is None) if problem is None else (problem in (found or ""))


def test_serve_refuses_to_listen_openly_without_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VECSHIFT_API_TOKEN", raising=False)
    result = runner.invoke(
        cli,
        ["--no-env-file", "serve", "--host", "0.0.0.0", "--jobs", str(tmp_path)],  # noqa: S104
    )
    assert result.exit_code == 2 and "VECSHIFT_API_TOKEN" in result.output


def test_serve_says_how_to_install_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", None)
    result = runner.invoke(cli, ["--no-env-file", "serve"])
    assert result.exit_code == 2 and 'pip install "vecshift[server]"' in result.output


def test_the_committed_schema_is_current() -> None:
    """docs/openapi.json is the API's contract. After a deliberate change, regenerate it with
    VECSHIFT_UPDATE_CONTRACTS=1 and call it out in the changelog."""
    schema = openapi()
    if os.environ.get("VECSHIFT_UPDATE_CONTRACTS"):
        OPENAPI.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    assert json.loads(OPENAPI.read_text(encoding="utf-8")) == schema


def test_problem_texts_hide_url_query_strings() -> None:
    from vecshift.server.problems import scrub

    text = "bad: compat/m,url=https://x.example/v1?token=sk-abc,dims=8 (see http://h:1/a?b=c)"
    assert (
        scrub(text, ["dims=8"]) == "bad: compat/m,url=https://x.example/v1?…,… (see http://h:1/a?…)"
    )


def test_the_cli_lists_the_choices_too(jobs: Path, fake: FakeQdrant) -> None:
    fake.add("other", {"size": 4, "distance": "Cosine"}, points(2))
    result = runner.invoke(cli, ["--no-env-file", "doctor", "--qdrant", "http://localhost:6333"])
    assert result.exit_code == 2
    assert (
        "Vectors found:\n  docs_v1  float32 vector(2)\n  other  float32 vector(4)\n"
        in result.output
    )
