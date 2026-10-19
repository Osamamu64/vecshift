"""Job files over HTTP (create, change, delete) and bench runs, against temp job folders and
a fake Qdrant."""

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests import test_server
from tests.test_qdrant import FakeQdrant
from tests.test_server import PG_JOB, api
from vecshift.cli import app as cli
from vecshift.jobs import load_job
from vecshift.jobs.spec import template
from vecshift.migrate import JobState

fake = test_server.fake
jobs = test_server.jobs
POST = {"X-Vecshift": "1"}
NEW = {"source": {"table": "public.notes", "text_column": "body"}, "model": "hash/8"}


def create(client: TestClient, name: str, spec: dict[str, Any], **extra: Any) -> Any:
    body = {"name": name, "spec": spec, **extra}
    return client.post("/api/v1/jobs", json=body, headers=POST)


def change(client: TestClient, name: str, spec: dict[str, Any], **extra: Any) -> Any:
    return client.put(f"/api/v1/jobs/{name}", json={"spec": spec, **extra}, headers=POST)


def hold(client: TestClient, job: str | None) -> threading.Event:
    """Start a run that lasts until the returned event is set."""
    release = threading.Event()
    registry = client.app.state.registry  # type: ignore[attr-defined]
    registry.start(job, "apply", lambda run: release.wait(30))
    return release


# --- creating


def test_creating_a_job_writes_inits_commented_file(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    response = create(client, "notes", NEW)
    assert response.status_code == 201, response.text
    assert response.headers["location"] == "/api/v1/jobs/notes"
    body = response.json()
    assert body["written"] and body["comments_kept"]
    assert body["spec"]["name"] == "notes", "the name inside defaults to the file's"
    path = jobs / "notes.yaml"
    text = path.read_text(encoding="utf-8")
    assert body["yaml"] == text
    assert "# Check it with `vecshift plan`" in text
    assert load_job(path).model_dump(mode="json") == body["spec"]
    assert client.get("/api/v1/jobs/notes").json()["spec"] == body["spec"]
    assert "notes" in [j["name"] for j in client.get("/api/v1/jobs").json()["jobs"]]


def test_a_dry_run_writes_nothing(jobs: Path, fake: FakeQdrant) -> None:
    response = create(api(jobs), "notes", NEW, dry_run=True)
    assert response.status_code == 200
    assert response.json()["written"] is False
    assert 'table: "public.notes"' in response.json()["yaml"]
    assert not (jobs / "notes.yaml").exists()


def test_a_new_job_is_checked_first(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    response = create(client, "notes", {"source": {"type": "qdrant"}, "model": "nope"})
    assert response.status_code == 422
    problem = response.json()
    assert problem["code"] == "job.invalid"
    assert problem["detail"].startswith("The job has problems:\n  source.collection: required")
    assert {"loc": "source.collection", "message": "required"} in problem["errors"]
    assert create(client, "docs", NEW).json()["code"] == "job.exists"
    assert create(client, "articles", NEW).json()["code"] == "job.exists", ".yml counts too"
    for name in ("../up", "a/b", ".hidden", ""):
        assert create(client, name, NEW).status_code == 422, name
    taken = create(client, "notes", {**NEW, "name": "docs"})
    assert taken.status_code == 409 and taken.json()["code"] == "job.name_taken"
    assert not list(jobs.glob("notes.y*ml"))


def test_a_model_url_with_a_query_string_isnt_taken(jobs: Path, fake: FakeQdrant) -> None:
    model = "compat/m,url=https://models.example.com/v1?token=sk-sent,dims=8"
    response = create(api(jobs), "notes", {**NEW, "model": model})
    assert response.status_code == 422
    assert "sk-sent" not in response.text
    assert not (jobs / "notes.yaml").exists()


# --- changing


def test_changing_a_job_keeps_its_comments(jobs: Path, fake: FakeQdrant) -> None:
    path = jobs / "notes.yaml"
    path.write_text(template("notes", "hash/8", budget_usd=5), encoding="utf-8")
    client = api(jobs)
    spec = client.get("/api/v1/jobs/notes").json()["spec"]
    spec["model"] = "hash/16"
    spec["limits"]["budget_usd"] = None
    response = change(client, "notes", spec)
    assert response.status_code == 200, response.text
    assert response.json()["comments_kept"] and response.json()["written"]
    text = path.read_text(encoding="utf-8")
    assert "model: hash/16" in text
    assert "  # budget_usd: 5 " in text
    assert "# Check it with `vecshift plan`" in text
    assert load_job(path).model == "hash/16"


def test_a_dry_run_change_writes_nothing(jobs: Path, fake: FakeQdrant) -> None:
    before = (jobs / "docs.yaml").read_text(encoding="utf-8")
    client = api(jobs)
    spec = client.get("/api/v1/jobs/docs").json()["spec"]
    response = change(client, "docs", {**spec, "model": "hash/16"}, dry_run=True)
    assert response.json()["written"] is False
    assert "hash/16" in response.json()["yaml"]
    assert (jobs / "docs.yaml").read_text(encoding="utf-8") == before


def test_a_flow_style_file_is_rewritten_and_says_so(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    spec = client.get("/api/v1/jobs/docs").json()["spec"]
    response = change(client, "docs", {**spec, "model": "hash/16"})
    assert response.json()["comments_kept"] is False
    assert load_job(jobs / "docs.yaml").model == "hash/16"


def test_an_invalid_file_can_be_fixed(jobs: Path, fake: FakeQdrant) -> None:
    response = change(api(jobs), "compose", {"name": "compose", **NEW})
    assert response.status_code == 200, response.text
    assert load_job(jobs / "compose.yaml").name == "compose"


def test_a_hidden_query_string_stays_in_the_file(jobs: Path, fake: FakeQdrant) -> None:
    """GET shows a model URL's query string as ?…; sending that back keeps the file's."""
    client = api(jobs)
    spec = client.get("/api/v1/jobs/articles").json()["spec"]
    assert "?…" in spec["model"]
    response = change(client, "articles", {**spec, "limits": {"budget_usd": 9}})
    assert response.status_code == 200, response.text
    assert "sk-in-the-url" not in response.text
    job = load_job(jobs / "articles.yml")
    assert job.model == yaml.safe_load(PG_JOB)["model"]
    assert job.limits.budget_usd == 9
    sneaky = {**spec, "model": spec["model"].replace("?…", "?token=other")}
    assert change(client, "articles", sneaky).status_code == 422


def test_renaming_a_job_with_progress_is_refused(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    spec = client.get("/api/v1/jobs/docs").json()["spec"]
    taken = change(client, "docs", {**spec, "name": "articles"})
    assert taken.status_code == 409 and taken.json()["code"] == "job.name_taken"
    assert change(client, "docs", {**spec, "name": "fresh"}).status_code == 200, "no state yet"
    JobState.for_job(jobs / "docs.yaml", "fresh").save()
    refused = change(client, "docs", {**spec, "name": "again"})
    assert refused.status_code == 409 and refused.json()["code"] == "job.renamed"
    assert load_job(jobs / "docs.yaml").name == "fresh"


def test_no_changes_while_a_run_is_active(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    spec = client.get("/api/v1/jobs/docs").json()["spec"]
    release = hold(client, "docs")
    try:
        for response in (
            change(client, "docs", {**spec, "model": "hash/16"}),
            client.delete("/api/v1/jobs/docs", headers=POST),
        ):
            assert response.status_code == 409 and response.json()["code"] == "job.running"
        assert change(client, "docs", spec, dry_run=True).status_code == 200
    finally:
        release.set()
    assert load_job(jobs / "docs.yaml").model == "hash/8"


# --- deleting


def test_deleting_removes_only_the_file(jobs: Path, fake: FakeQdrant) -> None:
    state = JobState.for_job(jobs / "docs.yaml", "docs")
    state.save()
    client = api(jobs)
    response = client.delete("/api/v1/jobs/docs", headers=POST)
    assert response.status_code == 204 and response.content == b""
    assert not (jobs / "docs.yaml").exists()
    assert state.path.exists(), "the job's progress stays"
    assert client.get("/api/v1/jobs/docs").status_code == 404
    assert client.delete("/api/v1/jobs/docs", headers=POST).status_code == 404
    assert "docs" in fake.collections or "docs_v1" in fake.collections


def test_writes_need_the_csrf_header(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    assert client.post("/api/v1/jobs", json={"name": "n", "spec": NEW}).status_code == 403
    assert client.delete("/api/v1/jobs/docs").status_code == 403
    assert (jobs / "docs.yaml").exists()


# --- bench


def documents(n: int = 60) -> list[dict[str, str]]:
    words = ["vector", "index", "migration", "model", "embedding", "search", "recall", "budget"]
    return [
        {
            "id": str(i),
            "text": " ".join(
                f"{words[(i + j) % len(words)].capitalize()} {words[(i * j) % len(words)]} "
                f"number {i} sentence {j} about {words[(i * 7 + j) % len(words)]} here."
                for j in range(4)
            ),
        }
        for i in range(1, n + 1)
    ]


def finished(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        run: dict[str, Any] = client.get(f"/api/v1/runs/{run_id}").json()
        if run["state"] not in {"queued", "running", "stopping"}:
            return run
        time.sleep(0.05)
    raise AssertionError("the run didn't finish")


def bench(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/api/v1/bench/runs", json=body, headers=POST)


def events(client: TestClient, run_id: str) -> list[tuple[str, Any]]:
    text = client.get(f"/api/v1/runs/{run_id}/events").text
    found = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            found.append((fields["event"], json.loads(fields["data"])))
    return found


def timeless(result: dict[str, Any]) -> dict[str, Any]:
    """The result without the timings, which differ from run to run."""
    timings = {"docs_per_second", "query_ms_p50", "query_ms_p95"}
    models = [{k: v for k, v in m.items() if k not in timings} for m in result["models"]]
    return {**result, "models": models}


def test_bench_over_http_matches_the_cli(jobs: Path, fake: FakeQdrant, tmp_path: Path) -> None:
    docs = documents()
    client = api(jobs)
    started = bench(
        client,
        {
            "documents": {"type": "documents", "documents": docs},
            "models": ["hash/32", "hash/64"],
            "num_queries": 20,
            "cache": False,
        },
    )
    assert started.status_code == 202, started.text
    run = finished(client, started.json()["run_id"])
    assert run["state"] == "completed", run
    assert run["job"] is None and run["action"] == "bench"

    file = tmp_path / "corpus" / "documents"
    file.parent.mkdir()
    file.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            *("--no-env-file", "bench", "--docs", str(file), "-m", "hash/32", "-m", "hash/64"),
            *("--num-queries", "20", "--no-cache", "--json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert timeless(run["result"]) == timeless(json.loads(result.stdout))

    seen = events(client, run["run_id"])
    kinds = [kind for kind, _ in seen]
    assert kinds == ["plan", "step", "step", "result"]
    plan = seen[0][1]
    assert plan["documents"] == 60
    assert [m["model"] for m in plan["models"]] == ["hash/32", "hash/64"]
    assert seen[1][1] == {"message": "Embedding with hash/32…"}

    page = client.get(f"/api/v1/runs/{run['run_id']}/report")
    assert page.status_code == 200 and "Embedding models on documents" in page.text
    assert "script-src" in page.headers["content-security-policy"]
    assert client.get(f"/api/v1/runs/{run['run_id']}/report?format=json").json() == run["result"]
    listed = client.get("/api/v1/bench/runs").json()["runs"]
    assert [r["run_id"] for r in listed] == [run["run_id"]]


def test_bench_samples_a_jobs_source(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    body = {
        "documents": {"type": "job", "job": "docs"},
        "models": ["hash/8"],
        "queries": [{"query": "text 1", "relevant": ["1", "2"]}],
        "cache": False,
    }
    run = finished(client, bench(client, body).json()["run_id"])
    assert run["state"] == "completed", run
    assert run["result"]["source"] == "docs_v1"
    assert run["result"]["documents"] == 5
    assert run["result"]["query_source"] == "labeled"


def test_bench_needs_allow_remote_and_valid_options(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    inline = {"type": "documents", "documents": documents(20)}
    remote = bench(client, {"documents": inline, "models": ["openai/text-embedding-3-small"]})
    assert remote.status_code == 422 and remote.json()["code"] == "consent.required"
    assert "https://api.openai.com" in remote.json()["detail"]
    writer = bench(client, {"documents": inline, "generate_queries": "openai/gpt-4o-mini"})
    assert writer.json()["code"] == "consent.required"
    keyed = bench(client, {"documents": inline, "models": ["compat/m,url=http://h/v1?k=sk-x"]})
    assert keyed.status_code == 422 and "sk-x" not in keyed.text
    both = bench(
        client,
        {
            "documents": inline,
            "queries": [{"query": "q", "relevant": ["1"]}],
            "generate_queries": "ollama/llama3",
        },
    )
    assert both.json()["code"] == "options.invalid"
    for documents_ in (
        {"type": "pgvector", "dsn_env": "HOME"},
        {"type": "qdrant", "url_env": "PATH"},
        {"type": "pgvector"},
    ):
        response = bench(client, {"documents": documents_})
        assert response.status_code in (409, 422), response.text
        assert response.json()["code"] in {"options.invalid", "env.missing"}
    assert bench(client, {"documents": {"type": "job", "job": "nope"}}).status_code == 404
    assert client.get("/api/v1/bench/runs").json() == {"runs": []}, "nothing started"


def test_a_bench_where_every_model_fails(
    jobs: Path, fake: FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = api(jobs)
    body = {
        "documents": {"type": "documents", "documents": documents(30)},
        "models": ["openai/text-embedding-3-small"],
        "allow_remote": True,
        "num_queries": 10,
    }
    run = finished(client, bench(client, body).json()["run_id"])
    assert run["state"] == "failed"
    assert run["problem"]["code"] == "model.failed"
    assert "OPENAI_API_KEY" in run["result"]["models"][0]["error"]
    assert client.get(f"/api/v1/runs/{run['run_id']}/report").status_code == 200


def test_one_bench_at_a_time(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    release = hold(client, None)
    try:
        response = bench(client, {"documents": {"type": "documents", "documents": documents(20)}})
        assert response.status_code == 409
        assert response.json()["detail"].startswith("A apply is already running")
    finally:
        release.set()


def test_a_bench_run_survives_a_restart(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    body = {"documents": {"type": "documents", "documents": documents(20)}, "num_queries": 5}
    run = finished(client, bench(client, body).json()["run_id"])
    again = api(jobs)
    assert again.get(f"/api/v1/runs/{run['run_id']}").json()["job"] is None
    assert [r["run_id"] for r in again.get("/api/v1/bench/runs").json()["runs"]] == [run["run_id"]]
