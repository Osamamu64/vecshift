"""Runs over HTTP, against a fake Qdrant: the whole migration, live events, confirmation,
one run per job, stopping, reports, and what survives a restart."""

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.test_qdrant import FakeQdrant
from vecshift.cli import app as cli
from vecshift.connectors import qdrant
from vecshift.server import ServerSettings, create_app
from vecshift.server.runs import Run

JOB = """\
name: docs
source: {type: qdrant, collection: docs}
target: {type: qdrant, alias: live}
model: hash/8
"""
POST = {"X-Vecshift": "1"}
REAL_CLIENT = qdrant.QdrantClient


def points(n: int) -> list[dict[str, Any]]:
    return [
        {"id": i, "vector": [float(i % 3 + 1), 1.0], "payload": {"page_content": f"text {i}"}}
        for i in range(1, n + 1)
    ]


def fresh_qdrant(monkeypatch: pytest.MonkeyPatch) -> FakeQdrant:
    server = FakeQdrant()
    server.add("docs", {"size": 2, "distance": "Cosine"}, points(30))

    def make(settings: qdrant.QdrantSettings, **kwargs: Any) -> qdrant.QdrantClient:
        return REAL_CLIENT(settings, transport=httpx.MockTransport(server))

    monkeypatch.setattr(qdrant, "QdrantClient", make)
    return server


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeQdrant:
    for name in ("VECSHIFT_DSN", "DATABASE_URL", "QDRANT_API_KEY", "VECSHIFT_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VECSHIFT_QDRANT_URL", "http://localhost:6333")
    return fresh_qdrant(monkeypatch)


@pytest.fixture
def jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "docs.yaml").write_text(JOB, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def api(jobs: Path) -> TestClient:
    return TestClient(create_app(ServerSettings(jobs=jobs)), base_url="http://localhost")


def start(client: TestClient, body: dict[str, Any], job: str = "docs") -> str:
    response = client.post(f"/api/v1/jobs/{job}/runs", json=body, headers=POST)
    assert response.status_code == 202, response.json()
    run_id: str = response.json()["run_id"]
    return run_id


def finished(client: TestClient, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        run: dict[str, Any] = client.get(f"/api/v1/runs/{run_id}").json()
        if run["state"] not in {"queued", "running", "stopping"}:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} didn't finish")


def sse(text: str) -> list[dict[str, Any]]:
    """Parse server-sent events: [{id, event, data}]."""
    events = []
    for block in text.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        if fields:
            events.append(
                {
                    "id": int(fields["id"]),
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return events


def run(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    return finished(client, start(client, body))


def timeless(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "seconds"}


# --- a whole migration


def test_a_migration_over_http_matches_the_cli(
    jobs: Path, fake: FakeQdrant, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: Any
) -> None:
    client = api(jobs)
    flow = [
        ({"action": "apply", "confirm": {"apply": True}}, ["apply", "--yes"]),
        ({"action": "cutover", "confirm": {"switch": True}}, ["cutover", "--yes"]),
        ({"action": "rollback", "confirm": {"switch": True}}, ["rollback", "--yes"]),
        ({"action": "cutover", "confirm": {"switch": True}}, ["cutover", "--yes"]),
        ({"action": "cleanup", "confirm": {"name": "docs"}}, ["cleanup", "--yes"]),
    ]
    over_http = []
    for body, _ in flow:
        result = run(client, body)
        assert result["state"] == "completed", result
        over_http.append(timeless(result["result"]))
    assert "docs" not in fake.collections, "cleanup deleted the old collection"
    assert dict(fake.aliases)["live"] == "live_v2"

    # The same steps through the CLI, on a fresh store, in a fresh folder.
    fresh_qdrant(monkeypatch)
    folder = tmp_path_factory.mktemp("cli")
    (folder / "docs.yaml").write_text(JOB, encoding="utf-8")
    monkeypatch.chdir(folder)
    through_cli = []
    for _, args in flow:
        out = CliRunner().invoke(cli, ["--no-env-file", *args, "docs.yaml", "--json"])
        assert out.exit_code == 0, out.output
        last = json.loads(out.stdout.strip().splitlines()[-1])
        last.pop("event", None)
        through_cli.append(timeless(last))
    assert over_http == through_cli


def test_events_stream_and_resume(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    run_id = start(client, {"action": "apply", "confirm": {"apply": True}})
    stream = client.get(f"/api/v1/runs/{run_id}/events")  # ends when the run does
    assert stream.headers["content-type"].startswith("text/event-stream")
    events = sse(stream.text)
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "column" and {"start", "batch", "done"} <= set(kinds)
    assert kinds[-1] == "result"
    run_info = client.get(f"/api/v1/runs/{run_id}").json()
    assert events[-1]["data"] == run_info["result"] and run_info["events"] == len(events)

    later = client.get(f"/api/v1/runs/{run_id}/events", headers={"Last-Event-ID": "3"})
    assert sse(later.text) == events[3:]
    assert sse(client.get(f"/api/v1/runs/{run_id}/events?after=5").text) == events[5:]
    runs = client.get("/api/v1/jobs/docs/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [run_id]


def test_runs_are_kept_on_disk_for_their_owner(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    run_id = start(client, {"action": "apply", "confirm": {"apply": True}})
    finished(client, run_id)
    folder = jobs / ".vecshift" / "runs"
    summary = json.loads((folder / f"{run_id}.json").read_text(encoding="utf-8"))
    assert summary["state"] == "completed" and summary["result"]["status"] == "complete"
    lines = (folder / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "result"
    if os.name == "posix":
        for path in folder.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path

    # A new server (after a restart) still knows the run, and streams it from disk.
    again = api(jobs)
    assert again.get(f"/api/v1/runs/{run_id}").json()["state"] == "completed"
    assert len(sse(again.get(f"/api/v1/runs/{run_id}/events").text)) == len(lines)


def test_a_run_cut_short_by_a_restart_is_failed(jobs: Path) -> None:
    folder = jobs / ".vecshift" / "runs"
    folder.mkdir(parents=True)
    run_id = "01JABCDEFGHJKMNPQRSTVWXYZ0"
    summary = {
        "job": "docs",
        "action": "apply",
        "state": "running",
        "started_at": "2026-10-19T10:00:00+00:00",
    }
    (folder / f"{run_id}.json").write_text(json.dumps(summary), encoding="utf-8")
    (folder / f"{run_id}.jsonl").write_text(
        json.dumps({"id": 1, "event": "start", "data": {"pending": 30}}) + '\n{"id": 2, "ev',
        encoding="utf-8",
    )
    client = api(jobs)
    found = client.get(f"/api/v1/runs/{run_id}").json()
    assert found["state"] == "failed" and found["problem"]["code"] == "run.interrupted"
    events = sse(client.get(f"/api/v1/runs/{run_id}/events").text)
    assert [e["event"] for e in events] == ["start", "error"], "the torn line is dropped"


# --- confirmation and consent


@pytest.mark.parametrize(
    "body",
    [
        {"action": "apply"},
        {"action": "apply", "confirm": {"apply": False}},
        {"action": "cutover"},
        {"action": "rollback", "confirm": {"apply": True}},
        {"action": "cleanup"},
        {"action": "cleanup", "confirm": {"name": ""}},
    ],
)
def test_changes_need_their_confirmation(
    jobs: Path, fake: FakeQdrant, body: dict[str, Any]
) -> None:
    client = api(jobs)
    response = client.post("/api/v1/jobs/docs/runs", json=body, headers=POST)
    assert response.status_code == 422 and response.json()["code"] == "confirm.required"
    assert client.get("/api/v1/jobs/docs/runs").json()["runs"] == [], "nothing started"
    assert "live_v2" not in fake.collections


def test_a_check_needs_no_confirmation(jobs: Path, fake: FakeQdrant) -> None:
    result = run(api(jobs), {"action": "cutover", "options": {"check_only": True}})
    assert result["state"] == "completed" and result["result"]["status"] == "blocked"


def test_cleanup_needs_the_right_name(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    run(client, {"action": "apply", "confirm": {"apply": True}})
    run(client, {"action": "cutover", "confirm": {"switch": True}})
    for wrong in ("docs_v2", "DOCS", "live"):
        result = run(client, {"action": "cleanup", "confirm": {"name": wrong}})
        assert result["state"] == "failed" and result["problem"]["code"] == "confirm.mismatch"
    assert "docs" in fake.collections, "nothing deleted"


def test_remote_models_need_allow_remote(
    jobs: Path, fake: FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    (jobs / "docs.yaml").write_text(
        JOB.replace("hash/8", "openai/text-embedding-3-small"), encoding="utf-8"
    )
    client = api(jobs)
    for body in (
        {"action": "apply", "confirm": {"apply": True}},
        {"action": "eval", "options": {"old_model": "hash/8"}},
    ):
        response = client.post("/api/v1/jobs/docs/runs", json=body, headers=POST)
        assert response.status_code == 422 and response.json()["code"] == "consent.required"
        assert "https://api.openai.com/v1" in response.json()["detail"]


def test_options_are_checked(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    for body in (
        {"action": "apply", "confirm": {"apply": True}, "options": {"until": 100}},
        {"action": "apply", "confirm": {"apply": True}, "options": {"unknown": 1}},
        {"action": "bench"},
        {"action": "cleanup", "options": {"check_only": True}, "confirm": {"name": "x"}},
    ):
        response = client.post("/api/v1/jobs/docs/runs", json=body, headers=POST)
        assert response.status_code == 422 and response.json()["code"] == "request.invalid", body


# --- one at a time, and stopping


def blocking(registry: Any, job: str, *, stoppable: bool) -> tuple[Run, threading.Event]:
    release = threading.Event()

    def work(run: Run) -> None:
        while not release.wait(0.01):
            if run.stop_requested:
                run.finish("stopped")
                return

    return registry.start(job, "apply", work, stoppable=stoppable), release


def test_one_run_per_job(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    held, release = blocking(client.app.state.registry, "docs", stoppable=False)  # type: ignore[attr-defined]
    try:
        response = client.post(
            "/api/v1/jobs/docs/runs",
            json={"action": "cutover", "options": {"check_only": True}},
            headers=POST,
        )
        assert response.status_code == 409 and response.json()["code"] == "job.running"
        assert held.run_id in response.json()["detail"]
        stop = client.post(f"/api/v1/runs/{held.run_id}/stop", headers=POST)
        assert stop.status_code == 409 and stop.json()["code"] == "run.not_stoppable"
    finally:
        release.set()
    assert finished(client, held.run_id)["state"] == "completed"
    again = client.post(f"/api/v1/runs/{held.run_id}/stop", headers=POST)
    assert again.status_code == 409 and again.json()["code"] == "run.finished"


def test_stop_asks_apply_to_finish_the_batch(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    held, _ = blocking(client.app.state.registry, "docs", stoppable=True)  # type: ignore[attr-defined]
    response = client.post(f"/api/v1/runs/{held.run_id}/stop", headers=POST)
    assert response.status_code == 202 and response.json()["state"] in {"stopping", "stopped"}
    assert finished(client, held.run_id)["state"] == "stopped"


def test_apply_stops_between_batches(
    jobs: Path, fake: FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real apply, asked to stop as it starts: it ends stopped, and resumes later."""
    from vecshift.server import runs

    real_emit = runs.Run.emit

    def emit(self: Run, kind: str, data: dict[str, Any]) -> None:
        if kind == "start":
            self.stop_requested = True
        real_emit(self, kind, data)

    monkeypatch.setattr(runs.Run, "emit", emit)
    client = api(jobs)
    stopped = run(client, {"action": "apply", "confirm": {"apply": True}})
    assert stopped["state"] == "stopped" and stopped["result"]["status"] == "stopped"
    monkeypatch.setattr(runs.Run, "emit", real_emit)
    resumed = run(client, {"action": "apply", "confirm": {"apply": True}})
    assert resumed["state"] == "completed"


# --- eval


def test_eval_runs_with_a_report(jobs: Path, fake: FakeQdrant) -> None:
    client = api(jobs)
    run(client, {"action": "apply", "confirm": {"apply": True}})
    labeled = [{"query": f"text {i}", "relevant": [str(i)]} for i in range(1, 11)]
    body = {
        "action": "eval",
        "options": {"num_queries": 10, "queries": labeled, "old_model": "hash/8"},
    }
    run_id = start(client, body)
    result = finished(client, run_id)
    assert result["state"] == "completed", result
    assert result["result"]["query_source"] == "labeled"
    events = sse(client.get(f"/api/v1/runs/{run_id}/events").text)
    assert events[0]["event"] == "step" and events[-1]["event"] == "result"
    assert not list((jobs / ".vecshift" / "runs").glob("*.queries.jsonl")), "queries removed"

    page = client.get(f"/api/v1/runs/{run_id}/report")
    assert page.headers["content-type"].startswith("text/html")
    policy = page.headers["content-security-policy"]
    assert "script-src 'sha256-" in policy and policy in page.text, "the report's own policy"
    assert client.get(f"/api/v1/runs/{run_id}/report?format=json").json() == result["result"]
    applied = client.get("/api/v1/jobs/docs/runs").json()["runs"][-1]["run_id"]
    missing = client.get(f"/api/v1/runs/{applied}/report")
    assert missing.status_code == 404 and missing.json()["code"] == "report.missing"


def test_unknown_runs(jobs: Path) -> None:
    client = api(jobs)
    for run_id in ("01JABCDEFGHJKMNPQRSTVWXYZ0", "../../etc", "x"):
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 404, run_id
