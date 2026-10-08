"""``vecshift apply`` against a real PostgreSQL, with a local stand-in for the model API."""

import asyncio
import hashlib
import json
import math
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

from tests.integration.test_pgvector_doctor import create_healthy
from tests.integration.test_pgvector_plan import job_file
from vecshift.cli import app
from vecshift.connectors.pgvector.writer import AlreadyRunning, Busy, Layout, PgWriter
from vecshift.migrate import JobState, apply

runner = CliRunner()
DIMS = 8


def vector(text: str, dims: int = DIMS) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = [digest[i] - 127.5 for i in range(dims)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


class _Model(BaseHTTPRequestHandler):
    reject = "REJECT-ME"

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        texts = body["input"]
        if any(self.reject in t for t in texts):
            self.send_response(400)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        dims = int(body.get("dimensions") or DIMS)
        data = json.dumps(
            {
                "data": [{"index": i, "embedding": vector(t, dims)} for i, t in enumerate(texts)],
                # Twice what vecshift estimates from length, as real tokenizers can be.
                "usage": {"prompt_tokens": sum(max(1, len(t) // 2) for t in texts)},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def model_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


def apply_job(
    tmp_path: Path, model_url: str, budget: str = "# budget_usd: 50", batch: int = 64
) -> Path:
    spec = f"compat/fake,url={model_url},dims={DIMS},price=1,batch={batch}"
    return job_file(tmp_path, spec, **{"# budget_usd: 50": budget})


def run_apply(path: Path, dsn: str, *args: str, input: str | None = None) -> Any:
    return runner.invoke(app, ["apply", str(path), *args], env={"VECSHIFT_DSN": dsn}, input=input)


def json_lines(output: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def mismatched(conn: psycopg.Connection, column: str = "embedding_v2") -> int:
    """Rows whose new vector is missing or doesn't match their current text."""
    rows = conn.execute(f"SELECT content, {column}::text FROM public.documents").fetchall()
    bad = 0
    for text, stored in rows:
        if stored is None:
            bad += 1
            continue
        values = [float(x) for x in stored.strip("[]").split(",")]
        if any(abs(a - b) > 1e-5 for a, b in zip(values, vector(text), strict=True)):
            bad += 1
    return bad


def test_apply_end_to_end(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, model_url: str
) -> None:
    create_healthy(setup, rows=300)
    setup.execute("ANALYZE public.documents")
    path = apply_job(tmp_path, model_url)
    result = run_apply(path, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    events = json_lines(result.stdout)
    assert events[-1]["event"] == "result" and events[-1]["status"] == "complete"
    assert events[-1]["rows_written"] == 300 and events[-1]["index"] == "built"
    assert {"column", "trigger", "start", "batch", "pass", "index"} <= {e["event"] for e in events}
    assert mismatched(setup) == 0

    index = setup.execute(
        "SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) FROM pg_index i "
        "JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = %s",
        ("documents_embedding_v2_hnsw_idx",),
    ).fetchone()
    assert index is not None and index[0] and "extensions.vector_cosine_ops" in index[1]
    assert setup.execute(
        "SELECT 1 FROM pg_trigger WHERE tgname = 'vecshift_sync_embedding_v2'"
    ).fetchone()

    state = json.loads((tmp_path / ".vecshift" / "documents-reembed.state.json").read_text())
    assert state["rows_written"] == 300 and state["runs"] == 1
    assert "chunk" not in json.dumps(state), "the state file never holds text"

    # The trigger keeps the column in sync: edited rows lose their vector and come back.
    setup.execute("UPDATE public.documents SET content = content || ' edited' WHERE id <= 20")
    setup.execute("UPDATE public.documents SET metadata = '{}' WHERE id BETWEEN 21 AND 30")
    setup.execute("INSERT INTO public.documents (content, metadata) VALUES ('fresh', '{}')")
    assert mismatched(setup) == 21
    again = run_apply(path, db_dsn, "--yes", "--json")
    assert again.exit_code == 0, again.output
    final = json_lines(again.stdout)[-1]
    assert final["rows_written"] == 21 and final["index"] == "exists"
    assert mismatched(setup) == 0

    idle = json_lines(run_apply(path, db_dsn, "--yes", "--json").stdout)[-1]
    assert idle["rows_written"] == 0 and idle["tokens"] == 0


def test_apply_needs_confirmation(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, model_url: str
) -> None:
    create_healthy(setup, rows=10)
    path = apply_job(tmp_path, model_url)
    result = run_apply(path, db_dsn)
    assert result.exit_code == 2 and "--yes" in result.output
    columns = setup.execute(
        "SELECT 1 FROM pg_attribute WHERE attrelid = 'public.documents'::regclass "
        "AND attname = 'embedding_v2'"
    ).fetchone()
    assert columns is None, "nothing changes without confirmation"


def test_budget_stops_and_resumes(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, model_url: str
) -> None:
    create_healthy(setup, rows=200)
    setup.execute("UPDATE public.documents SET content = repeat('word ', 2000)")
    setup.execute("ANALYZE public.documents")
    # vecshift estimates 2,500 tokens a row ($0.50 in all at $1 per million), but the
    # model bills twice that, so the run has to stop itself at the budget.
    path = apply_job(tmp_path, model_url, "budget_usd: 0.1", batch=5)
    blocked = run_apply(path, db_dsn, "--yes", "--json")
    assert blocked.exit_code == 1, "the plan rejects a budget below the estimate"
    path.write_text(path.read_text().replace("budget_usd: 0.1", "budget_usd: 0.6"))
    stopped = run_apply(path, db_dsn, "--yes", "--json")
    assert stopped.exit_code == 3, stopped.output
    first = json_lines(stopped.stdout)[-1]
    assert first["status"] == "budget" and first["rows_written"] == 120
    assert first["total_spent_usd"] <= 0.6 + 1e-9

    path.write_text(path.read_text().replace("budget_usd: 0.6", "budget_usd: 5"))
    done = json_lines(run_apply(path, db_dsn, "--yes", "--json").stdout)[-1]
    assert done["status"] == "complete"
    assert first["rows_written"] + done["rows_written"] == 200, "no row is embedded twice"
    assert done["total_spent_usd"] == pytest.approx(
        first["spent_usd"] + done["spent_usd"], rel=1e-6
    )


def test_rejected_rows_are_recorded(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, model_url: str
) -> None:
    create_healthy(setup, rows=50)
    setup.execute("UPDATE public.documents SET content = 'REJECT-ME' WHERE id = 17")
    path = apply_job(tmp_path, model_url)
    result = run_apply(path, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    final = json_lines(result.stdout)[-1]
    assert final["rows_written"] == 49 and final["rows_failed"] == 1
    state = json.loads((tmp_path / ".vecshift" / "documents-reembed.state.json").read_text())
    assert list(state["failed"]) == ["17"]
    assert "REJECT-ME" not in json.dumps(state)


# --- The writer, directly


def layout(target: str = "embedding_v2") -> Layout:
    return Layout(
        schema="public",
        table="documents",
        pk="id",
        text="content",
        target=target,
        vector_type="vector",
        dims=DIMS,
        extension_schema="extensions",
    )


class LocalModel:
    def __init__(self, before: Any = None) -> None:
        self.tokens = 0
        self.before = before

    async def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        if self.before:
            self.before(texts)
        self.tokens += len(texts)
        return [vector(t) for t in texts]


def run_engine(writer: PgWriter, model: LocalModel, tmp_path: Path, **kwargs: Any) -> Any:
    state = JobState.for_job(tmp_path / "j.yaml", "t")
    options: dict[str, Any] = {
        "dims": DIMS,
        "price_per_million": None,
        "budget_usd": None,
        "chunk_rows": 25,
        "index": ("hnsw", "cosine"),
    }
    options.update(kwargs)
    return asyncio.run(apply(writer, model, state, **options))


def test_writes_during_the_backfill_are_never_lost(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path
) -> None:
    """Edits and inserts that race the backfill still end with vectors of the current text."""
    create_healthy(setup, rows=200)
    calls = 0

    def app_writes(texts: Sequence[str]) -> None:
        # Runs between reading a batch and writing its vectors, like a busy application.
        nonlocal calls
        calls += 1
        if calls == 2:
            # A row in this very batch, rows already done, and rows not reached yet.
            setup.execute("UPDATE public.documents SET content = 'raced ' || id WHERE id = 30")
            setup.execute("UPDATE public.documents SET content = 'later ' || id WHERE id < 10")
            setup.execute("UPDATE public.documents SET content = 'ahead ' || id WHERE id > 190")
            setup.execute("INSERT INTO public.documents (content, metadata) VALUES ('new', '{}')")
        if calls == 4:
            setup.execute("DELETE FROM public.documents WHERE id = 120")

    with psycopg.connect(db_dsn, autocommit=True) as conn:
        result = run_engine(PgWriter(conn, layout()), LocalModel(app_writes), tmp_path)
    assert result.status == "complete", result.message
    assert mismatched(setup) == 0


def test_second_apply_is_refused(db_dsn: str, setup: psycopg.Connection) -> None:
    create_healthy(setup, rows=5)
    with (
        psycopg.connect(db_dsn, autocommit=True) as one,
        psycopg.connect(db_dsn, autocommit=True) as two,
    ):
        first, second = PgWriter(one, layout()), PgWriter(two, layout())
        first.acquire()
        with pytest.raises(AlreadyRunning):
            second.acquire()
        PgWriter(two, layout("other_column")).acquire()  # a different column is fine
        first.release()
        second.acquire()


def test_gives_up_on_a_locked_table(db_dsn: str, setup: psycopg.Connection) -> None:
    """A schema change waits a moment for its lock, then backs off instead of queueing."""
    create_healthy(setup, rows=5)
    sleeps: list[float] = []
    with (
        psycopg.connect(db_dsn) as blocker,
        psycopg.connect(db_dsn, autocommit=True) as conn,
    ):
        blocker.execute("LOCK TABLE public.documents IN ACCESS SHARE MODE")
        writer = PgWriter(conn, layout(), lock_retries=3, lock_timeout_ms=100, sleep=sleeps.append)
        with pytest.raises(Busy, match="add the new column"):
            writer.ensure_column()
        assert sleeps == [1.0, 2.0]
        blocker.rollback()
        assert writer.ensure_column() is True
        assert writer.ensure_column() is False


def test_invalid_index_is_rebuilt(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    create_healthy(setup, rows=50)
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        writer = PgWriter(conn, layout())
        assert run_engine(writer, LocalModel(), tmp_path).index == "built"
        # What a failed CREATE INDEX CONCURRENTLY leaves behind.
        setup.execute(
            "UPDATE pg_index SET indisvalid = false "
            "WHERE indexrelid = 'public.documents_embedding_v2_hnsw_idx'::regclass"
        )
        assert writer.index_state("hnsw") == "invalid"
        assert run_engine(writer, LocalModel(), tmp_path).index == "rebuilt"
        assert writer.index_state("hnsw") == "valid"


def test_awkward_names_are_quoted(db_dsn: str, setup: psycopg.Connection, tmp_path: Path) -> None:
    setup.execute('CREATE TABLE public."Order Items" ("select" bigint PRIMARY KEY, "Text" text)')
    setup.execute(
        """INSERT INTO public."Order Items" SELECT g, 'item ' || g FROM generate_series(1, 30) g"""
    )
    lay = Layout(
        schema="public",
        table="Order Items",
        pk="select",
        text="Text",
        target="order",
        vector_type="halfvec",
        dims=DIMS,
        extension_schema="extensions",
    )
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        result = run_engine(PgWriter(conn, lay), LocalModel(), tmp_path, index=("ivfflat", "l2"))
    assert result.status == "complete" and result.rows_written == 30
    filled = setup.execute('SELECT count("order") FROM public."Order Items"').fetchone()
    assert filled == (30,)
    setup.execute("""UPDATE public."Order Items" SET "Text" = 'changed' WHERE "select" = 1""")
    assert setup.execute(
        'SELECT "order" FROM public."Order Items" WHERE "select" = 1'
    ).fetchone() == (None,)
