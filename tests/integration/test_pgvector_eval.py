"""``vecshift eval`` against a real PostgreSQL, with Arabic and English rows."""

import hashlib
import json
import math
import random
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest
from typer.testing import CliRunner

from vecshift.cli import app

runner = CliRunner()

_EN_WORDS = (
    "market price growth energy water policy school health travel ocean river forest city "
    "music science history village coffee garden football network memory planet engine bridge"
)
EN = _EN_WORDS.split()
_AR_WORDS = (
    "سوق سعر نمو طاقة ماء سياسة مدرسة صحة سفر محيط نهر غابة مدينة موسيقى علم تاريخ قرية قهوة "
    "حديقة كرة شبكة ذاكرة كوكب محرك"
)
AR = _AR_WORDS.split()


def bag_of_words(text: str, dims: int) -> list[float]:
    """A tiny embedding model: hashed words. Fewer dimensions mean more collisions."""
    vector = [0.0] * dims
    for word in text.lower().replace(".", " ").replace("؟", " ").split():
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "little") % dims] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class _Server(BaseHTTPRequestHandler):
    prompts: ClassVar[list[str]] = []

    def log_message(self, *args: object) -> None:
        pass

    def _send(self, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.endswith("/chat/completions"):
            prompt = body["messages"][0]["content"]
            self.prompts.append(prompt)
            if "in Arabic" in prompt:
                answer = "سؤال عن السوق والمدينة"
            elif "in English" in prompt:
                answer = "a question about the market and the city"
            else:
                answer = " ".join(prompt.split("Passage:")[1].split()[:5])
            self._send({"choices": [{"message": {"content": answer}}], "usage": {}})
            return
        dims = int(body.get("dimensions") or 32)
        texts = body["input"]
        self._send(
            {
                "data": [
                    {"index": i, "embedding": bag_of_words(t, dims)} for i, t in enumerate(texts)
                ],
                "usage": {"prompt_tokens": sum(len(t) // 4 + 1 for t in texts)},
            }
        )


@pytest.fixture
def server() -> Iterator[str]:
    _Server.prompts.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()


def create_bilingual(conn: psycopg.Connection, old_dims: int, documents: int = 90) -> None:
    """Documents of five chunks each; a third of them in Arabic."""
    rng = random.Random(5)
    conn.execute(
        f"CREATE TABLE public.documents (id bigserial PRIMARY KEY, content text, "
        f"metadata jsonb, embedding extensions.vector({old_dims}))"
    )
    rows = []
    for doc in range(documents):
        words = AR if doc % 3 == 0 else EN
        topic = rng.sample(words, 5)
        for _ in range(5):
            sentences = []
            for _ in range(3):
                body = rng.sample(topic, 3) + rng.sample(words, 4)
                rng.shuffle(body)
                sentences.append(" ".join(body) + ("؟" if words is AR else "."))
            text = " ".join(sentences)
            rows.append(
                (text, json.dumps({"document_id": f"doc-{doc}"}), bag_of_words(text, old_dims))
            )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO public.documents (content, metadata, embedding) "
            "VALUES (%s, %s, %s::extensions.vector)",
            [(t, m, str(v)) for t, m, v in rows],
        )
    conn.execute(
        "CREATE INDEX ON public.documents USING hnsw (embedding extensions.vector_cosine_ops)"
    )
    conn.execute("ANALYZE public.documents")


def job(tmp_path: Path, url: str, *, old: int | None, new: int) -> Path:
    old_line = f'  model: "compat/old,url={url},dims={old}"\n' if old else ""
    path = tmp_path / "vecshift.yaml"
    path.write_text(
        "version: 1\nname: eval-test\nsource:\n  table: public.documents\n"
        f"{old_line}target:\n  column: embedding_v2\n"
        f'model: "compat/new,url={url},dims={new},price=1"\n'
    )
    return path


def cli(command: str, path: Path, dsn: str, *args: str) -> Any:
    return runner.invoke(app, [command, str(path), *args], env={"VECSHIFT_DSN": dsn})


def evaluate(path: Path, dsn: str, *args: str) -> tuple[int, dict[str, Any]]:
    result = cli("eval", path, dsn, "--yes", "--json", "--num-queries", "120", *args)
    assert result.exit_code in (0, 1, 3), result.output
    return result.exit_code, json.loads(result.stdout)


def schema_snapshot(conn: psycopg.Connection) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT relname, relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' ORDER BY 1"
    ).fetchall()


def test_better_model_is_go(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=4, new=48)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0
    before = schema_snapshot(setup)

    report = tmp_path / "eval.html"
    code, data = evaluate(path, db_dsn, "--html", str(report))
    assert code == 0 and data["verdict"]["status"] == "go", data["verdict"]
    page = report.read_text()
    assert "GO: the new vectors are ready" in page and "Arabic → arabic" in page
    assert data["mode"] == "queries" and not data["partial"]
    old, new = data["old"], data["new"]
    assert {"all", "arabic→arabic", "latin→latin"} <= set(new["scores"])
    for name in ("arabic→arabic", "latin→latin"):
        assert new["scores"][name]["recall_at_10"] > old["scores"][name]["recall_at_10"]
    assert [p["setting"] for p in new["sweep"]] == [40, 100, 200]
    assert all(p["latency"]["p95_ms"] > 0 and 0 <= p["index_recall"] <= 1 for p in new["sweep"])
    assert old["embed_latency"]["samples"] == 20
    assert new["same_group_at_10"] is not None
    assert schema_snapshot(setup) == before, "eval changes nothing"


def test_worse_model_is_no_go(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=48)
    path = job(tmp_path, server, old=48, new=4)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0
    code, data = evaluate(path, db_dsn)
    assert code == 1 and data["verdict"]["status"] == "no_go"
    assert any("fell from" in r for r in data["verdict"]["reasons"])

    text = cli("eval", path, db_dsn, "--yes", "--num-queries", "60")
    assert "Verdict: NO-GO" in text.output and "arabic→arabic" in text.output


def test_old_model_unavailable(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=None, new=48)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0

    dead = "compat/old,url=http://127.0.0.1:9/v1,dims=4"
    code, data = evaluate(path, db_dsn, "--old-model", dead)
    assert data["mode"] == "vectors" and "couldn't embed" in data["old"]["note"]
    assert data["old"]["sweep"], "latency measured from stored vectors"
    assert data["old"]["same_group_at_10"] is not None
    assert code == 0 and data["verdict"]["status"] == "go"

    wrong = f"compat/old,url={server},dims=16"
    _, data = evaluate(path, db_dsn, "--old-model", wrong)
    assert data["mode"] == "vectors" and "16 dimensions" in data["old"]["note"]


def test_partial_migration(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=4, new=48)
    assert cli("apply", path, db_dsn, "--yes", "--until", "50").exit_code == 3
    code, data = evaluate(path, db_dsn)
    assert data["partial"] and not data["new"]["sweep"]
    assert any("don't have a new vector yet" in n for n in data["notes"])
    assert (
        data["new"]["scores"]["all"]["recall_at_10"] > data["old"]["scores"]["all"]["recall_at_10"]
    )
    assert code == 0


def test_nothing_to_evaluate(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4, documents=6)
    path = job(tmp_path, server, old=4, new=48)
    result = cli("eval", path, db_dsn, "--yes")
    assert result.exit_code == 2 and "Run apply" in result.output
    bad = cli("eval", path, db_dsn, "--yes", "--cross-language")
    assert bad.exit_code == 2 and "--generate-queries" in bad.output


def test_latency_gate(db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=4, new=48)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0
    code, data = evaluate(path, db_dsn, "--max-p95-ms", "0.001")
    assert code == 1 and any("ms limit" in r for r in data["verdict"]["reasons"])


def test_cross_language_queries(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=4, new=48)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0
    chat = f"compat/chat,url={server}"
    _, data = evaluate(path, db_dsn, "--generate-queries", chat, "--cross-language")
    assert data["query_source"] == "cross-language"
    assert {"arabic→latin", "latin→arabic"} <= set(data["new"]["scores"])
    assert any("Write the query in Arabic" in p for p in _Server.prompts)
    assert any("Write the query in English" in p for p in _Server.prompts)


def test_labeled_queries_and_after_cutover(
    db_dsn: str, setup: psycopg.Connection, tmp_path: Path, server: str
) -> None:
    create_bilingual(setup, old_dims=4)
    path = job(tmp_path, server, old=4, new=48)
    assert cli("apply", path, db_dsn, "--yes").exit_code == 0
    rows = setup.execute("SELECT id, content FROM public.documents ORDER BY id LIMIT 40").fetchall()
    labeled = tmp_path / "queries.jsonl"
    labeled.write_text(
        "".join(
            json.dumps({"query": t.split(".")[0].split("؟")[0], "relevant": [i]}) + "\n"
            for i, t in rows
        )
    )
    code, data = evaluate(path, db_dsn, "--queries", str(labeled), "--min-slice", "10")
    assert data["query_source"] == "labeled" and data["queries"] == 40 and code == 0

    assert cli("cutover", path, db_dsn, "--yes").exit_code == 0
    _, after = evaluate(path, db_dsn)
    assert after["old"]["column"] == "embedding_old" and after["new"]["column"] == "embedding"
    assert after["verdict"]["status"] == "go"
