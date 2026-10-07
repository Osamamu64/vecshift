import asyncio
import json
import re
from pathlib import Path

import httpx
import numpy as np
import pytest
from typer.testing import CliRunner

from vecshift.bench import Benchmark, CorpusError, Document, Query, corpus, plan, run
from vecshift.bench.generate import GenerationError, QueryGenerator
from vecshift.bench.html import render_html
from vecshift.bench.metrics import evaluate, normalize
from vecshift.cli import app
from vecshift.embeddings import parse_spec, providers

TEXT = (
    "Vector databases store embeddings for semantic search. "
    "Changing the embedding model means every document must be embedded again. "
    "A shadow index lets the old one keep serving traffic during the change. "
    "Cutover happens by swapping an alias once the new index is verified."
)


def make_docs(n: int = 40) -> list[Document]:
    """Real, varied English paragraphs from Python's bundled documentation."""
    from pydoc_data.topics import topics

    docs = []
    for name, text in sorted(topics.items()):
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            para = " ".join(para.split())
            if len(para.split()) >= 40 and para.count(". ") >= 2:
                docs.append(Document(f"{name}-{i}", para))
    return docs[:n]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_documents_and_queries(tmp_path: Path) -> None:
    docs = write_jsonl(
        tmp_path / "d.jsonl", [{"id": "a", "text": "x"}, {"text": "y"}, {"text": " "}]
    )
    assert [d.id for d in corpus.load_documents(docs)] == ["a", "2"]
    q = write_jsonl(tmp_path / "q.jsonl", [{"query": "find a", "relevant": "a"}])
    assert corpus.load_queries(q) == [Query("find a", frozenset({"a"}))]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "a"}], "text"),
        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "duplicate"),
        ([], "no documents"),
    ],
)
def test_bad_documents(tmp_path: Path, rows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(CorpusError, match=message):
        corpus.load_documents(write_jsonl(tmp_path / "d.jsonl", rows))


def test_bad_json(tmp_path: Path) -> None:
    (tmp_path / "d.jsonl").write_text("{nope\n")
    with pytest.raises(CorpusError, match=":1: not valid JSON"):
        corpus.load_documents(tmp_path / "d.jsonl")


def test_sample_keeps_required_ids() -> None:
    docs = make_docs(50)
    keep = {docs[49].id, docs[3].id}
    picked = corpus.sample(docs, 10, keep, seed=1)
    assert len(picked) == 10 and keep <= {d.id for d in picked}
    assert corpus.sample(docs, 10, set(), 1) == corpus.sample(docs, 10, set(), 1)


def test_proxy_queries_remove_the_sentence() -> None:
    docs = [Document("a", TEXT), *make_docs(5)]
    edited, queries, notes = corpus.proxy_queries(docs, 100, seed=3)
    by_id = {d.id: d for d in edited}
    for q in queries:
        (doc_id,) = q.relevant
        assert q.text not in by_id[doc_id].text
        assert q.text in next(d.text for d in docs if d.id == doc_id) or doc_id != "a"
    assert notes and "proxy" in notes[0]


def test_proxy_queries_skip_repeated_sentences() -> None:
    boiler = "This page is part of the internal handbook and may change."
    docs = [
        Document(f"d{i}", f"{boiler} Item {i} covers topic {i} in depth today.") for i in range(5)
    ]
    _, queries, _ = corpus.proxy_queries(docs, 5, seed=1)
    assert queries and all(boiler not in q.text for q in queries)


def test_proxy_queries_need_long_documents() -> None:
    with pytest.raises(CorpusError, match="too short"):
        corpus.proxy_queries([Document("a", "Tiny.")], 5, seed=1)


def test_metrics() -> None:
    docs = normalize([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    queries = normalize([[1, 0.1, 0], [0.1, 0, 1], [0.6, 0.8, 0]])
    scores = evaluate(docs, queries, [{0}, {1}, {0, 1}])
    # q0 finds doc 0 first; q1 finds doc 1 at rank 3; q2 finds both of its docs at ranks 1-2.
    assert scores.recall_at_1 == pytest.approx((1 + 0 + 0.5) / 3)
    assert scores.recall_at_10 == pytest.approx(1.0)
    assert scores.mrr_at_10 == pytest.approx((1 + 1 / 3 + 1) / 3)
    assert 0 < scores.ndcg_at_10 < 1


def test_run_with_hashing_models() -> None:
    edited, queries, notes = corpus.proxy_queries(make_docs(), 20, seed=2)
    bench = Benchmark(edited, queries, "proxy", notes)
    specs = [parse_spec("hash/16"), parse_spec("hash/2048")]
    result = asyncio.run(run(bench, specs, None, "test"))
    by_name = {m.name: m for m in result.models}
    big, small = by_name["hash/2048"], by_name["hash/16"]
    assert big.scores and small.scores
    assert big.scores.recall_at_10 >= small.scores.recall_at_10
    assert big.dimensions == 2048 and big.gb_per_million_vectors == pytest.approx(8.192)
    assert big.cost_per_million_docs == 0 and big.query_ms_p50 is not None
    ranked = result.to_dict()["models"]
    assert ranked[0]["scores"]["recall_at_10"] >= ranked[1]["scores"]["recall_at_10"]

    item = plan(bench, [parse_spec("openai/text-embedding-3-small")])[0]
    assert item.est_tokens > 0 and item.est_cost == pytest.approx(item.est_tokens * 0.02 / 1e6)


def test_failed_model_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "MAX_ATTEMPTS", 1)
    edited, queries, notes = corpus.proxy_queries(make_docs(), 5, seed=2)
    bench = Benchmark(edited, queries, "proxy", notes)
    specs = [parse_spec("hash/64"), parse_spec("compat/m,url=http://127.0.0.1:9/v1")]
    result = asyncio.run(run(bench, specs, None, "test"))
    ok, failed = result.ranked()
    assert ok.scores and failed.scores is None
    assert failed.error and "couldn't reach" in failed.error


def test_html_leaderboard() -> None:
    edited, queries, notes = corpus.proxy_queries(make_docs(), 10, seed=2)
    bench = Benchmark(edited, queries, "proxy", notes)
    result = asyncio.run(
        run(bench, [parse_spec("hash/32"), parse_spec("hash/512")], None, "<b>evil</b>.jsonl")
    )
    page = render_html(result, version="9.9.9")
    assert "<b>evil</b>" not in page and "&lt;b&gt;evil&lt;/b&gt;" in page
    assert "Leaderboard" in page and page.count('class="pt"') == 4
    assert re.findall(r'\b(?:src|href)="([a-z]+):', page) == ["data"]
    assert "https://" not in page


def test_generator_cleans_replies() -> None:
    replies = iter(['"How do backups work?"\nextra line', "  ", "Which shard?"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": next(replies)}}],
                "usage": {"total_tokens": 5},
            },
        )

    gen = QueryGenerator(
        parse_spec("compat/chat,url=http://localhost:1/v1"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    queries = asyncio.run(gen.generate(make_docs(3), 3, seed=1))
    assert sorted(q.text for q in queries) == ["How do backups work?", "Which shard?"]
    assert gen.tokens == 15
    with pytest.raises(GenerationError):
        QueryGenerator(parse_spec("hash/8"))


runner = CliRunner()


def docs_file(tmp_path: Path) -> Path:
    return write_jsonl(tmp_path / "docs.jsonl", [{"id": d.id, "text": d.text} for d in make_docs()])


def test_cli_free_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    out = tmp_path / "lb.html"
    result = runner.invoke(
        app,
        [
            "bench",
            "--docs",
            str(docs_file(tmp_path)),
            "--num-queries",
            "10",
            "--json",
            "--html",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert [m["name"] for m in data["models"]] and data["query_source"] == "proxy"
    assert out.read_text().startswith("<!doctype html>")


def test_cli_labeled_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    docs = make_docs()
    q = write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"query": docs[0].text[:80], "relevant": [docs[0].id, docs[7].id]},
            {"query": docs[4].text[:80], "relevant": [docs[4].id, "missing"]},
            {"query": "nothing here", "relevant": ["missing"]},
        ],
    )
    result = runner.invoke(
        app,
        [
            "bench",
            "--docs",
            str(docs_file(tmp_path)),
            "--queries",
            str(q),
            "-m",
            "hash/512",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["query_source"] == "labeled" and data["queries"] == 2
    assert any("skipped" in n for n in data["notes"])


def test_cli_refuses_to_send_data_without_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sent: list[object] = []
    monkeypatch.setattr(httpx.AsyncClient, "post", lambda *a, **k: sent.append(a))
    result = runner.invoke(
        app, ["bench", "--docs", str(docs_file(tmp_path)), "-m", "openai/text-embedding-3-small"]
    )
    assert result.exit_code == 2
    assert "Pass --yes" in result.output and "sends the text of" in result.output
    assert sent == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--queries", "q.jsonl", "--generate-queries", "openai/gpt-4o-mini"], "not both"),
        (["--save-queries", "q.jsonl"], "only works with --generate-queries"),
        (["-m", "nope"], "provider/model"),
    ],
)
def test_cli_argument_errors(tmp_path: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["bench", "--docs", str(docs_file(tmp_path)), *args])
    assert result.exit_code == 2 and message in result.output


def test_cli_needs_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECSHIFT_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = runner.invoke(app, ["bench"])
    assert result.exit_code == 2 and "No documents" in result.output


def test_numpy_is_used_for_scoring() -> None:
    assert isinstance(normalize([[3, 4]]), np.ndarray)


def fake_api(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer chat and embedding requests without a network, recording the URLs called."""
    calls: list[str] = []
    replies = iter(f"Which section explains topic {i}?" for i in range(10_000))

    async def post(self: httpx.AsyncClient, url: str, json: dict[str, object]) -> httpx.Response:
        calls.append(url)
        request = httpx.Request("POST", url)
        if url.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": next(replies)}}], "usage": {}},
                request=request,
            )
        inputs = json["input"]
        assert isinstance(inputs, list)
        data = [
            {"index": i, "embedding": [float(len(t) % 7), 1.0, float(len(t) % 3)]}
            for i, t in enumerate(inputs)
        ]
        return httpx.Response(
            200, json={"data": data, "usage": {"prompt_tokens": 10 * len(inputs)}}, request=request
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return calls


def test_cli_generate_save_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls = fake_api(monkeypatch)
    saved = tmp_path / "queries.jsonl"
    docs = docs_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "bench",
            "--docs",
            str(docs),
            "--generate-queries",
            "compat/chat,url=http://127.0.0.1:1/v1",
            "--num-queries",
            "8",
            "--save-queries",
            str(saved),
            "-m",
            "hash/256",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["query_source"] == "generated" and data["queries"] == 8
    assert sum(u.endswith("/chat/completions") for u in calls) == 8
    assert len(corpus.load_queries(saved)) == 8

    reused = runner.invoke(
        app, ["bench", "--docs", str(docs), "--queries", str(saved), "-m", "hash/256", "--json"]
    )
    assert reused.exit_code == 0, reused.output
    assert json.loads(reused.stdout)["query_source"] == "labeled"


def test_cli_paid_model_with_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = fake_api(monkeypatch)
    result = runner.invoke(
        app,
        [
            "bench",
            "--docs",
            str(docs_file(tmp_path)),
            "-m",
            "openai/text-embedding-3-small,batch=16",
            "--num-queries",
            "10",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls and all(u == "https://api.openai.com/v1/embeddings" for u in calls)
    assert "openai/text-embedding-3-small" in result.stdout
    assert "<$0.01" in result.stdout or "$" in result.stdout  # 10 tokens/doc at $0.02/1M
