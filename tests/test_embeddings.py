import asyncio
import json
import math
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import httpx
import pytest

from vecshift.embeddings import (
    CachedEmbedder,
    EmbeddingCache,
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatEmbedder,
    SpecError,
    parse_spec,
    providers,
)

SECRET = "sk-test-not-a-real-key"
T = TypeVar("T")


def test_openai_preset() -> None:
    spec = parse_spec("openai/text-embedding-3-small")
    assert spec.url == "https://api.openai.com/v1"
    assert spec.key_env == "OPENAI_API_KEY"
    assert spec.price == 0.02
    assert not spec.is_local
    assert spec.name == "openai/text-embedding-3-small"


def test_options() -> None:
    spec = parse_spec("openai/text-embedding-3-large, dims=256, price=0.5, batch=16")
    assert (spec.dimensions, spec.price, spec.batch_size) == (256, 0.5, 16)
    assert spec.name == "openai/text-embedding-3-large (256d)"


def test_local_providers() -> None:
    assert parse_spec("ollama/nomic-embed-text").is_local
    assert parse_spec("compat/m,url=http://127.0.0.1:8080/v1").is_local
    assert not parse_spec("compat/m,url=https://api.example.com/v1").is_local
    hashed = parse_spec("hash/512")
    assert (hashed.dimensions, hashed.price, hashed.is_local, hashed.name) == (
        512,
        0.0,
        True,
        "hash/512",
    )


def test_model_names_may_contain_slashes() -> None:
    spec = parse_spec("compat/BAAI/bge-m3,url=http://localhost:8080/v1")
    assert spec.model == "BAAI/bge-m3"
    assert spec.price is None  # unknown unless given


@pytest.mark.parametrize(
    ("raw", "query", "doc"),
    [
        ("ollama/nomic-embed-text", "search_query: ", "search_document: "),
        ("compat/intfloat/e5-base-v2,url=http://x:1/v1", "query: ", "passage: "),
        ("compat/intfloat/multilingual-e5-large,url=http://x:1/v1", "query: ", "passage: "),
        ("compat/BAAI/bge-small-en-v1.5,url=http://x:1/v1", "Represent", ""),
        ("openai/text-embedding-3-small", "", ""),
        ("ollama/nomic-embed-text,query_prefix=,doc_prefix=", "", ""),
    ],
)
def test_known_prefixes(raw: str, query: str, doc: str) -> None:
    spec = parse_spec(raw)
    assert spec.query_prefix.startswith(query) and (bool(query) == bool(spec.query_prefix))
    assert spec.doc_prefix == doc


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("text-embedding-3-small", "provider/model"),
        ("acme/model", "Unknown provider"),
        ("compat/m", "url="),
        ("compat/m,url=ftp://x", "http"),
        ("openai/m,dims=0", "positive"),
        ("openai/m,dims=abc", "number"),
        ("openai/m,colour=blue", "Unknown option"),
        ("hash/big", "number"),
    ],
)
def test_bad_specs(raw: str, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        parse_spec(raw)


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_hashing_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(parse_spec("hash/128"))
    a, b, c = run(embedder.embed(["The cat sat", "the CAT sat", "dogs bark loudly"]))
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert sum(x * y for x, y in zip(a, c, strict=True)) < 0.5
    assert embedder.tokens == 9
    assert embedder.fingerprint.model_tag.startswith("hash/128@128#")


def client_with(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def embedding_response(request: httpx.Request, tokens: int | None = 3) -> httpx.Response:
    body = json.loads(request.content)
    data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])]
    data.reverse()  # servers may return items out of order
    usage = {"prompt_tokens": tokens} if tokens is not None else None
    return httpx.Response(200, json={"data": data, "usage": usage})


def test_openai_compat_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return embedding_response(request)

    spec = parse_spec("openai/text-embedding-3-large,dims=2,batch=2")
    embedder = OpenAICompatEmbedder(spec, client=client_with(handler))
    vectors = run(embedder.embed(["a", "bb", "ccc"], "document"))

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert len(seen) == 2  # batches of two
    first = json.loads(seen[0].content)
    assert first == {"model": "text-embedding-3-large", "input": ["a", "bb"], "dimensions": 2}
    assert seen[0].url == "https://api.openai.com/v1/embeddings"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    assert embedder.tokens == 6 and not embedder.tokens_estimated
    assert embedder.dimensions == 2


def test_prefixes_follow_mode() -> None:
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["input"])
        return embedding_response(request)

    embedder = OpenAICompatEmbedder(
        parse_spec("ollama/nomic-embed-text"), client=client_with(handler)
    )
    run(embedder.embed(["x"], "query"))
    run(embedder.embed(["x"], "document"))
    assert sent == [["search_query: x"], ["search_document: x"]]


def test_retries_rate_limits_then_succeeds() -> None:
    calls = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, headers={"retry-after": "2"}, json={"error": "slow down"})
        return embedding_response(request)

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    embedder = OpenAICompatEmbedder(
        parse_spec("compat/m,url=http://localhost:1/v1"),
        client=client_with(handler),
        sleep=fake_sleep,
    )
    assert run(embedder.embed(["hi"])) == [[2.0, 1.0]]
    assert calls == 3
    assert len(waits) == 2 and all(2 <= w <= 2.5 for w in waits)


def test_errors_are_clean_and_never_leak_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})

    embedder = OpenAICompatEmbedder(
        parse_spec("openai/text-embedding-3-small"), client=client_with(handler)
    )
    with pytest.raises(EmbeddingError) as exc:
        run(embedder.embed(["hi"]))
    assert "HTTP 401" in str(exc.value) and "Incorrect API key" in str(exc.value)
    assert SECRET not in str(exc.value)


def test_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "MAX_ATTEMPTS", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def no_sleep(_: float) -> None:
        return None

    embedder = OpenAICompatEmbedder(
        parse_spec("compat/m,url=http://localhost:1/v1"),
        client=client_with(handler),
        sleep=no_sleep,
    )
    with pytest.raises(EmbeddingError, match="couldn't reach"):
        run(embedder.embed(["hi"]))


def test_missing_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        OpenAICompatEmbedder(parse_spec("openai/text-embedding-3-small"))


def test_wrong_vector_count_and_missing_usage() -> None:
    def short(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAICompatEmbedder(
        parse_spec("compat/m,url=http://localhost:1/v1"), client=client_with(short)
    )
    with pytest.raises(EmbeddingError, match="got 1 vectors"):
        run(embedder.embed(["a", "b"]))
    assert run(embedder.embed(["abcdefgh"])) == [[1.0]]
    assert embedder.tokens_estimated


def test_cache_skips_known_texts(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["input"])
        return embedding_response(request)

    spec = parse_spec("compat/m,url=http://localhost:1/v1")
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    first = CachedEmbedder(OpenAICompatEmbedder(spec, client=client_with(handler)), cache)
    assert run(first.embed(["a", "bb"])) == [[1.0, 1.0], [2.0, 1.0]]
    assert run(first.embed(["bb", "ccc"])) == [[2.0, 1.0], [3.0, 1.0]]
    assert calls == [["a", "bb"], ["ccc"]]
    assert (first.hits, first.misses) == (1, 3)

    cache.close()
    reopened = EmbeddingCache(tmp_path / "c.sqlite")
    second = CachedEmbedder(OpenAICompatEmbedder(spec, client=client_with(handler)), reopened)
    run(second.embed(["a"]))
    run(second.embed(["a"], "query"))  # a different mode is a different vector
    assert calls[-1] == ["a"] and len(calls) == 3
