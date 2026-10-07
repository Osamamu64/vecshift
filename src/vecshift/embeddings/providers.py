"""Embedding providers that satisfy :class:`vecshift.core.contracts.EmbeddingProvider`."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import httpx

from vecshift.core.fingerprint import EmbeddingFingerprint
from vecshift.embeddings.spec import ModelSpec

EmbedMode = Literal["document", "query"]

RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
CONCURRENCY = 4

_STOPWORDS = (
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with which you your can not but if then than so such these those there their "
    "they them we our may also into when where how what"
)


class EmbeddingError(Exception):
    """A provider call failed for good. The message never contains credentials."""


class _Base:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.tokens = 0
        """Tokens billed so far, as reported by the provider."""
        self.tokens_estimated = False
        """Whether any token count had to be estimated."""
        self.request_seconds: list[float] = []
        self._dims: int | None = spec.dimensions

    @property
    def dimensions(self) -> int | None:
        return self._dims

    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        if not self._dims:
            raise EmbeddingError(f"{self.spec.name}: dimensions are unknown until the first call")
        return EmbeddingFingerprint(
            provider=self.spec.provider,
            model=self.spec.model,
            dimensions=self._dims,
            task=None,
            prefix=(self.spec.query_prefix + "|" + self.spec.doc_prefix) or None,
            normalized=True,
        )

    def _prefixed(self, texts: Sequence[str], mode: EmbedMode) -> list[str]:
        prefix = self.spec.query_prefix if mode == "query" else self.spec.doc_prefix
        return [prefix + t for t in texts] if prefix else list(texts)

    async def aclose(self) -> None:
        return None


class HashingEmbedder(_Base):
    """A lexical baseline: hashed bag of words, L2-normalized.

    Free, offline, and deterministic. It captures word overlap only, so it's a floor
    that real models should beat, and it makes a first run cost nothing.
    """

    _WORD = re.compile(r"\w+", re.UNICODE)
    _STOP = frozenset(_STOPWORDS.split())

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        assert spec.dimensions
        self._dims = spec.dimensions

    def _vector(self, text: str) -> list[float]:
        dims = self._dims or 1
        counts: dict[int, float] = {}
        words = self._WORD.findall(text.lower())
        self.tokens += len(words)
        words = [w for w in words if w not in self._STOP] or words
        for word in words:
            h = hashlib.blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(h[:4], "little") % dims
            sign = 1.0 if h[4] & 1 else -1.0
            counts[index] = counts.get(index, 0.0) + sign
        vec = [0.0] * dims
        for i, c in counts.items():
            vec[i] = math.copysign(1 + math.log(abs(c)), c) if c else 0.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: Sequence[str], mode: EmbedMode = "document") -> list[list[float]]:
        start = time.perf_counter()
        out = [self._vector(t) for t in self._prefixed(texts, mode)]
        self.request_seconds.append(time.perf_counter() - start)
        return out


class OpenAICompatEmbedder(_Base):
    """Any server that speaks OpenAI's ``POST /embeddings``: OpenAI itself, Ollama, vLLM,
    TEI, LM Studio, LiteLLM, and most hosted inference platforms."""

    def __init__(
        self,
        spec: ModelSpec,
        client: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        super().__init__(spec)
        if not spec.url:
            raise EmbeddingError(f"{spec.name}: no URL configured")
        key = os.environ.get(spec.key_env) if spec.key_env else None
        if spec.provider == "openai" and not key:
            raise EmbeddingError(
                f"{spec.name}: set {spec.key_env} to your OpenAI API key "
                "(or key_env=NAME in the spec to read it from another variable)"
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=15), headers=headers
        )
        if client is not None and key:
            self._client.headers.update(headers)
        self._owns_client = client is None
        self._sleep = sleep
        self._limit = asyncio.Semaphore(CONCURRENCY)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _describe(self, response: httpx.Response) -> str:
        try:
            body: Any = response.json()
            message = body.get("error", {})
            message = message.get("message") if isinstance(message, dict) else message
            message = message or body.get("detail") or response.text
        except ValueError:
            message = response.text
        message = " ".join(str(message).split())[:300]
        return f"{self.spec.name}: HTTP {response.status_code}: {message}"

    async def _post(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.spec.model, "input": batch}
        if self.spec.dimensions:
            payload["dimensions"] = self.spec.dimensions
        delay = 1.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                response = await self._client.post(f"{self.spec.url}/embeddings", json=payload)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise EmbeddingError(
                        f"{self.spec.name}: couldn't reach {self.spec.url} ({type(exc).__name__})"
                    ) from exc
                await self._sleep(delay + random.uniform(0, delay / 2))
                delay = min(delay * 2, 30)
                continue
            elapsed = time.perf_counter() - start
            if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                retry_after = response.headers.get("retry-after", "")
                wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
                await self._sleep(min(wait, 60) + random.uniform(0, 0.5))
                delay = min(delay * 2, 30)
                continue
            if response.status_code >= 400:
                raise EmbeddingError(self._describe(response))
            self.request_seconds.append(elapsed)
            return self._parse(response, batch)
        raise EmbeddingError(
            f"{self.spec.name}: gave up after {MAX_ATTEMPTS} attempts"
        )  # pragma: no cover

    def _parse(self, response: httpx.Response, batch: list[str]) -> list[list[float]]:
        try:
            body = response.json()
            items = sorted(body["data"], key=lambda d: d.get("index", 0))
            vectors = [list(map(float, item["embedding"])) for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(f"{self.spec.name}: unexpected response shape") from exc
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"{self.spec.name}: sent {len(batch)} texts, got {len(vectors)} vectors"
            )
        usage = body.get("usage") or {}
        tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
        if isinstance(tokens, int):
            self.tokens += tokens
        else:
            self.tokens += sum(max(1, len(t) // 4) for t in batch)
            self.tokens_estimated = True
        if vectors:
            self._dims = len(vectors[0])
        return vectors

    async def _batch(self, batch: list[str]) -> list[list[float]]:
        async with self._limit:
            return await self._post(batch)

    async def embed(self, texts: Sequence[str], mode: EmbedMode = "document") -> list[list[float]]:
        prepared = self._prefixed(texts, mode)
        size = self.spec.batch_size
        batches = [prepared[i : i + size] for i in range(0, len(prepared), size)]
        results = await asyncio.gather(*(self._batch(b) for b in batches))
        return [vec for batch in results for vec in batch]


Embedder = HashingEmbedder | OpenAICompatEmbedder


def create_embedder(spec: ModelSpec, client: httpx.AsyncClient | None = None) -> Embedder:
    if spec.provider == "hash":
        return HashingEmbedder(spec)
    return OpenAICompatEmbedder(spec, client=client)
