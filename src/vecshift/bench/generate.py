"""Generate realistic search queries for documents with an LLM.

Any OpenAI-compatible chat endpoint works. The generated queries can be saved and reused,
so a benchmark only pays for them once.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence

import httpx

from vecshift.bench.corpus import Document, Query
from vecshift.embeddings.providers import CONCURRENCY, MAX_ATTEMPTS, RETRY_STATUSES, api_key
from vecshift.embeddings.spec import ModelSpec

PROMPT = """Write one search query that a person might type to find the passage below.
The passage must answer the query. Don't copy long phrases from it: use the words a
searcher would use. Reply with the query only, on one line.

Passage:
{text}"""


class GenerationError(Exception):
    """Query generation failed. The message never contains credentials."""


class QueryGenerator:
    def __init__(self, spec: ModelSpec, client: httpx.AsyncClient | None = None) -> None:
        if spec.provider == "hash" or not spec.url:
            raise GenerationError("Query generation needs a chat model, such as openai/gpt-4o-mini")
        key = api_key(spec, GenerationError)
        self.spec = spec
        self.tokens = 0
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = client or httpx.AsyncClient(
            timeout=120, headers=headers, verify=True, follow_redirects=False
        )
        if client is not None and key:
            self._client.headers.update(headers)
        self._owns_client = client is None
        self._limit = asyncio.Semaphore(CONCURRENCY)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _one(self, doc: Document, language: str | None = None) -> Query | None:
        prompt = PROMPT.format(text=doc.text[:4000])
        if language:
            prompt = prompt.replace(
                "Reply with the query only", f"Write the query in {language}. Reply with it only"
            )
        payload = {
            "model": self.spec.model,
            "temperature": 0,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }
        delay = 1.0
        async with self._limit:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    r = await self._client.post(f"{self.spec.url}/chat/completions", json=payload)
                except httpx.TransportError as exc:
                    if attempt == MAX_ATTEMPTS:
                        raise GenerationError(
                            f"{self.spec.name}: couldn't reach {self.spec.url}"
                        ) from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
                    await asyncio.sleep(delay + random.uniform(0, 0.5))
                    delay *= 2
                    continue
                if r.status_code >= 400:
                    raise GenerationError(f"{self.spec.name}: HTTP {r.status_code}")
                body = r.json()
                self.tokens += int((body.get("usage") or {}).get("total_tokens") or 0)
                text = str(body["choices"][0]["message"]["content"] or "")
                line = text.strip().splitlines()[0] if text.strip() else ""
                line = line.strip().strip('"“”').strip()
                return Query(line, frozenset({doc.id})) if line else None
        return None  # pragma: no cover

    async def generate(
        self,
        docs: Sequence[Document],
        n: int,
        seed: int,
        language: Callable[[Document], str | None] | None = None,
    ) -> list[Query]:
        """One query for each of ``n`` sampled documents.

        ``language`` picks the language to write each query in, for cross-language tests
        such as an Arabic question about an English passage.
        """
        chosen = random.Random(seed).sample(list(docs), min(n, len(docs)))
        results = await asyncio.gather(
            *(self._one(d, language(d) if language else None) for d in chosen)
        )
        return [q for q in results if q is not None]
