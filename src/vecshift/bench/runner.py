"""Run a benchmark: embed, search, score, and measure each model."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from vecshift.bench.corpus import Benchmark
from vecshift.bench.metrics import Scores, evaluate, normalize
from vecshift.embeddings import (
    CachedEmbedder,
    EmbeddingCache,
    EmbeddingError,
    ModelSpec,
    create_embedder,
)

LATENCY_SAMPLES = 20
CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class ModelResult:
    name: str
    spec: str
    dimensions: int | None = None
    model_tag: str | None = None
    scores: Scores | None = None
    error: str | None = None
    docs_per_second: float | None = None
    """Documents embedded per second, or ``None`` when every document was cached."""
    query_ms_p50: float | None = None
    query_ms_p95: float | None = None
    tokens_per_doc: float | None = None
    tokens_estimated: bool = False
    price_per_million_tokens: float | None = None
    cached_docs: int = 0
    local: bool = False

    @property
    def cost_per_million_docs(self) -> float | None:
        if self.price_per_million_tokens is None or self.tokens_per_doc is None:
            return None
        return self.tokens_per_doc * self.price_per_million_tokens

    @property
    def gb_per_million_vectors(self) -> float | None:
        """float32 storage for a million vectors, in GB."""
        return self.dimensions * 4 / 1000 if self.dimensions else None

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "scores"}
        data["scores"] = asdict(self.scores) if self.scores else None
        data["cost_per_million_docs"] = self.cost_per_million_docs
        data["gb_per_million_vectors"] = self.gb_per_million_vectors
        return data


@dataclass(slots=True)
class BenchResult:
    source: str
    documents: int
    queries: int
    query_source: str
    notes: list[str] = field(default_factory=list)
    models: list[ModelResult] = field(default_factory=list)

    def ranked(self) -> list[ModelResult]:
        """Best recall@10 first; failed models last."""
        return sorted(
            self.models,
            key=lambda m: (m.scores is None, -(m.scores.recall_at_10 if m.scores else 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "documents": self.documents,
            "queries": self.queries,
            "query_source": self.query_source,
            "notes": self.notes,
            "models": [m.to_dict() for m in self.ranked()],
        }


@dataclass(frozen=True, slots=True)
class PlanItem:
    spec: ModelSpec
    est_tokens: int
    est_cost: float | None


def plan(bench: Benchmark, specs: Sequence[ModelSpec]) -> list[PlanItem]:
    """Estimated tokens and cost per model, before anything is sent."""
    chars = sum(len(d.text) for d in bench.documents) + sum(len(q.text) for q in bench.queries)
    chars += sum(len(q.text) for q in bench.queries[:LATENCY_SAMPLES])
    tokens = chars // CHARS_PER_TOKEN
    return [
        PlanItem(s, tokens, None if s.price is None else tokens * s.price / 1_000_000)
        for s in specs
    ]


def _percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


async def _run_one(bench: Benchmark, spec: ModelSpec, cache: EmbeddingCache | None) -> ModelResult:
    result = ModelResult(
        name=spec.name, spec=spec.raw, price_per_million_tokens=spec.price, local=spec.is_local
    )
    embedder = create_embedder(spec)
    use_cache = cache is not None and spec.provider != "hash"
    wrapped = CachedEmbedder(embedder, cache) if use_cache and cache else None
    try:
        doc_texts = [d.text for d in bench.documents]
        start = time.perf_counter()
        doc_vecs = await (wrapped or embedder).embed(doc_texts, "document")
        elapsed = time.perf_counter() - start
        fresh = wrapped.misses if wrapped else len(doc_texts)
        result.cached_docs = len(doc_texts) - fresh
        if fresh:
            result.docs_per_second = fresh / elapsed if elapsed > 0 else None
            result.tokens_per_doc = embedder.tokens / fresh
            result.tokens_estimated = embedder.tokens_estimated
        else:
            result.tokens_per_doc = (
                sum(len(t) for t in doc_texts) / len(doc_texts) / CHARS_PER_TOKEN
            )
            result.tokens_estimated = True

        query_texts = [q.text for q in bench.queries]
        query_vecs = await (wrapped or embedder).embed(query_texts, "query")

        timings = []
        for text in query_texts[:LATENCY_SAMPLES]:
            t0 = time.perf_counter()
            await embedder.embed([text], "query")
            timings.append((time.perf_counter() - t0) * 1000)
        if timings:
            result.query_ms_p50 = statistics.median(timings)
            result.query_ms_p95 = _percentile(timings, 95)

        index = {d.id: i for i, d in enumerate(bench.documents)}
        relevant = [{index[r] for r in q.relevant if r in index} for q in bench.queries]
        result.scores = evaluate(normalize(doc_vecs), normalize(query_vecs), relevant)
        result.dimensions = len(doc_vecs[0])
        result.model_tag = embedder.fingerprint.model_tag
    except EmbeddingError as exc:
        result.error = str(exc)
    finally:
        await embedder.aclose()
    return result


async def run(
    bench: Benchmark,
    specs: Sequence[ModelSpec],
    cache: EmbeddingCache | None,
    source: str,
    on_model: Callable[[ModelSpec], None] | None = None,
) -> BenchResult:
    """Benchmark each model in turn, so latency measurements don't compete."""
    result = BenchResult(
        source=source,
        documents=len(bench.documents),
        queries=len(bench.queries),
        query_source=bench.query_source,
        notes=list(bench.notes),
    )
    for spec in specs:
        if on_model:
            on_model(spec)
        result.models.append(await _run_one(bench, spec, cache))
    return result
