"""Comparing embedding models on a sample of your own documents: ``vecshift bench``.

Documents come from a JSONL file, a pgvector table, a Qdrant collection, or the caller (the
API's inline documents). Queries are labeled ones, written by a chat model, or proxies made
from the documents themselves. Before anything is embedded, ``confirm`` sees the plan: the
models, the estimated tokens and cost, and where each one sends the text.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.errors import Declined, ServiceError

if TYPE_CHECKING:
    from vecshift.bench import BenchResult, Document, PlanItem, Query
    from vecshift.embeddings import ModelSpec

DEFAULT_MODELS = ("hash/256", "hash/1024")


@dataclass(frozen=True, slots=True)
class DocumentsFile:
    """A JSONL file of ``{"id": ..., "text": ...}`` lines."""

    path: Path


@dataclass(frozen=True, slots=True)
class Documents:
    """Documents the caller already has, as (id, text)."""

    rows: tuple[tuple[str, str], ...]
    name: str = "documents"


@dataclass(frozen=True, slots=True)
class Database:
    """A sample of a pgvector table."""

    dsn: str
    table: str | None = None
    column: str | None = None
    text_column: str | None = None


@dataclass(frozen=True, slots=True)
class Collection:
    """A sample of a Qdrant collection (or alias)."""

    url: str
    key_env: str = "QDRANT_API_KEY"
    collection: str | None = None
    vector: str | None = None
    text_field: str | None = None


Source = DocumentsFile | Documents | Database | Collection


@dataclass(frozen=True, slots=True)
class BenchOptions:
    models: tuple[str, ...] = DEFAULT_MODELS
    sample_size: int = 1000
    num_queries: int = 100
    queries: Path | tuple[Query, ...] | None = None
    """Labeled queries: a JSONL file, or the queries themselves."""
    generate_queries: str | None = None
    """A chat model that writes the queries."""
    save_queries: Path | None = None
    """Where to keep the generated queries, to reuse as labeled ones."""
    seed: int = 7
    cache: bool = True


@dataclass(frozen=True, slots=True)
class BenchNotice:
    """What's about to be embedded, and where it goes."""

    items: tuple[PlanItem, ...]
    generator: ModelSpec | None
    documents: int

    @property
    def remote(self) -> tuple[tuple[str, str], ...]:
        """(model, URL) for each model that isn't on this machine."""
        specs = [i.spec for i in self.items]
        if self.generator is not None:
            specs.append(self.generator)
        return tuple((s.name, s.url or "") for s in specs if not s.is_local)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "models": [
                {
                    "model": i.spec.safe_raw,
                    "tokens": i.est_tokens,
                    "cost_usd": i.est_cost,
                    "where": None if i.spec.is_local else _shown(i.spec.url),
                }
                for i in self.items
            ],
            "generator": None
            if self.generator is None
            else {
                "model": self.generator.safe_raw,
                "where": None if self.generator.is_local else _shown(self.generator.url),
            },
        }


ConfirmBench = Callable[[BenchNotice], bool]
"""Shown the plan before anything is embedded, whether or not text leaves the machine.
Returning False stops with ``Declined``; it may also raise ``ServiceError`` itself."""


def _shown(url: str | None) -> str | None:
    return url.split("?", 1)[0] if url else url


def check(options: BenchOptions) -> tuple[list[ModelSpec], ModelSpec | None]:
    """The models to compare and the query writer, or the problem with the options."""
    from vecshift.embeddings import SpecError, parse_spec

    if options.queries is not None and options.generate_queries:
        raise ServiceError(
            "Use either --queries or --generate-queries, not both.", code="options.invalid"
        )
    if options.save_queries and not options.generate_queries:
        raise ServiceError(
            "--save-queries only works with --generate-queries.",
            "Proxy queries change the documents they come from, so they can't be reused.",
            "options.invalid",
        )
    try:
        specs = [parse_spec(m) for m in options.models or DEFAULT_MODELS]
        generator = parse_spec(options.generate_queries) if options.generate_queries else None
    except SpecError as exc:
        raise ServiceError(str(exc), code="options.invalid") from exc
    return specs, generator


def run_bench(
    source: Source | None,
    options: BenchOptions,
    *,
    confirm: ConfirmBench | None = None,
    on_step: Callable[[str], None] | None = None,
) -> BenchResult:
    """Sample the documents, make or load the queries, and score each model in turn."""
    from vecshift.bench import Benchmark, CorpusError, Query, plan, run
    from vecshift.bench import corpus as corpus_mod
    from vecshift.embeddings import EmbeddingCache

    step = on_step or (lambda message: None)
    specs, generator = check(options)
    try:
        if isinstance(options.queries, Path):
            labeled: list[Query] | None = corpus_mod.load_queries(options.queries)
        else:
            labeled = list(options.queries) if options.queries is not None else None
        wanted = {r for q in labeled for r in q.relevant} if labeled else set()
        documents, name = _documents(source, options, wanted)
    except CorpusError as exc:
        raise ServiceError(str(exc), code="documents.invalid") from exc

    notes: list[str] = []
    if labeled is not None:
        ids = {d.id for d in documents}
        kept = [Query(q.text, q.relevant & ids) for q in labeled if q.relevant & ids]
        if len(kept) < len(labeled):
            notes.append(
                f"{len(labeled) - len(kept)} labeled queries had no relevant document "
                "in the corpus and were skipped."
            )
        if not kept:
            raise ServiceError(
                "None of the labeled queries' documents are in the corpus.",
                code="documents.invalid",
            )
        benchmark = Benchmark(documents, kept, "labeled", notes)
    elif generator is not None:
        benchmark = Benchmark(documents, [], "generated", notes)
    else:
        try:
            edited, proxy, proxy_notes = corpus_mod.proxy_queries(
                documents, options.num_queries, options.seed
            )
        except CorpusError as exc:
            raise ServiceError(str(exc), code="documents.invalid") from exc
        benchmark = Benchmark(edited, proxy, "proxy", proxy_notes)

    if generator is not None:
        estimate = Benchmark(documents, [Query("x" * 60, frozenset())] * options.num_queries, "")
        _confirm(confirm, BenchNotice(tuple(plan(estimate, specs)), generator, len(documents)))
        benchmark.queries = asyncio.run(
            _generate(generator, documents, options.num_queries, options.seed)
        )
        if not benchmark.queries:
            raise ServiceError("The chat model didn't return any queries.", code="model.failed")
        benchmark.notes.append(f"Queries were written by {generator.name}.")
        if options.save_queries:
            corpus_mod.save_queries(options.save_queries, benchmark.queries)
            step(f"Saved {len(benchmark.queries)} queries to {options.save_queries}")
    else:
        notice = BenchNotice(tuple(plan(benchmark, specs)), None, len(benchmark.documents))
        _confirm(confirm, notice)

    cache = EmbeddingCache() if options.cache else None
    try:
        return asyncio.run(
            run(
                benchmark,
                specs,
                cache,
                name,
                on_model=lambda spec: step(f"Embedding with {spec.name}…"),
            )
        )
    finally:
        if cache:
            cache.close()


def _confirm(confirm: ConfirmBench | None, notice: BenchNotice) -> None:
    if confirm is None:
        if notice.remote:
            raise ServiceError(
                "Not sending data without confirmation.",
                "Allow sending text to: " + ", ".join(url for _, url in notice.remote),
                code="consent.required",
            )
        return
    if not confirm(notice):
        raise Declined("Not sending anything.")


def _documents(
    source: Source | None, options: BenchOptions, wanted: set[str]
) -> tuple[list[Document], str]:
    from vecshift.bench import corpus as corpus_mod

    size, seed = options.sample_size, options.seed
    if source is None:
        raise ServiceError(
            "No documents to benchmark.",
            "Pass --docs FILE, a database with --dsn (or VECSHIFT_DSN) and --table, or "
            "Qdrant with --qdrant URL and --collection.",
            "options.invalid",
        )
    if isinstance(source, DocumentsFile):
        loaded = corpus_mod.load_documents(source.path)
        return corpus_mod.sample(loaded, size, wanted, seed), source.path.name
    if isinstance(source, Documents):
        return corpus_mod.sample(_rows(source), size, wanted, seed), source.name
    if isinstance(source, Collection):
        return _from_qdrant(source, size, wanted)
    return _from_database(source, size, wanted)


def _rows(source: Documents) -> list[Document]:
    from vecshift.bench import CorpusError, Document
    from vecshift.bench.corpus import clip

    documents: list[Document] = []
    seen: set[str] = set()
    for doc_id, text in source.rows:
        if doc_id in seen:
            raise CorpusError(f"Duplicate document id {doc_id!r}.")
        seen.add(doc_id)
        if text.strip():
            documents.append(Document(doc_id, clip(text)))
    if not documents:
        raise CorpusError("There are no documents with text.")
    return documents


async def _generate(
    spec: ModelSpec, documents: Sequence[Document], n: int, seed: int
) -> list[Query]:
    from vecshift.bench.generate import GenerationError, QueryGenerator

    try:
        generator = QueryGenerator(spec)
    except GenerationError as exc:
        raise ServiceError(str(exc), code="model.failed") from exc
    try:
        return await generator.generate(list(documents), n, seed)
    except GenerationError as exc:
        raise ServiceError(str(exc), code="model.failed") from exc
    finally:
        await generator.aclose()


def _from_qdrant(source: Collection, size: int, wanted: set[str]) -> tuple[list[Document], str]:
    from vecshift.bench import Document
    from vecshift.bench.corpus import clip
    from vecshift.connectors import qdrant
    from vecshift.connectors.qdrant.documents import sample_documents

    try:
        settings = qdrant.prepare(source.url, source.key_env)
        with qdrant.QdrantClient(settings) as client:
            rows, name = sample_documents(
                client, source.collection, source.vector, source.text_field, size, wanted
            )
    except qdrant.QdrantError as exc:
        raise ServiceError(str(exc), exc.hint, "qdrant.error") from exc
    return [Document(i, clip(t)) for i, t in rows], name


def _from_database(source: Database, size: int, wanted: set[str]) -> tuple[list[Document], str]:
    from vecshift.bench import Document
    from vecshift.bench.corpus import clip
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.documents import sample_documents

    try:
        settings = pgvector.prepare(source.dsn)
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    try:
        rows, name = sample_documents(
            conn, source.table, source.column, source.text_column, size, wanted
        )
    except pgvector.TargetSelectionError as exc:
        raise ServiceError(str(exc), code="source.not_found") from exc
    finally:
        conn.rollback()
        conn.close()
    return [Document(i, clip(t)) for i, t in rows], name
