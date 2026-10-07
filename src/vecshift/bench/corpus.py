"""Documents and queries for a benchmark."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 8000
"""Longer documents are cut to this length, well inside common model input limits."""


class CorpusError(ValueError):
    """Documents or queries couldn't be loaded."""


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    relevant: frozenset[str]
    """IDs of the documents that answer this query."""


@dataclass(slots=True)
class Benchmark:
    documents: list[Document]
    queries: list[Query]
    query_source: str
    """``proxy``, ``generated``, or ``labeled``."""
    notes: list[str] = field(default_factory=list)


def clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS]


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusError(f"Couldn't read {path}: {exc.strerror}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise CorpusError(f"{path}:{number}: expected a JSON object")
        yield number, row


def load_documents(path: Path) -> list[Document]:
    """Read ``{"id": ..., "text": ...}`` lines. ``id`` is optional and defaults to the line."""
    docs: list[Document] = []
    seen: set[str] = set()
    for number, row in _read_jsonl(path):
        text = row.get("text")
        if not isinstance(text, str):
            raise CorpusError(f'{path}:{number}: each line needs a "text" string')
        if not text.strip():
            continue
        doc_id = str(row.get("id", number))
        if doc_id in seen:
            raise CorpusError(f"{path}:{number}: duplicate id {doc_id!r}")
        seen.add(doc_id)
        docs.append(Document(doc_id, clip(text)))
    if not docs:
        raise CorpusError(f"{path} has no documents")
    return docs


def load_queries(path: Path) -> list[Query]:
    """Read ``{"query": ..., "relevant": [ids]}`` lines."""
    queries: list[Query] = []
    for number, row in _read_jsonl(path):
        text, relevant = row.get("query"), row.get("relevant")
        if not isinstance(text, str) or not text.strip():
            raise CorpusError(f'{path}:{number}: each line needs a "query" string')
        if isinstance(relevant, (str, int)):
            relevant = [relevant]
        if not isinstance(relevant, list) or not relevant:
            raise CorpusError(f'{path}:{number}: "relevant" must list at least one document id')
        queries.append(Query(text.strip(), frozenset(str(r) for r in relevant)))
    if not queries:
        raise CorpusError(f"{path} has no queries")
    return queries


def save_queries(path: Path, queries: Sequence[Query]) -> None:
    lines = [
        json.dumps({"query": q.text, "relevant": sorted(q.relevant)}, ensure_ascii=False)
        for q in queries
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample(docs: Sequence[Document], n: int, keep: set[str], seed: int) -> list[Document]:
    """Up to ``n`` documents, always including the IDs in ``keep``."""
    kept = [d for d in docs if d.id in keep]
    rest = [d for d in docs if d.id not in keep]
    room = max(0, n - len(kept))
    chosen = random.Random(seed).sample(rest, min(room, len(rest)))
    return kept + chosen


def _norm(sentence: str) -> str:
    return " ".join(sentence.lower().split())


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORDS = re.compile(r"\S+")


def proxy_queries(
    docs: Sequence[Document], n: int, seed: int
) -> tuple[list[Document], list[Query], list[str]]:
    """Make queries from the documents themselves, with no LLM.

    For each chosen document, one sentence becomes the query and is removed from the
    document, so a model has to match it by meaning to the rest of the text (the
    inverse cloze task). Single-sentence documents use their opening words instead.
    Returns the edited documents, the queries, and notes for the report.
    """
    rng = random.Random(seed)
    split = {doc.id: [s for s in _SENTENCE.split(doc.text.strip()) if s.strip()] for doc in docs}
    # A sentence repeated across documents (boilerplate, templates) would make the true
    # document the only one *without* the query text, so only unique sentences qualify.
    counts = Counter(_norm(s) for sentences in split.values() for s in set(sentences))
    candidates: dict[str, tuple[str, str]] = {}
    for doc in docs:
        sentences = split[doc.id]
        usable = [
            i
            for i, s in enumerate(sentences)
            if 6 <= len(_WORDS.findall(s)) <= 40 and counts[_norm(s)] == 1
        ]
        if len(sentences) >= 2 and usable:
            i = rng.choice(usable)
            rest = " ".join(s for j, s in enumerate(sentences) if j != i)
            if len(_WORDS.findall(rest)) >= 8:
                candidates[doc.id] = (sentences[i], rest)
                continue
        words = _WORDS.findall(doc.text)
        if len(words) >= 20:
            cut = min(12, len(words) // 3)
            candidates[doc.id] = (" ".join(words[:cut]), " ".join(words[cut:]))

    if not candidates:
        raise CorpusError(
            "The documents are too short to make proxy queries from. Use --generate-queries "
            "or --queries instead."
        )
    chosen = set(rng.sample(sorted(candidates), min(n, len(candidates))))
    edited = [Document(d.id, candidates[d.id][1]) if d.id in chosen else d for d in docs]
    queries = [Query(candidates[i][0], frozenset({i})) for i in sorted(chosen)]
    notes = [
        "Queries are sentences taken out of the documents (a proxy, no LLM involved). "
        "That's good for ranking models against each other; use --generate-queries or "
        "--queries for numbers closer to real search."
    ]
    if len(chosen) < n:
        notes.append(f"Only {len(chosen)} documents were long enough to make queries from.")
    return edited, queries, notes
