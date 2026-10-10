"""Read-only inspection of a Qdrant collection for ``vecshift doctor``.

Qdrant can't compute norms or hashes server-side, so a sample of points is read with
their vectors, summarised here, and dropped. Nothing about a single point is kept or
shown: only counts, dimensions, norms, and hashes.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vecshift.connectors.qdrant.client import QdrantClient, QdrantError
from vecshift.doctor.profile import AnnIndex, IndexProfile, Sample

# Payload keys, most likely first. LangChain stores page_content and metadata.
TEXT_KEYS = ("page_content", "content", "text", "document", "chunk", "chunk_text", "body")
MODEL_KEYS = ("model_tag", "embedding_model", "model", "model_name")
METADATA_KEYS = ("metadata", "meta")
UPDATED_AT_KEYS = ("updated_at", "modified_at", "last_modified", "last_modified_at", "changed_at")

METRICS = {"Cosine": "cosine", "Dot": "inner_product", "Euclid": "l2", "Manhattan": "l1"}
MAX_DIMENSIONS = 65_536
PAGE = 256


class SelectionError(QdrantError):
    """The collection or vector to inspect is ambiguous or missing."""

    def __init__(self, message: str, candidates: Sequence[VectorField] = ()) -> None:
        super().__init__(message)
        self.candidates = list(candidates)


@dataclass(frozen=True, slots=True)
class VectorField:
    collection: str
    name: str | None
    """The named vector, or ``None`` for a collection's single unnamed vector."""
    dimensions: int | None
    distance: str | None
    datatype: str
    hnsw_m: int | None

    @property
    def qualified(self) -> str:
        return self.collection if self.name is None else f"{self.collection}:{self.name}"

    @property
    def type(self) -> str:
        return f"{self.datatype} vector"


def _fields(collection: str, info: dict[str, Any]) -> list[VectorField]:
    config = info.get("config") or {}
    params = config.get("params") or {}
    vectors = params.get("vectors") or {}
    default_m = (config.get("hnsw_config") or {}).get("m")

    def field(name: str | None, p: dict[str, Any]) -> VectorField:
        m = (p.get("hnsw_config") or {}).get("m", default_m)
        return VectorField(
            collection=collection,
            name=name,
            dimensions=int(p["size"]) if p.get("size") else None,
            distance=p.get("distance"),
            datatype=str(p.get("datatype") or "float32").lower(),
            hnsw_m=int(m) if m is not None else None,
        )

    if "size" in vectors:  # one unnamed vector
        return [field(None, vectors)]
    return [field(str(name), p) for name, p in sorted(vectors.items()) if isinstance(p, dict)]


def find_vector_fields(client: QdrantClient) -> list[VectorField]:
    """Every dense vector in every collection."""
    found = []
    for name in client.collections():
        found.extend(_fields(name, client.collection(name)))
    return found


def select_field(
    client: QdrantClient, collection: str | None, vector: str | None
) -> tuple[VectorField, dict[str, Any]]:
    """The vector to inspect, and its collection's details.

    ``collection`` may be an alias; it's resolved to the collection it points at.
    """
    if collection is None:
        fields = find_vector_fields(client)
        if vector is not None:
            fields = [f for f in fields if f.name == vector]
        if not fields:
            raise SelectionError("No collections with dense vectors found in this Qdrant.")
        if len(fields) > 1:
            raise SelectionError(
                "Found more than one vector. Choose one with --collection (and --vector).",
                fields,
            )
        collection = fields[0].collection
    resolved = dict(client.aliases()).get(collection, collection)
    try:
        info = client.collection(resolved)
    except QdrantError as exc:
        if "404" in str(exc):
            raise SelectionError(
                f"No collection or alias named {collection!r}.", find_vector_fields(client)
            ) from exc
        raise
    fields = _fields(resolved, info)
    if vector is not None:
        fields = [f for f in fields if f.name == vector]
        if not fields:
            raise SelectionError(
                f"Collection {resolved!r} has no vector named {vector!r}.", _fields(resolved, info)
            )
    if len(fields) > 1:
        raise SelectionError(
            f"Collection {resolved!r} has several named vectors. Choose one with --vector.",
            fields,
        )
    if not fields:
        raise SelectionError(f"Collection {resolved!r} has no dense vectors.")
    return fields[0], info


# --- reading payloads


def _get(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _vector(point: dict[str, Any], name: str | None) -> list[float] | None:
    raw = point.get("vector")
    if isinstance(raw, dict):
        raw = raw.get(name or "")
    if isinstance(raw, list) and raw and all(isinstance(x, (int, float)) for x in raw):
        return [float(x) for x in raw]
    return None


def _pick_text(points: Sequence[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for point in points:
        payload = point.get("payload") or {}
        for key in TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                counts[key] += 1
    for key in TEXT_KEYS:
        if counts[key]:
            return key
    return None


def _model(payload: dict[str, Any]) -> tuple[str, str] | None:
    """(where, model) for a payload that records its model."""
    for key in MODEL_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return f"payload.{key}", value
    for meta in METADATA_KEYS:
        inner = payload.get(meta)
        if isinstance(inner, dict):
            for key in MODEL_KEYS:
                value = inner.get(key)
                if isinstance(value, str) and value:
                    return f"payload.{meta}.{key}", value
    return None


def _points(
    client: QdrantClient, collection: str, vector: str | bool, size: int, total: int
) -> tuple[list[dict[str, Any]], str]:
    """Up to ``size`` points and how they were chosen."""
    if total > size:
        sampled = client.random_sample(collection, size, vector=vector)
        if sampled is not None:
            return sampled, "random sample"
    points: list[dict[str, Any]] = []
    offset = None
    while len(points) < size:
        page, offset = client.scroll(
            collection, min(PAGE, size - len(points)), offset, vector=vector
        )
        points.extend(page)
        if offset is None:
            break
    return points, "full collection" if len(points) >= total else "first points"


def _sample(
    points: Sequence[dict[str, Any]], name: str | None, method: str, text_field: str | None
) -> Sample:
    sample = Sample(rows=len(points), method=method)
    sample.texts_present = 0 if text_field else None
    vector_hashes: Counter[bytes] = Counter()
    text_hashes: Counter[bytes] = Counter()
    sources: Counter[str] = Counter()
    for point in points:
        payload = point.get("payload") or {}
        values = _vector(point, name)
        if values is None:
            sample.null_vectors += 1
        else:
            sample.dimensions[len(values)] += 1
            sample.norms.append(math.sqrt(sum(x * x for x in values)))
            packed = struct.pack(f"{len(values)}f", *values)
            vector_hashes[hashlib.blake2b(packed, digest_size=16).digest()] += 1
        if text_field:
            text = _get(payload, text_field)
            if isinstance(text, str) and text.strip():
                sample.texts_present = (sample.texts_present or 0) + 1
                text_hashes[hashlib.blake2b(text.encode(), digest_size=16).digest()] += 1
        recorded = _model(payload)
        if recorded and values is not None:
            sources[recorded[0]] += 1
            sample.models[recorded[1]] += 1
    sample.duplicate_vectors = sum(n - 1 for n in vector_hashes.values())
    sample.duplicate_texts = sum(n - 1 for n in text_hashes.values())
    sample.model_source = sources.most_common(1)[0][0] if sources else None
    return sample


def inspect(
    client: QdrantClient,
    collection: str | None = None,
    vector: str | None = None,
    text_field: str | None = None,
    sample_size: int = 2000,
) -> IndexProfile:
    """Profile one dense vector of a Qdrant collection. Read-only."""
    field, info = select_field(client, collection, vector)
    total = info.get("points_count")
    total = int(total) if total is not None else client.count(field.collection)
    points, method = _points(
        client, field.collection, field.name if field.name else True, sample_size, total
    )
    if text_field is None:
        text_field = _pick_text(points)
    elif points and not any(_get(p.get("payload") or {}, text_field) for p in points):
        raise SelectionError(
            f"No sampled point has a payload field {text_field!r} in {field.collection!r}."
        )
    sample = _sample(points, field.name, method, text_field)
    updated = next(
        (k for k in UPDATED_AT_KEYS if any(k in (p.get("payload") or {}) for p in points)), None
    )
    metric = METRICS.get(field.distance or "")
    indexes = [AnnIndex(field.collection, "hnsw", metric)] if field.hnsw_m != 0 else []
    aliases = tuple(sorted(a for a, c in client.aliases() if c == field.collection))
    return IndexProfile(
        store="qdrant",
        target=field.qualified,
        vector_type=field.type,
        declared_dimensions=field.dimensions,
        estimated_rows=total,
        sample=sample,
        text_field=f"payload.{text_field}" if text_field else None,
        model_field=sample.model_source,
        updated_at_field=f"payload.{updated}" if updated else None,
        has_primary_key=True,
        ann_indexes=indexes,
        max_indexable_dimensions=MAX_DIMENSIONS,
        aliases=aliases,
    )
