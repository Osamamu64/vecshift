"""Read a sample of documents (ID and text) from a Qdrant collection, for benchmarks."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from vecshift.connectors.qdrant.client import QdrantClient
from vecshift.connectors.qdrant.inspect import (
    SelectionError,
    _get,
    _pick_text,
    _points,
    select_field,
)


def _point_id(raw: str) -> int | str:
    """Qdrant IDs are unsigned integers or UUIDs; labeled queries name them as strings."""
    return int(raw) if raw.isdigit() else raw


def sample_documents(
    client: QdrantClient,
    collection: str | None,
    vector: str | None,
    text_field: str | None,
    size: int,
    include_ids: Collection[str] = (),
) -> tuple[list[tuple[str, str]], str]:
    """Up to ``size`` (id, text) points, plus any ``include_ids``, and the collection name."""
    field, info = select_field(client, collection, vector)
    total = int(info.get("points_count") or 0)
    points, _ = _points(client, field.collection, False, size, total)
    text = text_field or _pick_text(points)
    if text is None:
        raise SelectionError(
            f"No text payload field found in {field.collection!r}. Pass --text-column."
        )
    seen = {str(p.get("id")) for p in points}
    missing = [_point_id(i) for i in include_ids if i not in seen]
    points.extend(client.retrieve(field.collection, missing))

    rows: list[tuple[str, str]] = []
    for point in points:
        value: Any = _get(point.get("payload") or {}, text)
        if isinstance(value, str) and value.strip():
            rows.append((str(point.get("id")), value))
    return rows, field.collection
