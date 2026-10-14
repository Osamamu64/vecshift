"""What the target of a migration into Qdrant looks like: the new collection and the alias
the application searches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vecshift.connectors.qdrant.client import QdrantClient
from vecshift.connectors.qdrant.inspect import VectorField, _fields


@dataclass(frozen=True, slots=True)
class QdrantTargetState:
    collection: str
    """The new collection."""
    vector: str | None
    """The new vector's name, or ``None`` for an unnamed one."""
    exists: bool = False
    dimensions: int | None = None
    distance: str | None = None
    points: int | None = None
    status: str | None = None
    """Qdrant's collection status: green when it's done indexing."""
    alias: str | None = None
    """The alias cutover points at the new collection."""
    alias_target: str | None = None
    """The collection that alias points at now, if it exists."""
    live: str | None = None
    """The collection the source vectors are in, for a Qdrant source."""
    source_is_alias: bool = False
    other_vectors: tuple[str, ...] = ()
    """Other dense or sparse vectors in the source collection, which a copy would drop."""
    payload_indexes: dict[str, Any] = field(default_factory=dict)
    """Payload indexes on the source collection, recreated on the new one."""
    collections: frozenset[str] = frozenset()
    """Every collection on the server."""


def payload_indexes(info: dict[str, Any]) -> dict[str, Any]:
    """Field → schema, in the form creating an index takes."""
    found: dict[str, Any] = {}
    for name, entry in (info.get("payload_schema") or {}).items():
        if not isinstance(entry, dict):
            continue
        params = entry.get("params")
        schema = (
            params if isinstance(params, dict) and params.get("type") else entry.get("data_type")
        )
        if schema:
            found[str(name)] = schema
    return found


def inspect_target(
    client: QdrantClient,
    collection: str,
    vector: str | None,
    alias: str | None,
    source: tuple[str, VectorField, dict[str, Any]] | None = None,
) -> QdrantTargetState:
    """Read-only. ``source`` is (name the job gives, field, collection info) for a Qdrant
    source, whose alias is the default one to switch."""
    collections = frozenset(client.collections())
    aliases = dict(client.aliases())
    live = None
    source_is_alias = False
    other: tuple[str, ...] = ()
    indexes: dict[str, Any] = {}
    if source is not None:
        given, field_, info = source
        live = field_.collection
        source_is_alias = given in aliases
        if alias is None and source_is_alias:
            alias = given
        names = [f.name or "" for f in _fields(live, info) if f.name != field_.name]
        sparse = ((info.get("config") or {}).get("params") or {}).get("sparse_vectors") or {}
        other = tuple(sorted([*names, *(str(n) for n in sparse)]))
        indexes = payload_indexes(info)

    state: dict[str, Any] = {}
    if collection in collections:
        info = client.collection(collection)
        fields = [f for f in _fields(collection, info) if f.name == vector]
        found = fields[0] if fields else None
        state = {
            "exists": True,
            "dimensions": found.dimensions if found else None,
            "distance": found.distance if found else None,
            "points": info.get("points_count"),
            "status": info.get("status"),
        }
    return QdrantTargetState(
        collection=collection,
        vector=vector,
        alias=alias,
        alias_target=aliases.get(alias) if alias else None,
        live=live,
        source_is_alias=source_is_alias,
        other_vectors=other,
        payload_indexes=indexes,
        collections=collections,
        **state,
    )
