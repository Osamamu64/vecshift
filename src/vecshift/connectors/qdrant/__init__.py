"""Qdrant, over its REST API."""

from vecshift.connectors.qdrant.client import (
    QdrantClient,
    QdrantError,
    QdrantSettings,
    prepare,
)
from vecshift.connectors.qdrant.inspect import (
    SelectionError,
    VectorField,
    find_vector_fields,
    inspect,
    select_field,
)

__all__ = [
    "QdrantClient",
    "QdrantError",
    "QdrantSettings",
    "SelectionError",
    "VectorField",
    "find_vector_fields",
    "inspect",
    "prepare",
    "select_field",
]
