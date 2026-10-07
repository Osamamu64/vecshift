"""Capability flags connectors declare, so the planner can choose a strategy."""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    """Something a connector can do.

    The planner never assumes a capability a connector hasn't declared.
    """

    ALIAS = "alias"
    """Atomic cutover by repointing an alias (OpenSearch, Elasticsearch, Qdrant)."""

    NAMED_VECTORS = "named_vectors"
    """Several vectors per record side by side (Qdrant)."""

    CHANGE_FEED = "change_feed"
    """Stream inserts, updates, and deletes (Postgres logical replication)."""

    SCROLL = "scroll"
    """Resumable full scans with a cursor."""

    SPARSE = "sparse"
    """Sparse vectors, for hybrid search."""

    DELETE_DETECTION = "delete_detection"
    """Report deletes directly rather than by diffing."""

    STORES_TEXT = "stores_text"
    """Original text is kept alongside vectors, so records can be re-embedded."""
