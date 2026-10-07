"""The canonical record every connector reads into and writes from.

Connectors convert between their store's native format and :class:`Record`,
so supporting N stores needs N connectors rather than N x N pairwise bridges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse vector as parallel index/value arrays."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices and values must have equal length "
                f"({len(self.indices)} != {len(self.values)})"
            )


@dataclass(slots=True)
class Record:
    """One document or chunk moving through a migration.

    Attributes:
        id: Stable identifier from the source store.
        text: Original text. Required for re-embedding; ``None`` when the
            store holds vectors only.
        vectors: Dense vectors keyed by name (``"default"`` for single-vector stores).
        sparse: Sparse vectors keyed by name, for hybrid search.
        metadata: Arbitrary payload carried across unchanged unless remapped.
        updated_at: Last-modified time, used to stop backfill from overwriting
            newer writes that arrived through a change feed.
        model_tag: Fingerprint of the vector space the vectors belong to.
        deleted: Tombstone marker for deletes observed through a change feed.
    """

    id: str
    text: str | None = None
    vectors: dict[str, list[float]] = field(default_factory=dict)
    sparse: dict[str, SparseVector] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime | None = None
    model_tag: str | None = None
    deleted: bool = False

    @property
    def can_reembed(self) -> bool:
        """Whether this record carries the text needed to embed it again."""
        return bool(self.text)

    @classmethod
    def tombstone(cls, id: str, updated_at: datetime | None = None) -> Record:
        """A delete marker for ``id``."""
        return cls(id=id, updated_at=updated_at, deleted=True)

    def is_newer_than(self, other: Record) -> bool:
        """Whether this record should win an upsert race against ``other``.

        A record without ``updated_at`` never beats one that has it, so a
        timestamp-less backfill can't overwrite a timestamped live write.
        """
        if self.updated_at is None:
            return False
        if other.updated_at is None:
            return True
        return self.updated_at > other.updated_at
