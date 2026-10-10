"""What a connector learns about an index, in a store-independent shape.

Connectors fill in an :class:`IndexProfile`; the checks in :mod:`vecshift.doctor.checks`
only ever look at the profile, never at the store.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AnnIndex:
    """An approximate-nearest-neighbour index over the vector field."""

    name: str
    method: str
    """For example ``hnsw`` or ``ivfflat``."""
    metric: str | None
    """``cosine``, ``inner_product``, ``l2``, ``l1``, or ``None`` if unknown."""


@dataclass(slots=True)
class Sample:
    """Statistics over a sample of rows. Vectors themselves are never fetched."""

    rows: int
    method: str
    """How rows were chosen, for example ``"full table"`` or ``"table sample"``."""
    null_vectors: int = 0
    dimensions: Counter[int] = field(default_factory=Counter)
    norms: list[float] = field(default_factory=list)
    texts_present: int | None = None
    """Rows with non-empty text, or ``None`` when there's no text field."""
    duplicate_texts: int = 0
    duplicate_vectors: int = 0
    models: Counter[str] = field(default_factory=Counter)
    """Recorded model per row, for rows that record one."""
    model_source: str | None = None
    """Where most rows record their model, for example ``metadata->>'model'``."""

    @property
    def vectors(self) -> int:
        """Rows that have a vector."""
        return self.rows - self.null_vectors


@dataclass(slots=True)
class IndexProfile:
    """Everything the checks need to know about one vector field."""

    store: str
    target: str
    """Human-readable location, for example ``public.documents.embedding``."""
    vector_type: str
    declared_dimensions: int | None
    estimated_rows: int | None
    sample: Sample
    text_field: str | None = None
    model_field: str | None = None
    updated_at_field: str | None = None
    has_primary_key: bool = True
    logical_replication: bool = False
    ann_indexes: list[AnnIndex] = field(default_factory=list)
    max_indexable_dimensions: int | None = None
    """The most dimensions the store can put in an ANN index, if it has a limit."""
    rows_hidden_by_access_rules: bool = False
    """Row-level security (or similar) may hide rows from the connecting role."""
    aliases: tuple[str, ...] | None = None
    """Aliases that point at this collection, for stores that switch by alias; ``None``
    where aliases don't apply."""
