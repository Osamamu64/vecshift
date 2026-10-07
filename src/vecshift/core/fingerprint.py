"""Fingerprints that identify which vector space a vector belongs to.

Two vectors are only comparable if they were produced by the same model,
with the same dimensions, task mode, prefix, and normalization. A change to
any of those is a different vector space, and mixing spaces in one index
silently degrades retrieval. The fingerprint makes that provenance explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

_HASH_LENGTH = 12


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """Everything that determines an embedding's vector space."""

    provider: str
    model: str
    dimensions: int
    version: str | None = None
    task: str | None = None
    prefix: str | None = None
    normalized: bool = True

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")

    @property
    def digest(self) -> str:
        """A stable hash over every field, insensitive to provider/model casing."""
        fields = asdict(self)
        fields["provider"] = self.provider.strip().lower()
        fields["model"] = self.model.strip().lower()
        canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:_HASH_LENGTH]

    @property
    def model_tag(self) -> str:
        """A readable, unique tag such as ``openai/text-embedding-3-small@1536#a1b2c3d4e5f6``."""
        return (
            f"{self.provider.strip().lower()}/{self.model.strip().lower()}"
            f"@{self.dimensions}#{self.digest}"
        )

    def is_compatible_with(self, other: EmbeddingFingerprint) -> bool:
        """Whether vectors from ``self`` and ``other`` share one vector space."""
        return self.digest == other.digest
