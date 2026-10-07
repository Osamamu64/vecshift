"""Plugin contracts. Third-party connectors and providers implement these.

Plugins register under the ``vecshift.plugins`` entry-point group.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal, Protocol, runtime_checkable

from vecshift.core.capabilities import Capability
from vecshift.core.fingerprint import EmbeddingFingerprint
from vecshift.core.record import Record

EmbedMode = Literal["document", "query"]


@runtime_checkable
class SourceConnector(Protocol):
    """Reads records from a store."""

    def capabilities(self) -> frozenset[Capability]: ...

    async def count(self) -> int | None:
        """Total records, or ``None`` if the store can't count cheaply."""
        ...

    def read(
        self, batch_size: int, cursor: str | None = None
    ) -> AsyncIterator[tuple[Sequence[Record], str | None]]:
        """Yield batches with the cursor to resume after each one."""
        ...


@runtime_checkable
class TargetConnector(Protocol):
    """Writes records to a store."""

    def capabilities(self) -> frozenset[Capability]: ...

    async def upsert(self, records: Sequence[Record]) -> None:
        """Idempotently write records, honouring tombstones and ``updated_at``."""
        ...

    async def count(self) -> int | None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    @property
    def fingerprint(self) -> EmbeddingFingerprint: ...

    async def embed(
        self, texts: Sequence[str], mode: EmbedMode = "document"
    ) -> list[list[float]]: ...
