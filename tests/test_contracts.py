from collections.abc import AsyncIterator, Sequence

from vecshift import Capability, EmbeddingFingerprint, Record
from vecshift.core.contracts import EmbeddingProvider, SourceConnector, TargetConnector


class _Memory:
    """A minimal in-memory store satisfying both connector contracts."""

    def __init__(self) -> None:
        self.records: dict[str, Record] = {}

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.SCROLL, Capability.STORES_TEXT})

    async def count(self) -> int | None:
        return len(self.records)

    async def read(
        self, batch_size: int, cursor: str | None = None
    ) -> AsyncIterator[tuple[Sequence[Record], str | None]]:
        yield list(self.records.values()), None

    async def upsert(self, records: Sequence[Record]) -> None:
        for r in records:
            self.records[r.id] = r


class _Mock:
    @property
    def fingerprint(self) -> EmbeddingFingerprint:
        return EmbeddingFingerprint(provider="mock", model="mock", dimensions=2)

    async def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_reference_implementations_satisfy_contracts() -> None:
    store = _Memory()
    assert isinstance(store, SourceConnector)
    assert isinstance(store, TargetConnector)
    assert isinstance(_Mock(), EmbeddingProvider)
