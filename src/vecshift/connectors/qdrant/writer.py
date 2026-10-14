"""What ``vecshift apply`` does to Qdrant: fill a new collection from a source.

The source is a Qdrant collection or a PostgreSQL table, read page by page in key order.
Each source point (or row) becomes a point in the new collection with the same payload and
a vector from the new model. Nothing records progress except the new collection itself: a
point needs embedding when it's missing there or its text differs from the source's, so
apply can be stopped and run again at any time, and edits made meanwhile are picked up.
Payload-only edits are copied without re-embedding, and points deleted from the source are
removed by :meth:`QdrantWriter.remove_deleted`.

The source is only read. Writes go to the new collection, which the application doesn't
search until cutover points its alias there.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from vecshift.connectors.qdrant.client import QdrantClient, QdrantError
from vecshift.connectors.qdrant.inspect import _get

PAGE = 256
INDEX_WAIT_S = 6 * 3600.0
"""How long apply waits for Qdrant to finish indexing before it gives up waiting."""

DISTANCES = {"cosine": "Cosine", "inner_product": "Dot", "l2": "Euclid", "l1": "Manhattan"}


class AlreadyRunning(Exception):
    """Another apply holds the lock for this job."""


@dataclass(frozen=True, slots=True)
class Item:
    """A source point or row with text: what the engine embeds."""

    key: Any
    """Position in the source's key order: its point ID or primary key."""
    text: str
    point: int | str
    """The ID it gets in the new collection."""
    payload: dict[str, Any] = field(hash=False, compare=False)

    @property
    def id(self) -> str:
        return str(self.key)


class Source(Protocol):
    """Reads a source in a stable key order."""

    text_field: str
    """Where the text is in the new collection's payload, as a dotted path."""

    def page(self, after: Any, limit: int) -> tuple[list[Item], Any]:
        """Up to ``limit`` entries after key ``after``, and the key to continue after, or
        ``None`` at the end. Entries without text are skipped but still advance the key."""
        ...

    def count(self) -> int:
        """Entries with text."""
        ...

    def present(self, points: Sequence[dict[str, Any]]) -> set[int | str]:
        """Of these new-collection points, the IDs whose source still exists with text."""
        ...


class QdrantSource:
    """Points of a Qdrant collection, in point ID order."""

    def __init__(
        self, client: QdrantClient, collection: str, text_field: str, vector: str | None
    ) -> None:
        self.client = client
        self.collection = collection
        self.text_field = text_field
        self.vector = vector

    def page(self, after: Any, limit: int) -> tuple[list[Item], Any]:
        # Scroll starts at an ID, inclusive, whether or not it still exists.
        points, more = self.client.scroll(self.collection, limit + 1, after, payload=True)
        if points and after is not None and points[0].get("id") == after:
            points = points[1:]
        elif len(points) > limit:
            points = points[:limit]
            more = points[-1]["id"]
        items = []
        for point in points:
            payload = point.get("payload") or {}
            text = _get(payload, self.text_field)
            if isinstance(text, str) and text.strip():
                items.append(Item(point["id"], text, point["id"], payload))
        last = points[-1]["id"] if points else None
        return items, (last if more is not None else None)

    def count(self) -> int:
        count, cursor = 0, None
        while True:
            items, cursor = self.page(cursor, PAGE)
            count += len(items)
            if cursor is None:
                return count

    def present(self, points: Sequence[dict[str, Any]]) -> set[int | str]:
        found = self.client.retrieve(self.collection, [p["id"] for p in points])
        return {
            p["id"]
            for p in found
            if isinstance(t := _get(p.get("payload") or {}, self.text_field), str) and t.strip()
        }


@dataclass(frozen=True, slots=True)
class Layout:
    collection: str
    """The new collection."""
    vector: str | None
    """Its vector's name, or ``None`` for a single unnamed vector."""
    dims: int
    distance: str
    """Qdrant's name for it: Cosine, Dot, Euclid, or Manhattan."""
    payload_indexes: dict[str, Any] = field(default_factory=dict)
    """Field → schema, copied from the source so filters stay fast."""

    def vectors_config(self) -> dict[str, Any]:
        params = {"size": self.dims, "distance": self.distance}
        return params if self.vector is None else {self.vector: params}

    def matches(self, info: dict[str, Any]) -> bool:
        vectors = ((info.get("config") or {}).get("params") or {}).get("vectors") or {}
        if self.vector is not None:
            vectors = vectors.get(self.vector) or {}
        elif "size" not in vectors:
            return False
        return vectors.get("size") == self.dims and vectors.get("distance") == self.distance


class QdrantWriter:
    """The engine's writer for a new Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        source: Source,
        layout: Layout,
        lock_path: Path | None = None,
        *,
        page_size: int = PAGE,
        index_wait_s: float = INDEX_WAIT_S,
        sleep: Callable[[float], None] = time.sleep,
        read_only: bool = False,
    ) -> None:
        """``read_only`` counts without copying changed payloads, for status."""
        self.client = client
        self.source = source
        self.layout = layout
        self.lock_path = lock_path
        self.page_size = page_size
        self.index_wait_s = index_wait_s
        self._sleep = sleep
        self.read_only = read_only
        self._lock: IO[str] | None = None
        self.payloads_copied = 0
        """Points whose payload changed in the source and was copied without re-embedding."""

    # --- concurrency guard

    def acquire(self) -> None:
        """Make sure no other apply runs this job from the same directory at the same time.

        Qdrant has no locks, so this guards against a second run on this machine only.
        """
        if self.lock_path is None:
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            return
        self.lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = self.lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise AlreadyRunning(
                f"Another vecshift apply is already filling {self.layout.collection}."
            ) from exc
        self._lock = handle

    def release(self) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    # --- the new collection

    def ensure_column(self) -> bool:
        """Create the new collection if it's missing. Returns whether it was created."""
        lay = self.layout
        if self.client.exists(lay.collection):
            if not lay.matches(self.client.collection(lay.collection)):
                raise QdrantError(
                    f"Collection {lay.collection!r} already exists with a different vector "
                    "size or distance.",
                    hint="Pick another target.collection, or delete that one if it's unused.",
                )
            return False
        self.client.create_collection(lay.collection, lay.vectors_config())
        for name, schema in sorted(lay.payload_indexes.items()):
            self.client.create_payload_index(lay.collection, name, schema)
        return True

    def ensure_trigger(self) -> bool:
        """Qdrant has no triggers: each apply run compares the source with the copy."""
        return False

    # --- comparing source and copy

    def _scan(self, after: Any, exclude: Sequence[str]) -> Iterator[tuple[Item, bool]]:
        """Source entries after ``after`` with whether each needs embedding. Payload-only
        changes are copied on the way."""
        excluded = set(exclude)
        cursor = after
        while True:
            items, cursor = self.source.page(cursor, self.page_size)
            copies = {
                p["id"]: p.get("payload") or {}
                for p in self.client.retrieve(self.layout.collection, [i.point for i in items])
            }
            checked: list[tuple[Item, bool]] = []
            stale: list[dict[str, Any]] = []
            for item in items:
                if item.id in excluded:
                    continue
                copy = copies.get(item.point)
                needs = copy is None or _get(copy, self.source.text_field) != item.text
                if not needs and copy != item.payload:
                    stale.append(
                        {"overwrite_payload": {"payload": item.payload, "points": [item.point]}}
                    )
                checked.append((item, needs))
            if stale and not self.read_only:
                self.client.batch(self.layout.collection, stale)
                self.payloads_copied += len(stale)
            yield from checked
            if cursor is None:
                return

    def total_count(self) -> int:
        """Source entries with text, which should all end up in the new collection."""
        return self.source.count()

    def pending_count(self, exclude: Sequence[str] = ()) -> int:
        return sum(1 for _, needs in self._scan(None, exclude) if needs)

    def fetch(self, after: Any, limit: int, exclude: Sequence[str] = ()) -> list[Item]:
        """The next entries after ``after`` that need a vector."""
        found: list[Item] = []
        for item, needs in self._scan(after, exclude):
            if needs:
                found.append(item)
                if len(found) >= limit:
                    break
        return found

    def write(self, items: Sequence[tuple[Item, Sequence[float]]]) -> int:
        """Store points with their source payload and new vector.

        If the source text changed after it was read, the copy's text differs from the
        source's again, so the next pass embeds the new text.
        """
        lay = self.layout
        points = [
            {
                "id": item.point,
                "vector": list(vector) if lay.vector is None else {lay.vector: list(vector)},
                "payload": item.payload,
            }
            for item, vector in items
        ]
        self.client.upsert(lay.collection, points)
        return len(points)

    def remove_deleted(self) -> int:
        """Delete points whose source is gone or has lost its text. Returns how many."""
        removed, offset = 0, None
        while True:
            points, offset = self.client.scroll(
                self.layout.collection, self.page_size, offset, payload=True
            )
            if points:
                keep = self.source.present(points)
                gone = [p["id"] for p in points if p["id"] not in keep]
                self.client.delete_points(self.layout.collection, gone)
                removed += len(gone)
            if offset is None:
                return removed

    # --- index

    def build_index(self, method: str, metric: str, rows: int) -> str:
        """Wait for Qdrant to finish optimizing and indexing the new collection.

        Qdrant builds its HNSW index itself as points arrive.
        """
        waited = False
        deadline = time.monotonic() + self.index_wait_s
        while True:
            status = str(self.client.collection(self.layout.collection).get("status", ""))
            if status == "red":
                raise QdrantError(
                    f"Qdrant reports collection {self.layout.collection!r} as failed (red).",
                    hint="Check the Qdrant logs, then run apply again.",
                )
            if status != "yellow":
                return "built" if waited else "exists"
            if time.monotonic() > deadline:
                return "pending"
            waited = True
            self._sleep(2.0)

    def dimensions_in_use(self) -> set[int]:
        lay = self.layout
        points, _ = self.client.scroll(
            lay.collection, 100, vector=lay.vector if lay.vector else True, payload=False
        )
        sizes = set()
        for point in points:
            raw = point.get("vector")
            if isinstance(raw, dict):
                raw = raw.get(lay.vector or "")
            if isinstance(raw, list):
                sizes.add(len(raw))
        return sizes
