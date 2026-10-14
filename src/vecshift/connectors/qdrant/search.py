"""Read-only searches over a migration's old and new vectors, for ``vecshift eval``.

``SplitSearcher`` gives eval one searcher over two sides that can live in different
places: two Qdrant collections, or a pgvector column (old) and a Qdrant collection (new).
Every side reports rows by the same key, so results can be compared.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from vecshift.connectors.qdrant.client import QdrantClient
from vecshift.connectors.qdrant.inspect import _get, _points, _vector

Side = Literal["old", "new"]
Hit = tuple[str, float]
SWEEP = (32, 64, 128, 256)
METHOD = "qdrant-hnsw"
"""How eval reports a Qdrant index; its search setting is hnsw_ef."""


class SideSearcher(Protocol):
    def search(self, vector: Sequence[float], k: int, setting: int | None = None) -> list[Hit]: ...
    def exact(self, vector: Sequence[float], k: int) -> list[Hit]: ...
    def index(self) -> tuple[str | None, int | None, list[int]]: ...
    def neighbours(self, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]: ...
    def vectors(self, keys: Sequence[str]) -> list[list[float]]: ...


class Rows(Protocol):
    def sample(self, text: str, size: int) -> list[tuple[str, str]]: ...
    def texts(self, text: str, keys: Sequence[str]) -> dict[str, str]: ...
    def translate(self, keys: Sequence[str]) -> dict[str, str]: ...


def point_from_key(key: str) -> int | str:
    """A Qdrant point ID from its text form."""
    return int(key) if key.isdigit() else key


class QdrantSide:
    """One collection's vector, searched as the application would."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector: str | None,
        distance: str,
        *,
        key_field: str | None = None,
        to_points: Callable[[Sequence[str]], list[int | str]] | None = None,
        group_field: str | None = None,
    ) -> None:
        """``key_field`` names the payload field holding each point's key, when keys aren't
        point IDs (a pgvector source's primary key); ``to_points`` turns keys back into
        point IDs (those it can't are left out)."""
        self.client = client
        self.collection = collection
        self.vector = vector
        self.distance = distance
        self.key_field = key_field
        self.to_points = to_points or (lambda keys: [point_from_key(k) for k in keys])
        self.group_field = group_field
        self._index: tuple[str | None, int | None, list[int]] | None = None

    def _key(self, point: dict[str, Any]) -> str:
        if self.key_field:
            return str(_get(point.get("payload") or {}, self.key_field))
        return str(point["id"])

    def _distance(self, score: float) -> float:
        # Qdrant returns similarity for Cosine and Dot, distance otherwise.
        if self.distance == "Cosine":
            return 1.0 - score
        if self.distance == "Dot":
            return -score
        return score

    def _payload(self) -> bool | list[str]:
        fields = [f.split(".")[0] for f in (self.key_field, self.group_field) if f]
        return sorted(set(fields)) if fields else False

    def _query(
        self,
        vector: Sequence[float],
        k: int,
        params: dict[str, Any],
        exclude: int | str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "query": [float(x) for x in vector],
            "limit": k,
            "params": params,
            "with_payload": self._payload(),
        }
        if self.vector:
            body["using"] = self.vector
        if exclude is not None:
            body["filter"] = {"must_not": [{"has_id": [exclude]}]}
        return self.client.query(self.collection, body)

    def search(self, vector: Sequence[float], k: int, setting: int | None = None) -> list[Hit]:
        params = {"hnsw_ef": int(setting)} if setting is not None else {}
        return [
            (self._key(p), self._distance(float(p["score"])))
            for p in self._query(vector, k, params)
        ]

    def exact(self, vector: Sequence[float], k: int) -> list[Hit]:
        points = self._query(vector, k, {"exact": True})
        return [(self._key(p), self._distance(float(p["score"]))) for p in points]

    def index(self) -> tuple[str | None, int | None, list[int]]:
        """HNSW unless it's turned off (m = 0). Without an ``hnsw_ef``, searches use the
        collection's ``ef_construct``."""
        if self._index is None:
            info = self.client.collection(self.collection)
            hnsw = (info.get("config") or {}).get("hnsw_config") or {}
            vectors = ((info.get("config") or {}).get("params") or {}).get("vectors") or {}
            own = (vectors.get(self.vector) or {}) if self.vector else vectors
            hnsw = {**hnsw, **(own.get("hnsw_config") or {})}
            if hnsw.get("m") == 0:
                self._index = (None, None, [])
            else:
                current = int(hnsw.get("ef_construct") or 100)
                self._index = (METHOD, current, sorted({current, *SWEEP}))
        return self._index

    def _points(self, keys: Sequence[str]) -> list[dict[str, Any]]:
        return self.client.vectors(self.collection, self.to_points(keys), self.vector)

    def neighbours(self, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        own = self._points([key])
        values = _vector(own[0], self.vector) if own else None
        if not own or values is None:
            return None, []
        group = self._group(own[0])
        found = self._query(values, k, {"exact": True}, exclude=own[0]["id"])
        return group, [(self._key(p), self._group(p) or "") for p in found]

    def _group(self, point: dict[str, Any]) -> str | None:
        if not self.group_field:
            return None
        value = _get(point.get("payload") or {}, self.group_field)
        return None if value is None else str(value)

    def vectors(self, keys: Sequence[str]) -> list[list[float]]:
        found = []
        for point in self._points(keys):
            values = _vector(point, self.vector)
            if values is not None:
                found.append(values)
        return found


class QdrantRows:
    """Sample texts from a collection whose payload holds them."""

    def __init__(self, side: QdrantSide, text_field: str) -> None:
        self.side = side
        self.text_field = text_field

    def sample(self, text: str, size: int) -> list[tuple[str, str]]:
        side = self.side
        total = int(side.client.collection(side.collection).get("points_count") or 0)
        points, _ = _points(side.client, side.collection, False, size, total)
        rows = []
        for point in points:
            value = _get(point.get("payload") or {}, self.text_field)
            if isinstance(value, str) and value.strip():
                rows.append((side._key(point), value))
        return sorted(rows)[:size]

    def translate(self, keys: Sequence[str]) -> dict[str, str]:
        """Labeled queries name points by their IDs, which is how eval names them too."""
        return {k: k for k in keys}

    def texts(self, text: str, keys: Sequence[str]) -> dict[str, str]:
        found = {}
        for point in self.side._points(keys):
            value = _get(point.get("payload") or {}, self.text_field)
            found[self.side._key(point)] = value if isinstance(value, str) else ""
        return found


class PgSide:
    """One column of a pgvector table, through eval's PostgreSQL searcher."""

    def __init__(self, searcher: Any, side: Side = "old") -> None:
        self.searcher = searcher
        self.side = side

    def search(self, vector: Sequence[float], k: int, setting: int | None = None) -> list[Hit]:
        return list(self.searcher.search(self.side, vector, k, setting))

    def exact(self, vector: Sequence[float], k: int) -> list[Hit]:
        return list(self.searcher.exact(self.side, vector, k))

    def index(self) -> tuple[str | None, int | None, list[int]]:
        result: tuple[str | None, int | None, list[int]] = self.searcher.index(self.side)
        return result

    def neighbours(self, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        result: tuple[str | None, list[tuple[str, str]]] = self.searcher.neighbours(
            self.side, key, k
        )
        return result

    def vectors(self, keys: Sequence[str]) -> list[list[float]]:
        return list(self.searcher.vectors(self.side, keys))


class SplitSearcher:
    """eval's searcher over two sides, with rows (key and text) from one place."""

    def __init__(self, rows: Rows, sides: Mapping[Side, SideSearcher]) -> None:
        self.rows = rows
        self.sides = dict(sides)

    def sample(self, text: str, size: int) -> list[tuple[str, str]]:
        return self.rows.sample(text, size)

    def texts(self, text: str, keys: Sequence[str]) -> dict[str, str]:
        return self.rows.texts(text, keys)

    def translate(self, keys: Sequence[str]) -> dict[str, str]:
        return self.rows.translate(keys)

    def search(
        self, side: Side, vector: Sequence[float], k: int, setting: int | None = None
    ) -> list[Hit]:
        return self.sides[side].search(vector, k, setting)

    def exact(self, side: Side, vector: Sequence[float], k: int) -> list[Hit]:
        return self.sides[side].exact(vector, k)

    def index(self, side: Side) -> tuple[str | None, int | None, list[int]]:
        return self.sides[side].index()

    def neighbours(self, side: Side, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        return self.sides[side].neighbours(key, k)

    def vectors(self, side: Side, keys: Sequence[str]) -> list[list[float]]:
        return self.sides[side].vectors(keys)
