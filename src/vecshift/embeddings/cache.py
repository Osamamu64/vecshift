"""An on-disk embedding cache, so re-running a benchmark doesn't pay twice.

Keys are hashes of the model configuration, the mode, and the text; the text itself is
never stored. The cache lives in the user's cache directory, not the project.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path

from vecshift.embeddings.providers import Embedder, EmbedMode


def default_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "vecshift" / "embeddings.sqlite"


class EmbeddingCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key BLOB PRIMARY KEY, vector BLOB NOT NULL)"
        )

    @staticmethod
    def key(identity: str, mode: str, text: str) -> bytes:
        return hashlib.sha256(f"{identity}\0{mode}\0{text}".encode()).digest()

    def get_many(self, keys: Sequence[bytes]) -> dict[bytes, list[float]]:
        found: dict[bytes, list[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = list(keys[i : i + 500])
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            for key, blob in rows:
                found[key] = array("f", blob).tolist()
        return found

    def put_many(self, items: Sequence[tuple[bytes, Sequence[float]]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(k, array("f", v).tobytes()) for k, v in items],
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class CachedEmbedder:
    """Wraps an embedder; only texts missing from the cache reach the provider."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache
        self.hits = 0
        self.misses = 0

    async def embed(self, texts: Sequence[str], mode: EmbedMode = "document") -> list[list[float]]:
        identity = self.inner.spec.identity
        keys = [EmbeddingCache.key(identity, mode, t) for t in texts]
        found = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = await self.inner.embed([texts[i] for i in missing], mode)
            self.cache.put_many([(keys[i], v) for i, v in zip(missing, vectors, strict=True)])
            for i, v in zip(missing, vectors, strict=True):
                found[keys[i]] = v
        return [found[k] for k in keys]
