"""A small client for Qdrant's REST API: reads for doctor and bench, and the few writes
``vecshift apply`` makes to a new collection.

The API key comes from an environment variable only. It is sent only over HTTPS, or over
plain HTTP to this machine, redirects are never followed, and no message ever includes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from vecshift.embeddings.spec import LOCAL_HOSTS

DEFAULT_KEY_ENV = "QDRANT_API_KEY"
TIMEOUT_S = 30.0


class QdrantError(Exception):
    """A request failed. The message is safe to show: it never includes the API key."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class QdrantSettings:
    url: str
    """Base URL, without a trailing slash, query string, or credentials."""
    key_env: str
    key: str | None

    @property
    def display(self) -> str:
        return self.url

    @property
    def description(self) -> str:
        return self.url


def prepare(url: str, key_env: str = DEFAULT_KEY_ENV) -> QdrantSettings:
    """Check a Qdrant URL, and read its API key from the environment."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise QdrantError(
            f"That doesn't look like a Qdrant URL: {url!r}.",
            hint="Expected something like http://localhost:6333 or https://xyz.cloud.qdrant.io",
        )
    if parsed.username or parsed.password:
        raise QdrantError(
            "Don't put credentials in the Qdrant URL.",
            hint=f"Put the API key in the {key_env} environment variable instead.",
        )
    if parsed.query or parsed.fragment:
        raise QdrantError("The Qdrant URL can't have a query string or fragment.")
    key = os.environ.get(key_env) or None
    local = (parsed.hostname or "").lower() in LOCAL_HOSTS
    if key and parsed.scheme == "http" and not local:
        raise QdrantError(
            "Won't send the Qdrant API key over plain http to a remote host.",
            hint="Use the https:// URL. Qdrant Cloud always serves https.",
        )
    port = f":{parsed.port}" if parsed.port else ""
    base = f"{parsed.scheme}://{parsed.hostname}{port}{parsed.path.rstrip('/')}"
    return QdrantSettings(url=base, key_env=key_env, key=key)


class QdrantClient:
    def __init__(
        self,
        settings: QdrantSettings,
        *,
        timeout: float = TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings
        headers = {"api-key": settings.key} if settings.key else {}
        self._http = httpx.Client(
            base_url=settings.url,
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> QdrantClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- plumbing

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        wait: bool = False,
    ) -> Any:
        where = self.settings.display
        params = {"wait": "true"} if wait else None
        try:
            response = self._http.request(method, path, json=body, params=params)
        except httpx.TimeoutException as exc:
            raise QdrantError(f"Qdrant at {where} didn't answer in time.") from exc
        except httpx.HTTPError as exc:
            raise QdrantError(
                f"Couldn't reach Qdrant at {where}: {type(exc).__name__}.",
                hint="Check the URL and port (6333 for REST), and that Qdrant is running.",
            ) from exc
        if response.status_code in {401, 403}:
            raise QdrantError(
                "Qdrant refused the request: the API key is missing or wrong.",
                hint=f"Set {self.settings.key_env} to a key with read access.",
            )
        if response.is_redirect:
            raise QdrantError(f"Qdrant at {where} answered with a redirect, which isn't followed.")
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code >= 400:
            detail = ""
            if isinstance(data, dict):
                status = data.get("status")
                if isinstance(status, dict) and status.get("error"):
                    detail = f": {str(status['error'])[:200]}"
            raise QdrantError(f"Qdrant returned {response.status_code}{detail}")
        if not isinstance(data, dict):
            raise QdrantError(f"Qdrant at {where} sent a response vecshift doesn't understand.")
        return data.get("result")

    # --- reads

    def version(self) -> str:
        response = self._request("GET", "/")
        return str(response) if response is not None else ""

    def collections(self) -> list[str]:
        result = self._request("GET", "/collections") or {}
        return sorted(str(c["name"]) for c in result.get("collections", []))

    def collection(self, name: str) -> dict[str, Any]:
        result = self._request("GET", f"/collections/{_segment(name)}")
        if not isinstance(result, dict):
            raise QdrantError(f"Qdrant sent no details for collection {name!r}.")
        return result

    def aliases(self) -> list[tuple[str, str]]:
        """(alias, collection) pairs."""
        result = self._request("GET", "/aliases") or {}
        return [
            (str(a["alias_name"]), str(a["collection_name"])) for a in result.get("aliases", [])
        ]

    def count(self, name: str) -> int:
        result = self._request(
            "POST", f"/collections/{_segment(name)}/points/count", {"exact": True}
        )
        return int((result or {}).get("count", 0))

    def scroll(
        self,
        name: str,
        limit: int,
        offset: Any = None,
        *,
        vector: str | bool = False,
        payload: bool = True,
    ) -> tuple[list[dict[str, Any]], Any]:
        """A page of points and the offset of the next page (``None`` at the end)."""
        body: dict[str, Any] = {
            "limit": limit,
            "with_payload": payload,
            "with_vector": [vector] if isinstance(vector, str) else vector,
        }
        if offset is not None:
            body["offset"] = offset
        result = self._request("POST", f"/collections/{_segment(name)}/points/scroll", body) or {}
        return list(result.get("points", [])), result.get("next_page_offset")

    def retrieve(
        self, name: str, ids: list[int | str], *, payload: bool = True
    ) -> list[dict[str, Any]]:
        """The points with these IDs that exist, with their payloads."""
        if not ids:
            return []
        result = self._request(
            "POST",
            f"/collections/{_segment(name)}/points",
            {"ids": ids, "with_payload": payload, "with_vector": False},
        )
        return list(result or [])

    def exists(self, name: str) -> bool:
        """Whether a collection (not an alias) with this name exists."""
        return name in self.collections()

    def random_sample(
        self, name: str, limit: int, *, vector: str | bool = False
    ) -> list[dict[str, Any]] | None:
        """A random sample of points, or ``None`` if this Qdrant can't sample (before 1.11)."""
        body = {
            "query": {"sample": "random"},
            "limit": limit,
            "with_payload": True,
            "with_vector": [vector] if isinstance(vector, str) else vector,
        }
        try:
            result = self._request("POST", f"/collections/{_segment(name)}/points/query", body)
        except QdrantError as exc:
            if str(exc).startswith(("Qdrant returned 400", "Qdrant returned 404")):
                return None
            raise
        return list((result or {}).get("points", []))

    # --- writes

    def create_collection(self, name: str, vectors: dict[str, Any]) -> None:
        self._request("PUT", f"/collections/{_segment(name)}", {"vectors": vectors})

    def create_payload_index(self, name: str, field: str, schema: Any) -> None:
        self._request(
            "PUT",
            f"/collections/{_segment(name)}/index",
            {"field_name": field, "field_schema": schema},
            wait=True,
        )

    def upsert(self, name: str, points: list[dict[str, Any]]) -> None:
        if points:
            self._request(
                "PUT", f"/collections/{_segment(name)}/points", {"points": points}, wait=True
            )

    def batch(self, name: str, operations: list[dict[str, Any]]) -> None:
        """Several point updates (overwrite_payload, delete, ...) in one request."""
        if operations:
            self._request(
                "POST",
                f"/collections/{_segment(name)}/points/batch",
                {"operations": operations},
                wait=True,
            )

    def update_aliases(self, actions: list[dict[str, Any]]) -> None:
        """Alias changes, applied together: Qdrant applies one request atomically."""
        self._request("POST", "/collections/aliases", {"actions": actions})

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{_segment(name)}")

    def query(self, name: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Points nearest a vector (Qdrant 1.10+)."""
        result = self._request("POST", f"/collections/{_segment(name)}/points/query", body)
        return list((result or {}).get("points", []))

    def vectors(self, name: str, ids: list[int | str], vector: str | None) -> list[dict[str, Any]]:
        """The points with these IDs, with their vector (and payload)."""
        if not ids:
            return []
        result = self._request(
            "POST",
            f"/collections/{_segment(name)}/points",
            {"ids": ids, "with_payload": True, "with_vector": [vector] if vector else True},
        )
        return list(result or [])

    def delete_points(self, name: str, ids: list[int | str]) -> None:
        if ids:
            self.batch(name, [{"delete": {"points": ids}}])


def _segment(name: str) -> str:
    """A collection name as one URL path segment."""
    from urllib.parse import quote

    return quote(name, safe="")
