"""Read a pgvector table in primary-key order, as the source of a migration to Qdrant.

Only reads, each in its own short read-only transaction, so a long migration never holds
a snapshot open. Rows become points with every non-vector column as their payload.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import psycopg
from psycopg import sql

from vecshift.connectors.pgvector.inspect import VECTOR_TYPES
from vecshift.connectors.pgvector.writer import STATEMENT_TIMEOUT_MS, _limits
from vecshift.connectors.qdrant.writer import Item

POINT_NAMESPACE = uuid.UUID("5c1f4a52-8a0e-4d7c-9a51-6f0b1d3c2e77")
"""Primary keys that aren't unsigned integers or UUIDs become UUIDv5 point IDs in this
namespace, so the same row always gets the same point."""


def point_id(key: Any) -> int | str:
    """The Qdrant point ID for a primary key value."""
    if isinstance(key, int) and not isinstance(key, bool) and 0 <= key < 2**64:
        return key
    if isinstance(key, uuid.UUID):
        return str(key)
    if isinstance(key, str):
        try:
            return str(uuid.UUID(key))
        except ValueError:
            pass
    return str(uuid.uuid5(POINT_NAMESPACE, str(key)))


class PgSource:
    def __init__(
        self,
        conn: psycopg.Connection,
        schema: str,
        table: str,
        pk: str,
        text: str,
    ) -> None:
        if not conn.autocommit:
            raise ValueError("PgSource needs an autocommit connection")
        self.conn = conn
        self.table = sql.Identifier(schema, table)
        self.pk = pk
        self.text_field = text
        row = conn.execute(
            """
            SELECT n.nspname, t.typname FROM pg_attribute a
            JOIN pg_type t ON t.oid = a.atttypid
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE a.attrelid = %s::regclass AND a.attname = %s
            """,
            (self.table.as_string(conn), pk),
        ).fetchone()
        if row is None:
            raise ValueError(f"no column {pk!r}")
        self.pk_type = sql.Identifier(row[0], row[1])
        vectors = conn.execute(
            """
            SELECT a.attname FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
            WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
              AND t.typname = ANY(%s)
            """,
            (self.table.as_string(conn), [*VECTOR_TYPES, "sparsevec"]),
        ).fetchall()
        self.vector_columns = sorted(str(r[0]) for r in vectors)

    def _has_text(self) -> sql.Composable:
        return sql.SQL("{t} IS NOT NULL AND btrim({t}::text) <> ''").format(
            t=sql.Identifier(self.text_field)
        )

    def _read(self, query: sql.Composed, params: Sequence[Any]) -> list[tuple[Any, ...]]:
        with self.conn.transaction():
            self.conn.execute("SET TRANSACTION READ ONLY")
            _limits(self.conn, STATEMENT_TIMEOUT_MS)
            return self.conn.execute(query, params).fetchall()

    def page(self, after: Any, limit: int) -> tuple[list[Item], Any]:
        pk = sql.Identifier(self.pk)
        query = sql.SQL(
            "SELECT {pk}, {text}::text, to_jsonb(r) - %s::text[] FROM {table} r WHERE {has}"
        ).format(
            pk=pk, text=sql.Identifier(self.text_field), table=self.table, has=self._has_text()
        )
        params: list[Any] = [self.vector_columns]
        if after is not None:
            query += sql.SQL(" AND {pk} > %s").format(pk=pk)
            params.append(after)
        query += sql.SQL(" ORDER BY {pk} LIMIT %s").format(pk=pk)
        params.append(limit)
        rows = self._read(query, params)
        items = [Item(key, text, point_id(key), payload) for key, text, payload in rows]
        return items, (items[-1].key if len(items) == limit else None)

    def count(self) -> int:
        query = sql.SQL("SELECT count(*) FROM {table} WHERE {has}").format(
            table=self.table, has=self._has_text()
        )
        rows = self._read(query, ())
        return int(rows[0][0]) if rows else 0

    def present(self, points: Sequence[dict[str, Any]]) -> set[int | str]:
        keys: dict[str, int | str] = {}
        for point in points:
            key = (point.get("payload") or {}).get(self.pk)
            if key is not None:
                keys[str(key)] = point["id"]
        if not keys:
            return set()
        # Cast the keys to the key's own type, so the lookup uses the primary key index.
        query = sql.SQL(
            "SELECT {pk}::text FROM {table} WHERE {pk} = ANY(%s::text[]::{type}[]) AND {has}"
        ).format(
            pk=sql.Identifier(self.pk), table=self.table, type=self.pk_type, has=self._has_text()
        )
        try:
            rows = self._read(query, (list(keys),))
        except psycopg.errors.InvalidTextRepresentation:
            return set()
        return {keys[str(r[0])] for r in rows if str(r[0]) in keys}
