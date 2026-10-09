"""The statements ``vecshift apply`` runs against PostgreSQL.

Everything here writes, so every statement is composed with ``psycopg.sql`` and runs under
a lock timeout: if a lock isn't free within a few seconds the statement gives up and retries
later, instead of queueing behind application queries and blocking them.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import errors, sql

from vecshift.connectors.pgvector.target import OPCLASS_SUFFIX

LOCK_TIMEOUT_MS = 5_000
STATEMENT_TIMEOUT_MS = 120_000
SCAN_TIMEOUT_MS = 30 * 60_000
"""For full-table counts, which can take a while on large tables."""


class Busy(Exception):
    """A lock couldn't be taken in time, even after retries."""


class AlreadyRunning(Exception):
    """Another apply holds the lock for this table and column."""


def _limits(
    executor: psycopg.Connection | psycopg.Cursor[Any],
    statement_ms: int,
    lock_ms: int | None = None,
) -> None:
    """Set timeouts for the current transaction only (like ``SET LOCAL``), as parameters."""
    executor.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(statement_ms)),))
    if lock_ms is not None:
        executor.execute("SELECT set_config('lock_timeout', %s, true)", (str(int(lock_ms)),))


@dataclass(frozen=True, slots=True)
class Layout:
    """Where things are. Names are raw identifiers; they're quoted when composed."""

    schema: str
    table: str
    pk: str
    text: str
    target: str
    vector_type: str
    dims: int
    extension_schema: str
    live: str = ""
    """The column the application searches. Cutover gives the new vectors this name."""

    @property
    def qualified(self) -> sql.Identifier:
        return sql.Identifier(self.schema, self.table)

    @property
    def previous(self) -> str:
        """Where cutover keeps the old vectors, for rollback."""
        return f"{self.live}_old"[:63]

    def trigger_name(self, column: str | None = None) -> str:
        return f"vecshift_sync_{column or self.target}"[:63]

    def function_name(self, column: str | None = None) -> str:
        return f"vecshift_sync_{self.table}_{column or self.target}"[:63]

    def index_name(self, method: str, column: str | None = None) -> str:
        return f"{self.table}_{column or self.target}_{method}_idx"[:63]

    @property
    def lock_key(self) -> int:
        digest = hashlib.sha256(f"vecshift:{self.schema}.{self.table}.{self.target}".encode())
        return int.from_bytes(digest.digest()[:8], "big", signed=True)


@dataclass(frozen=True, slots=True)
class Row:
    key: Any
    """The primary key value, as the database returned it."""
    text: str

    @property
    def id(self) -> str:
        return str(self.key)


class PgWriter:
    def __init__(
        self,
        conn: psycopg.Connection,
        layout: Layout,
        *,
        lock_retries: int = 6,
        lock_timeout_ms: int = LOCK_TIMEOUT_MS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not conn.autocommit:
            raise ValueError("PgWriter needs an autocommit connection")
        if layout.vector_type not in {"vector", "halfvec"}:
            raise ValueError(f"unsupported vector type {layout.vector_type!r}")
        self.conn = conn
        self.layout = layout
        self.lock_retries = lock_retries
        self.lock_timeout_ms = int(lock_timeout_ms)
        self._sleep = sleep

    # --- plumbing

    def _locked(self, statements: Sequence[sql.Composed | sql.SQL], what: str) -> None:
        """Run statements in one transaction under a lock timeout, retrying when busy."""
        delay = 1.0
        for attempt in range(1, self.lock_retries + 1):
            try:
                with self.conn.transaction():
                    _limits(self.conn, STATEMENT_TIMEOUT_MS, self.lock_timeout_ms)
                    for statement in statements:
                        self.conn.execute(statement)
                return
            except errors.LockNotAvailable as exc:
                if attempt == self.lock_retries:
                    raise Busy(
                        f"Couldn't {what}: the table stayed locked by other sessions. "
                        "Try again when it's quieter."
                    ) from exc
                self._sleep(delay)
                delay = min(delay * 2, 30)

    def _type(self) -> sql.Composable:
        lay = self.layout
        return sql.SQL("{}({})").format(
            sql.Identifier(lay.extension_schema, lay.vector_type), sql.Literal(lay.dims)
        )

    def _pending_filter(self) -> sql.Composable:
        lay = self.layout
        return sql.SQL(
            "{target} IS NULL AND {text} IS NOT NULL AND btrim({text}::text) <> ''"
        ).format(target=sql.Identifier(lay.target), text=sql.Identifier(lay.text))

    # --- concurrency guard

    def acquire(self) -> None:
        """Make sure no other apply works on the same column at the same time.

        Session-level advisory locks need a real session, so apply can't run through a
        transaction pooler.
        """
        row = self.conn.execute(
            "SELECT pg_try_advisory_lock(%s)", (self.layout.lock_key,)
        ).fetchone()
        if not row or not row[0]:
            raise AlreadyRunning(
                f"Another vecshift apply is already working on {self.layout.table}."
                f"{self.layout.target}."
            )

    def release(self) -> None:
        self.conn.execute("SELECT pg_advisory_unlock(%s)", (self.layout.lock_key,))

    # --- schema changes

    def column_exists(self) -> bool:
        row = self.conn.execute(
            """
            SELECT 1 FROM pg_attribute
            WHERE attrelid = %s::regclass AND attname = %s AND attnum > 0 AND NOT attisdropped
            """,
            (self._regclass(), self.layout.target),
        ).fetchone()
        return row is not None

    def _regclass(self) -> str:
        return sql.Identifier(self.layout.schema, self.layout.table).as_string(self.conn)

    def ensure_column(self) -> bool:
        """Add the target column if it's missing. Returns whether it was added."""
        if self.column_exists():
            return False
        lay = self.layout
        statement = sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {type}").format(
            table=lay.qualified, col=sql.Identifier(lay.target), type=self._type()
        )
        self._locked([statement], "add the new column")
        return True

    def trigger_exists(self, column: str | None = None) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM pg_trigger WHERE tgrelid = %s::regclass AND tgname = %s",
            (self._regclass(), self.layout.trigger_name(column)),
        ).fetchone()
        return row is not None

    def sync_function_sql(self, column: str | None = None) -> sql.Composed:
        """A trigger function that clears ``column`` when a row's text changes.

        If the same update also writes ``column`` itself (an application already writing
        the new model's vectors), that value is kept.
        """
        lay = self.layout
        col = sql.Identifier(column or lay.target)
        tag = sql.SQL("$vs_" + secrets.token_hex(4) + "$")
        body = sql.SQL(
            "BEGIN IF NEW.{text} IS DISTINCT FROM OLD.{text} "
            "AND NEW.{col} IS NOT DISTINCT FROM OLD.{col} THEN NEW.{col} := NULL; END IF; "
            "RETURN NEW; END"
        ).format(text=sql.Identifier(lay.text), col=col)
        # A fixed search path finds pgvector's operators wherever the extension lives, and
        # stops other schemas from shadowing them.
        return sql.SQL(
            "CREATE OR REPLACE FUNCTION {fn}() RETURNS trigger LANGUAGE plpgsql "
            "SET search_path = pg_catalog, {ext} AS {tag}{body}{tag}"
        ).format(
            fn=sql.Identifier(lay.schema, lay.function_name(column)),
            ext=sql.Identifier(lay.extension_schema),
            tag=tag,
            body=body,
        )

    def sync_trigger_sql(self, column: str | None = None) -> sql.Composed:
        # Listing the guarded column makes the trigger depend on it, so dropping the column
        # by hand fails clearly instead of leaving a trigger that breaks every update.
        lay = self.layout
        return sql.SQL(
            "CREATE TRIGGER {name} BEFORE UPDATE OF {text}, {col} ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION {fn}()"
        ).format(
            name=sql.Identifier(lay.trigger_name(column)),
            text=sql.Identifier(lay.text),
            col=sql.Identifier(column or lay.target),
            table=lay.qualified,
            fn=sql.Identifier(lay.schema, lay.function_name(column)),
        )

    def drop_sync_sql(self, column: str | None = None) -> list[sql.Composed]:
        lay = self.layout
        return [
            sql.SQL("DROP TRIGGER IF EXISTS {name} ON {table}").format(
                name=sql.Identifier(lay.trigger_name(column)), table=lay.qualified
            ),
            sql.SQL("DROP FUNCTION IF EXISTS {fn}()").format(
                fn=sql.Identifier(lay.schema, lay.function_name(column))
            ),
        ]

    def ensure_trigger(self) -> bool:
        """Clear a row's new vector whenever its text changes, so it gets re-embedded.

        Returns whether the trigger was created. The function is always replaced, which
        needs no lock on the table, so older installs pick up fixes.
        """
        if self.trigger_exists():
            with self.conn.transaction():
                _limits(self.conn, STATEMENT_TIMEOUT_MS, self.lock_timeout_ms)
                self.conn.execute(self.sync_function_sql())
            return False
        self._locked([self.sync_function_sql(), self.sync_trigger_sql()], "add the sync trigger")
        return True

    # --- backfill

    def total_count(self) -> int:
        """Rows with text, which should all end up with a new vector."""
        lay = self.layout
        query = sql.SQL(
            "SELECT count(*) FROM {table} WHERE {text} IS NOT NULL AND btrim({text}::text) <> ''"
        ).format(table=lay.qualified, text=sql.Identifier(lay.text))
        with self.conn.transaction():
            _limits(self.conn, SCAN_TIMEOUT_MS)
            row = self.conn.execute(query).fetchone()
        return int(row[0]) if row else 0

    def pending_count(self, exclude: Sequence[str] = ()) -> int:
        lay = self.layout
        query = sql.SQL("SELECT count(*) FROM {table} WHERE {pending}").format(
            table=lay.qualified, pending=self._pending_filter()
        )
        params: tuple[Any, ...] = ()
        if exclude:
            query += sql.SQL(" AND NOT ({pk}::text = ANY(%s))").format(pk=sql.Identifier(lay.pk))
            params = (list(exclude),)
        with self.conn.transaction():
            _limits(self.conn, SCAN_TIMEOUT_MS)
            row = self.conn.execute(query, params).fetchone()
        return int(row[0]) if row else 0

    def fetch(self, after: Any, limit: int, exclude: Sequence[str] = ()) -> list[Row]:
        """The next rows that need a vector, in primary-key order after ``after``."""
        lay = self.layout
        pk = sql.Identifier(lay.pk)
        query = sql.SQL("SELECT {pk}, {text}::text FROM {table} WHERE {pending}").format(
            pk=pk,
            text=sql.Identifier(lay.text),
            table=lay.qualified,
            pending=self._pending_filter(),
        )
        params: list[Any] = []
        if after is not None:
            query += sql.SQL(" AND {pk} > %s").format(pk=pk)
            params.append(after)
        if exclude:
            query += sql.SQL(" AND NOT ({pk}::text = ANY(%s))").format(pk=pk)
            params.append(list(exclude))
        query += sql.SQL(" ORDER BY {pk} LIMIT %s").format(pk=pk)
        params.append(limit)
        with self.conn.transaction():
            _limits(self.conn, STATEMENT_TIMEOUT_MS)
            rows = self.conn.execute(query, params).fetchall()
        return [Row(key, text) for key, text in rows]

    def write(self, items: Sequence[tuple[Row, Sequence[float]]]) -> int:
        """Store vectors, but only where the row still needs one and its text is unchanged.

        Returns how many rows were written. A row whose text changed after it was read is
        left empty, so a later pass embeds the new text.
        """
        if not items:
            return 0
        lay = self.layout
        statement = sql.SQL(
            "UPDATE {table} SET {target} = %s::{type} "
            "WHERE {pk} = %s AND {target} IS NULL AND {text}::text = %s"
        ).format(
            table=lay.qualified,
            target=sql.Identifier(lay.target),
            type=self._type(),
            pk=sql.Identifier(lay.pk),
            text=sql.Identifier(lay.text),
        )
        params = [
            ("[" + ",".join(repr(float(x)) for x in vector) + "]", row.key, row.text)
            for row, vector in items
        ]
        written = 0
        delay = 1.0
        for attempt in range(1, self.lock_retries + 1):
            try:
                with self.conn.transaction(), self.conn.cursor() as cur:
                    _limits(cur, STATEMENT_TIMEOUT_MS, self.lock_timeout_ms)
                    written = 0
                    for p in params:
                        cur.execute(statement, p)
                        written += cur.rowcount
                return written
            except errors.LockNotAvailable as exc:
                if attempt == self.lock_retries:
                    raise Busy(
                        "Couldn't write vectors: rows stayed locked by other sessions."
                    ) from exc
                self._sleep(delay)
                delay = min(delay * 2, 30)
        return written  # pragma: no cover

    # --- index

    def index_state(self, method: str) -> str | None:
        """``valid``, ``invalid`` (a failed concurrent build), or ``None`` when absent."""
        row = self.conn.execute(
            """
            SELECT i.indisvalid FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
            WHERE i.indrelid = %s::regclass AND c.relname = %s AND a.attname = %s
            """,
            (self._regclass(), self.layout.index_name(method), self.layout.target),
        ).fetchone()
        if row is None:
            return None
        return "valid" if row[0] else "invalid"

    def build_index(self, method: str, metric: str, rows: int) -> str:
        """Build the vector index concurrently, so writes continue. Returns what happened."""
        if method not in {"hnsw", "ivfflat"} or metric not in OPCLASS_SUFFIX:
            raise ValueError(f"unsupported index {method!r} / {metric!r}")
        lay = self.layout
        name = sql.Identifier(lay.schema, lay.index_name(method))
        state = self.index_state(method)
        if state == "valid":
            return "exists"
        # Concurrent builds can't run in a transaction and can take a long time, so the
        # session's timeouts are lifted for these statements only.
        self.conn.execute("SET statement_timeout = 0")
        self.conn.execute("SET lock_timeout = 0")
        try:
            if state == "invalid":
                self.conn.execute(sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(name))
            statement = sql.SQL(
                "CREATE INDEX CONCURRENTLY {name} ON {table} USING {method} ({col} {ops})"
            ).format(
                name=sql.Identifier(lay.index_name(method)),
                table=lay.qualified,
                method=sql.SQL(method),
                col=sql.Identifier(lay.target),
                ops=sql.Identifier(
                    lay.extension_schema, f"{lay.vector_type}_{OPCLASS_SUFFIX[metric]}"
                ),
            )
            if method == "ivfflat":
                lists = max(1, rows // 1000 if rows <= 1_000_000 else int(rows**0.5))
                statement += sql.SQL(" WITH (lists = {})").format(sql.Literal(lists))
            self.conn.execute(statement)
        finally:
            self.conn.execute("RESET statement_timeout")
            self.conn.execute("RESET lock_timeout")
        return "rebuilt" if state == "invalid" else "built"

    # --- verification

    def dimensions_in_use(self) -> set[int]:
        lay = self.layout
        query = sql.SQL(
            "SELECT DISTINCT {fn}({target}) FROM (SELECT {target} FROM {table} "
            "WHERE {target} IS NOT NULL LIMIT 1000) s"
        ).format(
            fn=sql.Identifier(lay.extension_schema, "vector_dims"),
            target=sql.Identifier(lay.target),
            table=lay.qualified,
        )
        return {int(r[0]) for r in self.conn.execute(query).fetchall()}
