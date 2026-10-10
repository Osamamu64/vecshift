"""Read-only vector searches over a migration's old and new columns, for ``vecshift eval``.

Each search runs in its own read-only transaction with a statement timeout, and settings
such as ``hnsw.ef_search`` are set for that transaction only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import psycopg
from psycopg import sql

from vecshift.connectors.pgvector.target import OPCLASS_SUFFIX

Side = Literal["old", "new"]

OPERATORS = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}
SETTINGS = {"hnsw": "hnsw.ef_search", "ivfflat": "ivfflat.probes"}
DEFAULTS = {"hnsw": 40, "ivfflat": 1}
SWEEP = {"hnsw": (40, 100, 200), "ivfflat": (1, 5, 10, 20)}
TIMEOUT_MS = 60_000


class PgSearcher:
    def __init__(
        self,
        conn: psycopg.Connection,
        *,
        schema: str,
        table: str,
        pk: str,
        columns: Mapping[Side, str],
        types: Mapping[Side, str],
        extension_schema: str,
        metric: str,
        partial: bool = False,
        group: sql.Composable | None = None,
        timeout_ms: int = TIMEOUT_MS,
    ) -> None:
        if metric not in OPERATORS:
            raise ValueError(f"unsupported metric {metric!r}")
        if not set(types.values()) <= {"vector", "halfvec"}:
            raise ValueError("unsupported vector type")
        self.conn = conn
        self.table = sql.Identifier(schema, table)
        self.pk = sql.Identifier(pk)
        self.columns = {side: sql.Identifier(name) for side, name in columns.items()}
        self.names = dict(columns)
        self.types = {side: sql.Identifier(extension_schema, kind) for side, kind in types.items()}
        self.operator = sql.SQL("OPERATOR({}.{})").format(
            sql.Identifier(extension_schema), sql.SQL(OPERATORS[metric])
        )
        self.metric = metric
        self.partial = partial
        self.group = group if group is not None else sql.SQL("NULL")
        self.timeout_ms = int(timeout_ms)
        self._indexes: dict[Side, tuple[str | None, int | None, list[int]]] = {}

    # --- plumbing

    def _run(
        self, query: sql.SQL | sql.Composed, params: Sequence[Any], settings: Mapping[str, str]
    ) -> list[tuple[Any, ...]]:
        try:
            self.conn.execute(
                "SELECT set_config('statement_timeout', %s, true)", (str(self.timeout_ms),)
            )
            for name, value in settings.items():
                self.conn.execute("SELECT set_config(%s, %s, true)", (name, value))
            return self.conn.execute(query, params).fetchall()
        finally:
            self.conn.rollback()

    @staticmethod
    def _literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(repr(float(x)) for x in vector) + "]"

    def _filter(self, side: Side) -> sql.Composable:
        if self.partial:
            return sql.SQL("{} IS NOT NULL AND {} IS NOT NULL").format(
                self.columns["old"], self.columns["new"]
            )
        return sql.SQL("{} IS NOT NULL").format(self.columns[side])

    def _query(
        self, side: Side, *, where: sql.Composable | None, select: sql.Composable
    ) -> sql.Composed:
        clause = sql.SQL(" WHERE {}").format(where) if where is not None else sql.SQL("")
        return sql.SQL(
            "SELECT {select} FROM {table}{where} ORDER BY {col} {op} %s::{type} LIMIT %s"
        ).format(
            select=select,
            table=self.table,
            where=clause,
            col=self.columns[side],
            op=self.operator,
            type=self.types[side],
        )

    # --- rows

    def sample(self, text: str, size: int) -> list[tuple[str, str]]:
        """Up to ``size`` (key, text) rows spread across the table, with both vectors."""
        text_id = sql.Identifier(text)
        estimate = self._run(
            sql.SQL("SELECT reltuples::bigint FROM pg_class WHERE oid = %s::regclass"),
            (self.table.as_string(self.conn),),
            {},
        )
        rows_estimate = int(estimate[0][0]) if estimate and estimate[0][0] > 0 else 0
        tablesample: sql.Composable = sql.SQL("")
        if rows_estimate > size * 4:
            percent = min(100.0, 100.0 * size * 4 / rows_estimate)
            tablesample = sql.SQL(" TABLESAMPLE SYSTEM ({}) REPEATABLE (7)").format(
                sql.Literal(percent)
            )
        query = sql.SQL(
            "SELECT {pk}::text, {t}::text FROM {table}{ts} WHERE {both} AND {t} IS NOT NULL "
            "AND btrim({t}::text) <> '' ORDER BY md5({pk}::text) LIMIT %s"
        ).format(
            pk=self.pk,
            t=text_id,
            table=self.table,
            ts=tablesample,
            both=sql.SQL("{} IS NOT NULL AND {} IS NOT NULL").format(
                self.columns["old"], self.columns["new"]
            ),
        )
        return [(str(k), str(t)) for k, t in self._run(query, (size,), {})]

    def texts(self, text: str, keys: Sequence[str]) -> dict[str, str]:
        """The text of the given rows, by key."""
        if not keys:
            return {}
        query = sql.SQL(
            "SELECT {pk}::text, {t}::text FROM {table} WHERE {pk}::text = ANY(%s)"
        ).format(pk=self.pk, t=sql.Identifier(text), table=self.table)
        return {str(k): str(t or "") for k, t in self._run(query, (list(keys),), {})}

    # --- searches

    def search(
        self, side: Side, vector: Sequence[float], k: int, setting: int | None = None
    ) -> list[str]:
        """Top ``k`` keys through the index, as the application's queries would run."""
        settings = {}
        method = self.index(side)[0]
        if setting is not None and method:
            settings[SETTINGS[method]] = str(int(setting))
        query = self._query(side, where=None, select=sql.SQL("{}::text").format(self.pk))
        rows = self._run(query, (self._literal(vector), k), settings)
        return [r[0] for r in rows]

    def exact(self, side: Side, vector: Sequence[float], k: int) -> list[str]:
        """Top ``k`` keys by scanning every row: the true answer an index approximates."""
        query = self._query(
            side, where=self._filter(side), select=sql.SQL("{}::text").format(self.pk)
        )
        settings = {"enable_indexscan": "off", "enable_bitmapscan": "off"}
        rows = self._run(query, (self._literal(vector), k), settings)
        return [r[0] for r in rows]

    def index(self, side: Side) -> tuple[str | None, int | None, list[int]]:
        """The ANN index on a side's column, its current search setting, and a sweep."""
        if side in self._indexes:
            return self._indexes[side]
        rows = self._run(
            sql.SQL(
                """
                SELECT am.amname, opc.opcname FROM pg_index i
                JOIN pg_class ic ON ic.oid = i.indexrelid
                JOIN pg_am am ON am.oid = ic.relam
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                JOIN pg_opclass opc ON opc.oid = i.indclass[0]
                WHERE i.indrelid = %s::regclass AND a.attname = %s AND i.indisvalid
                  AND am.amname IN ('hnsw', 'ivfflat')
                """
            ),
            (self.table.as_string(self.conn), self.names[side]),
            {},
        )
        suffix = OPCLASS_SUFFIX[self.metric]
        method = next((str(r[0]) for r in rows if str(r[1]).endswith(suffix)), None)
        if method is None:
            self._indexes[side] = (None, None, [])
            return self._indexes[side]
        current_row = self._run(
            sql.SQL("SELECT current_setting(%s, true)"), (SETTINGS[method],), {}
        )
        raw = current_row[0][0] if current_row else None
        current = int(raw) if raw and str(raw).isdigit() else DEFAULTS[method]
        sweep = sorted({current, *SWEEP[method]})
        self._indexes[side] = (method, current, sweep)
        return self._indexes[side]

    def neighbours(self, side: Side, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        """A row's own group and its ``k`` nearest other rows, using that row's stored vector."""
        own = self._run(
            sql.SQL("SELECT {col}::text, ({group})::text FROM {table} WHERE {pk} = %s").format(
                col=self.columns[side], group=self.group, table=self.table, pk=self.pk
            ),
            (key,),
            {},
        )
        if not own or own[0][0] is None:
            return None, []
        vector, group = own[0]
        where = sql.SQL("{} <> %s AND {}").format(self.pk, self._filter(side))
        query = sql.SQL(
            "SELECT {pk}::text, ({group})::text FROM {table} WHERE {where} "
            "ORDER BY {col} {op} %s::{type} LIMIT %s"
        ).format(
            pk=self.pk,
            group=self.group,
            table=self.table,
            where=where,
            col=self.columns[side],
            op=self.operator,
            type=self.types[side],
        )
        settings = {"enable_indexscan": "off", "enable_bitmapscan": "off"} if self.partial else {}
        rows = self._run(query, (key, vector, k), settings)
        return group, [(str(r[0]), r[1]) for r in rows]

    def vectors(self, side: Side, keys: Sequence[str]) -> list[list[float]]:
        """Stored vectors of the given rows."""
        if not keys:
            return []
        query = sql.SQL(
            "SELECT {col}::text FROM {table} WHERE {pk}::text = ANY(%s) AND {col} IS NOT NULL"
        ).format(col=self.columns[side], table=self.table, pk=self.pk)
        rows = self._run(query, (list(keys),), {})
        return [[float(x) for x in str(r[0]).strip("[]").split(",")] for r in rows]
