"""What the target of a side-by-side migration looks like, and the SQL to create it."""

from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg

from vecshift.connectors.pgvector.inspect import (
    VectorColumn,
    _extension_schema,
    find_vector_columns,
    select_column,
)

OPCLASS_SUFFIX = {"cosine": "cosine_ops", "inner_product": "ip_ops", "l2": "l2_ops"}

_SIMPLE = re.compile(r"^[a-z_][a-z0-9_$]*$")

# PostgreSQL's reserved and type/function-name keywords, used when the server's own list
# isn't available. They must be quoted to be used as names.
_KEYWORDS = (
    "all analyse analyze and any array as asc asymmetric authorization binary both case "
    "cast check collate collation column concurrently constraint create cross "
    "current_catalog current_date current_role current_schema current_time "
    "current_timestamp current_user default deferrable desc distinct do else end except "
    "false fetch for foreign freeze from full grant group having ilike in initially inner "
    "intersect into is isnull join lateral leading left like limit localtime "
    "localtimestamp natural not notnull null offset on only or order outer overlaps "
    "placing primary references returning right select session_user similar some "
    "symmetric system_user table tablesample then to trailing true union unique user "
    "using variadic verbose when where window with"
)
FALLBACK_KEYWORDS = frozenset(_KEYWORDS.split())


def quote_ident(name: str, keywords: frozenset[str] = FALLBACK_KEYWORDS) -> str:
    """Quote an identifier exactly when PostgreSQL needs it, like its quote_ident()."""
    if _SIMPLE.match(name) and name not in keywords:
        return name
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True, slots=True)
class TargetState:
    source: VectorColumn
    extension_schema: str
    owner: bool
    """Whether the connecting role owns the table, which adding a column requires."""
    column_exists: bool
    column_type: str | None
    column_dimensions: int | None
    maintenance_work_mem: int
    """Bytes available to an index build before it slows down."""
    keywords: frozenset[str] = FALLBACK_KEYWORDS
    """Words this server needs quoted when used as names."""
    primary_key: tuple[str, ...] = ()
    """Primary key columns, in order. Apply needs exactly one."""
    text_column: str | None = None
    previous_column_exists: bool = False
    """Whether ``<column>_old`` exists: a cutover happened and hasn't been cleaned up."""

    def q(self, name: str) -> str:
        return quote_ident(name, self.keywords)


def inspect_target(
    conn: psycopg.Connection, table: str, vector_column: str | None, target_column: str
) -> TargetState:
    source = select_column(find_vector_columns(conn), table, vector_column)
    ext = _extension_schema(conn)
    owner_row = conn.execute(
        "SELECT pg_has_role(current_user, relowner, 'MEMBER') FROM pg_class WHERE oid = %s",
        (source.relid,),
    ).fetchone()
    col = conn.execute(
        """
        SELECT t.typname, a.atttypmod FROM pg_attribute a
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE a.attrelid = %s AND a.attname = %s AND a.attnum > 0 AND NOT a.attisdropped
        """,
        (source.relid, target_column),
    ).fetchone()
    mem = conn.execute(
        "SELECT setting::bigint, unit FROM pg_settings WHERE name = 'maintenance_work_mem'"
    ).fetchone()
    words = conn.execute("SELECT word FROM pg_get_keywords() WHERE catcode <> 'U'").fetchall()
    pk_rows = conn.execute(
        """
        SELECT a.attname FROM pg_index i
        CROSS JOIN LATERAL unnest(i.indkey::int2[]) WITH ORDINALITY AS k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
        WHERE i.indrelid = %s AND i.indisprimary
        ORDER BY k.ord
        """,
        (source.relid,),
    ).fetchall()
    units = {"kB": 1024, "MB": 1024**2, "8kB": 8192, "B": 1}
    mem_bytes = int(mem[0]) * units.get(mem[1] or "kB", 1024) if mem else 64 * 1024**2
    return TargetState(
        source=source,
        extension_schema=ext,
        owner=bool(owner_row and owner_row[0]),
        column_exists=col is not None,
        column_type=str(col[0]) if col else None,
        column_dimensions=int(col[1]) if col and col[1] > 0 else None,
        maintenance_work_mem=mem_bytes,
        keywords=frozenset(w[0] for w in words) or FALLBACK_KEYWORDS,
        primary_key=tuple(str(r[0]) for r in pk_rows),
        previous_column_exists=conn.execute(
            """
            SELECT 1 FROM pg_attribute
            WHERE attrelid = %s AND attname = %s AND attnum > 0 AND NOT attisdropped
            """,
            (source.relid, f"{source.column}_old"[:63]),
        ).fetchone()
        is not None,
    )


def resolve_source_column(
    conn: psycopg.Connection, table: str, vector_column: str | None, target_column: str
) -> str:
    """The column the application searches.

    It ignores the target column a previous run may have added, and the ``<name>_old``
    column a cutover keeps for rollback.
    """
    if vector_column:
        return vector_column
    columns = find_vector_columns(conn)
    schema, _, name = table.rpartition(".")
    table_columns = [c for c in columns if c.table == name and (not schema or c.schema == schema)]
    names = {c.column for c in table_columns}
    in_table = [
        c
        for c in table_columns
        if c.column != target_column
        and not (c.column.endswith("_old") and c.column.removesuffix("_old") in names)
    ]
    if len(in_table) == 1:
        return in_table[0].column
    return select_column(columns, table, None).column  # raises a helpful error


def add_column_sql(state: TargetState, column: str, vector_type: str, dims: int) -> str:
    if vector_type not in {"vector", "halfvec"}:
        raise ValueError(f"unsupported vector type {vector_type!r}")
    table = f"{state.q(state.source.schema)}.{state.q(state.source.table)}"
    kind = f"{state.q(state.extension_schema)}.{state.q(vector_type)}({int(dims)})"
    return f"ALTER TABLE {table} ADD COLUMN {state.q(column)} {kind};"


def index_sql(
    state: TargetState, column: str, vector_type: str, method: str, metric: str, rows: int
) -> str:
    if method not in {"hnsw", "ivfflat"} or vector_type not in {"vector", "halfvec"}:
        raise ValueError(f"unsupported index {method!r} on {vector_type!r}")
    table = f"{state.q(state.source.schema)}.{state.q(state.source.table)}"
    name = state.q(f"{state.source.table}_{column}_{method}_idx"[:63])
    ops = state.q(f"{vector_type}_{OPCLASS_SUFFIX[metric]}")
    target = f"{state.q(column)} {state.q(state.extension_schema)}.{ops}"
    using = f"CREATE INDEX CONCURRENTLY {name} ON {table} USING {method} ({target})"
    if method == "ivfflat":
        # pgvector's guidance: rows / 1000 lists up to a million rows, sqrt(rows) beyond.
        lists = max(1, rows // 1000 if rows <= 1_000_000 else int(rows**0.5))
        using += f" WITH (lists = {lists})"
    return using + ";"
