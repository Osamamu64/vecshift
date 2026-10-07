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


def quote_ident(name: str) -> str:
    """Quote an identifier for display the way PostgreSQL needs it."""
    return name if _SIMPLE.match(name) else '"' + name.replace('"', '""') + '"'


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
    )


def resolve_source_column(
    conn: psycopg.Connection, table: str, vector_column: str | None, target_column: str
) -> str:
    """The source vector column, ignoring the target column a previous run may have added."""
    if vector_column:
        return vector_column
    columns = find_vector_columns(conn)
    schema, _, name = table.rpartition(".")
    in_table = [
        c
        for c in columns
        if c.table == name and (not schema or c.schema == schema) and c.column != target_column
    ]
    if len(in_table) == 1:
        return in_table[0].column
    return select_column(columns, table, None).column  # raises a helpful error


def add_column_sql(state: TargetState, column: str, vector_type: str, dims: int) -> str:
    table = f"{quote_ident(state.source.schema)}.{quote_ident(state.source.table)}"
    ext = quote_ident(state.extension_schema)
    return f"ALTER TABLE {table} ADD COLUMN {quote_ident(column)} {ext}.{vector_type}({dims});"


def index_sql(
    state: TargetState, column: str, vector_type: str, method: str, metric: str, rows: int
) -> str:
    table = f"{quote_ident(state.source.schema)}.{quote_ident(state.source.table)}"
    name = quote_ident(f"{state.source.table}_{column}_{method}_idx"[:63])
    opclass = f"{quote_ident(state.extension_schema)}.{vector_type}_{OPCLASS_SUFFIX[metric]}"
    target = f"{quote_ident(column)} {opclass}"
    using = f"CREATE INDEX CONCURRENTLY {name} ON {table} USING {method} ({target})"
    if method == "ivfflat":
        # pgvector's guidance: rows / 1000 lists up to a million rows, sqrt(rows) beyond.
        lists = max(1, rows // 1000 if rows <= 1_000_000 else int(rows**0.5))
        using += f" WITH (lists = {lists})"
    return using + ";"
