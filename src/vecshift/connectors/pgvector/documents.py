"""Read a sample of documents (ID and text) from a pgvector table, for benchmarks."""

from __future__ import annotations

from collections.abc import Collection

import psycopg
from psycopg import sql

from vecshift.connectors.pgvector.inspect import (
    TEXT_COLUMNS,
    TEXT_TYPES,
    TargetSelectionError,
    _columns,
    _pick,
    find_vector_columns,
    select_column,
)


def _id_column(conn: psycopg.Connection, relid: int) -> str | None:
    row = conn.execute(
        """
        SELECT a.attname FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
        WHERE i.indrelid = %s AND i.indisprimary AND i.indnatts = 1
        """,
        (relid,),
    ).fetchone()
    return str(row[0]) if row else None


def sample_documents(
    conn: psycopg.Connection,
    table: str | None,
    column: str | None,
    text_column: str | None,
    size: int,
    include_ids: Collection[str] = (),
) -> tuple[list[tuple[str, str]], str]:
    """Up to ``size`` (id, text) rows spread across the table, plus any ``include_ids``.

    Returns the rows and the table's qualified name.
    """
    col = select_column(find_vector_columns(conn), table, column)
    columns = _columns(conn, col.relid)
    text = text_column or _pick(columns, TEXT_COLUMNS, TEXT_TYPES)
    if text is None or text not in columns:
        raise TargetSelectionError(
            f"No text column found in {col.schema}.{col.table}. Pass --text-column."
        )
    id_col = _id_column(conn, col.relid)
    id_expr = sql.Identifier(id_col) if id_col else sql.SQL("ctid")
    if include_ids and not id_col:
        raise TargetSelectionError(
            "Labeled queries need a single-column primary key to match document IDs."
        )
    table_id = sql.Identifier(col.schema, col.table)
    text_id = sql.Identifier(text)
    nonempty = sql.SQL("{t} IS NOT NULL AND btrim({t}::text) <> ''").format(t=text_id)

    est_row = conn.execute(
        "SELECT reltuples::bigint FROM pg_class WHERE oid = %s", (col.relid,)
    ).fetchone()
    estimated = int(est_row[0]) if est_row and est_row[0] > 0 else 0
    tablesample: sql.Composable = sql.SQL("")
    if estimated > size * 2:
        percent = min(100.0, 100.0 * size * 1.5 / estimated)
        tablesample = sql.SQL("TABLESAMPLE SYSTEM ({}) REPEATABLE (7)").format(sql.Literal(percent))

    rows = conn.execute(
        sql.SQL(
            "SELECT {id}::text, {t}::text FROM {table} {ts} WHERE {nonempty} "
            "ORDER BY md5(ctid::text) LIMIT {n}"
        ).format(
            id=id_expr,
            t=text_id,
            table=table_id,
            ts=tablesample,
            nonempty=nonempty,
            n=sql.Literal(size),
        )
    ).fetchall()
    docs: dict[str, str] = {r[0]: r[1] for r in rows}

    wanted = [i for i in include_ids if i not in docs]
    if wanted:
        extra = conn.execute(
            sql.SQL("SELECT {id}::text, {t}::text FROM {table} WHERE {id}::text = ANY(%s)").format(
                id=id_expr, t=text_id, table=table_id
            ),
            (wanted,),
        ).fetchall()
        for doc_id, body in extra:
            docs[doc_id] = body or ""
    return list(docs.items()), f"{col.schema}.{col.table}"
