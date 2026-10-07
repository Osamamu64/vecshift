"""Read-only inspection of a pgvector column for ``vecshift doctor``.

Only aggregates leave the database: per-row dimensions, norms, and hashes are computed
in SQL, so inspecting a table never downloads its vectors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from vecshift.doctor.profile import AnnIndex, IndexProfile, Sample

VECTOR_TYPES = ("vector", "halfvec")

# pgvector's HNSW and IVFFlat limits per type.
MAX_INDEXABLE_DIMENSIONS = {"vector": 2000, "halfvec": 4000}

# Schemas that never hold user vector tables, including Supabase's internal ones.
SYSTEM_SCHEMAS = (
    "pg_catalog",
    "information_schema",
    "pg_toast",
    "auth",
    "storage",
    "realtime",
    "_realtime",
    "supabase_functions",
    "supabase_migrations",
    "extensions",
    "graphql",
    "graphql_public",
    "pgsodium",
    "pgsodium_masks",
    "vault",
    "net",
    "cron",
    "pgbouncer",
)

# Common names, most likely first. Covers Supabase and LangChain (content, document),
# LlamaIndex (text), and others.
TEXT_COLUMNS = ("content", "text", "page_content", "document", "chunk", "chunk_text", "body")
MODEL_COLUMNS = ("model_tag", "embedding_model", "model", "model_name")
MODEL_METADATA_KEYS = ("model_tag", "embedding_model", "model", "model_name")
METADATA_COLUMNS = ("metadata", "cmetadata", "meta", "metadata_")
UPDATED_AT_COLUMNS = (
    "updated_at",
    "modified_at",
    "last_modified",
    "last_modified_at",
    "changed_at",
    "updated",
    "modified",
)

TEXT_TYPES = ("text", "varchar", "bpchar")
JSON_TYPES = ("jsonb", "json")
TIMESTAMP_TYPES = ("timestamptz", "timestamp")

_METRICS = (
    ("cosine", "cosine"),
    ("_ip_", "inner_product"),
    ("l2", "l2"),
    ("l1", "l1"),
    ("hamming", "hamming"),
    ("jaccard", "jaccard"),
)


class TargetSelectionError(Exception):
    """The vector column to inspect couldn't be determined."""

    def __init__(self, message: str, candidates: list[VectorColumn] | None = None) -> None:
        super().__init__(message)
        self.candidates = candidates or []


@dataclass(frozen=True, slots=True)
class VectorColumn:
    schema: str
    table: str
    column: str
    type: str
    dimensions: int | None
    relid: int
    attnum: int

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.table}.{self.column}"


def find_vector_columns(conn: psycopg.Connection) -> list[VectorColumn]:
    """Every vector or halfvec column in a user table, view, or materialized view."""
    rows = conn.execute(
        """
        SELECT n.nspname, c.relname, a.attname, t.typname, a.atttypmod, c.oid, a.attnum
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_type t ON t.oid = a.atttypid
        WHERE t.typname = ANY(%s)
          AND c.relkind IN ('r', 'p', 'm')
          AND NOT c.relispartition
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND n.nspname <> ALL(%s)
          AND n.nspname NOT LIKE 'pg\\_temp\\_%%'
        ORDER BY n.nspname, c.relname, a.attnum
        """,
        (list(VECTOR_TYPES), list(SYSTEM_SCHEMAS)),
    ).fetchall()
    return [
        VectorColumn(
            schema=r[0],
            table=r[1],
            column=r[2],
            type=r[3],
            dimensions=r[4] if r[4] > 0 else None,
            relid=r[5],
            attnum=r[6],
        )
        for r in rows
    ]


def select_column(
    columns: list[VectorColumn], table: str | None, column: str | None
) -> VectorColumn:
    """Pick the column to inspect from ``--table`` and ``--column``, if given."""
    matches = columns
    if table:
        schema, _, name = table.rpartition(".")
        matches = [c for c in matches if c.table == name and (not schema or c.schema == schema)]
        if not matches:
            raise TargetSelectionError(
                f"No vector columns found in table {table!r}.", candidates=columns
            )
    if column:
        matches = [c for c in matches if c.column == column]
        if not matches:
            raise TargetSelectionError(f"No vector column named {column!r}.", candidates=columns)

    if not matches:
        raise TargetSelectionError(
            "No pgvector columns found. Is the vector extension installed, and can this "
            "role see the tables?"
        )
    if len(matches) > 1:
        raise TargetSelectionError(
            "Found more than one vector column. Choose one with --table (and --column).",
            candidates=matches,
        )
    return matches[0]


def _extension_schema(conn: psycopg.Connection) -> str:
    # Supabase installs pgvector in "extensions", which isn't always on the search path,
    # so every pgvector function call is schema-qualified.
    row = conn.execute(
        "SELECT n.nspname FROM pg_extension e "
        "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
    ).fetchone()
    if row is None:
        raise TargetSelectionError("The pgvector extension isn't installed in this database.")
    return str(row[0])


def _columns(conn: psycopg.Connection, relid: int) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT a.attname, t.typname
        FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
        WHERE a.attrelid = %s AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (relid,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _pick(columns: dict[str, str], names: tuple[str, ...], types: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in columns}
    for candidate in names:
        actual = lowered.get(candidate)
        if actual is not None and columns[actual] in types:
            return actual
    return None


def _ann_indexes(conn: psycopg.Connection, col: VectorColumn) -> list[AnnIndex]:
    rows = conn.execute(
        """
        SELECT ic.relname, am.amname, opc.opcname
        FROM pg_index i
        JOIN pg_class ic ON ic.oid = i.indexrelid
        JOIN pg_am am ON am.oid = ic.relam
        CROSS JOIN LATERAL unnest(i.indkey::int2[], i.indclass::oid[]) AS k(attnum, opclass)
        JOIN pg_opclass opc ON opc.oid = k.opclass
        WHERE i.indrelid = %s
          AND am.amname IN ('hnsw', 'ivfflat')
          AND (
            k.attnum = %s
            -- Expression indexes, such as an HNSW index over embedding::halfvec(3072).
            OR (k.attnum = 0 AND pg_get_indexdef(i.indexrelid) LIKE '%%' || quote_ident(%s) || '%%')
          )
        ORDER BY ic.relname
        """,
        (col.relid, col.attnum, col.column),
    ).fetchall()
    indexes = []
    for name, method, opclass in rows:
        metric = next((m for key, m in _METRICS if key in opclass), None)
        indexes.append(AnnIndex(name=name, method=method, metric=metric))
    return indexes


def _rows_hidden(conn: psycopg.Connection, relid: int) -> bool:
    row = conn.execute(
        """
        SELECT c.relrowsecurity, c.relforcerowsecurity,
               pg_has_role(current_user, c.relowner, 'USAGE'),
               r.rolbypassrls OR r.rolsuper
        FROM pg_class c, pg_roles r
        WHERE c.oid = %s AND r.rolname = current_user
        """,
        (relid,),
    ).fetchone()
    if row is None:
        return False
    enabled, forced, owner, bypass = row
    if not enabled or bypass or (owner and not forced):
        return False
    # A permissive "using (true)" read policy for this role shows every row, unless a
    # restrictive policy narrows it again.
    sees_all = conn.execute(
        """
        WITH applicable AS (
            SELECT p.polpermissive, pg_get_expr(p.polqual, p.polrelid) AS qual
            FROM pg_policy p
            WHERE p.polrelid = %s
              AND p.polcmd IN ('r', '*')
              AND (
                0 = ANY(p.polroles)
                OR EXISTS (
                    SELECT 1 FROM unnest(p.polroles) AS r(oid)
                    WHERE pg_has_role(current_user, r.oid, 'MEMBER')
                )
              )
        )
        SELECT EXISTS (SELECT 1 FROM applicable WHERE polpermissive AND qual = 'true')
           AND NOT EXISTS (SELECT 1 FROM applicable WHERE NOT polpermissive)
        """,
        (relid,),
    ).fetchone()
    return not (sees_all and sees_all[0])


def _sample(
    conn: psycopg.Connection,
    col: VectorColumn,
    ext: str,
    estimated_rows: int | None,
    size: int,
    text_field: str | None,
    model_expr: sql.Composable,
) -> Sample:
    vec = sql.Identifier(col.column)
    fn = sql.Identifier(ext, "vector_dims")
    norm = sql.Identifier(ext, "vector_norm" if col.type == "vector" else "l2_norm")
    has_text: sql.Composable
    text_hash: sql.Composable
    if text_field:
        text = sql.Identifier(text_field)
        has_text = sql.SQL("({t} IS NOT NULL AND btrim({t}::text) <> '')").format(t=text)
        text_hash = sql.SQL("md5({t}::text)").format(t=text)
    else:
        has_text = sql.SQL("NULL::boolean")
        text_hash = sql.SQL("NULL::text")

    def query(tablesample: sql.Composable) -> list[tuple[Any, ...]]:
        q = sql.SQL(
            "SELECT {vec} IS NULL, {fn}({vec}), {norm}({vec}), md5({vec}::text), "
            "{has_text}, {text_hash}, {model} "
            "FROM {table} {tablesample} LIMIT {limit}"
        ).format(
            vec=vec,
            fn=fn,
            norm=norm,
            has_text=has_text,
            text_hash=text_hash,
            model=model_expr,
            table=sql.Identifier(col.schema, col.table),
            tablesample=tablesample,
            limit=sql.Literal(size),
        )
        return conn.execute(q).fetchall()

    # For big tables, sample whole pages instead of reading the start of the table.
    # SYSTEM sampling is cheap because it skips unsampled pages entirely.
    method = "first rows"
    rows: list[tuple[Any, ...]] = []
    if estimated_rows and estimated_rows > size * 2:
        percent = min(100.0, 100.0 * size * 2 / estimated_rows)
        rows = query(sql.SQL("TABLESAMPLE SYSTEM ({}) REPEATABLE (7)").format(sql.Literal(percent)))
        method = "table sample"
        if len(rows) < size // 2:
            rows, method = [], "first rows"
    if not rows:
        rows = query(sql.SQL(""))
        if len(rows) < size:
            method = "full table"

    sample = Sample(rows=len(rows), method=method)
    sample.texts_present = 0 if text_field else None
    vector_hashes: Counter[str] = Counter()
    text_hashes: Counter[str] = Counter()
    model_sources: Counter[str] = Counter()
    for is_null, dims, norm_value, vhash, text_ok, thash, model in rows:
        if is_null:
            sample.null_vectors += 1
        else:
            sample.dimensions[int(dims)] += 1
            sample.norms.append(float(norm_value))
            vector_hashes[vhash] += 1
        if text_ok:
            sample.texts_present = (sample.texts_present or 0) + 1
            text_hashes[thash] += 1
        if model is not None and not is_null:
            source, _, value = str(model).partition("\t")
            sample.models[value] += 1
            model_sources[source] += 1
    sample.duplicate_vectors = sum(n - 1 for n in vector_hashes.values())
    sample.duplicate_texts = sum(n - 1 for n in text_hashes.values())
    sample.model_source = model_sources.most_common(1)[0][0] if model_sources else None
    return sample


def _model_source(columns: dict[str, str]) -> tuple[str | None, sql.Composable]:
    """Where each row records its model, as a label and an expression yielding
    ``label<TAB>model`` (so rows can say which metadata key they used)."""
    column = _pick(columns, MODEL_COLUMNS, TEXT_TYPES)
    if column:
        return column, sql.SQL("{} || E'\\t' || {}::text").format(
            sql.Literal(column), sql.Identifier(column)
        )
    metadata = _pick(columns, METADATA_COLUMNS, JSON_TYPES)
    if metadata:
        meta = sql.Identifier(metadata)
        keys = sql.SQL(", ").join(
            sql.SQL("{} || E'\\t' || ({} ->> {})").format(
                sql.Literal(f"{metadata}->>'{k}'"), meta, sql.Literal(k)
            )
            for k in MODEL_METADATA_KEYS
        )
        return metadata, sql.SQL("COALESCE({})").format(keys)
    return None, sql.SQL("NULL::text")


def inspect(
    conn: psycopg.Connection,
    table: str | None = None,
    column: str | None = None,
    text_column: str | None = None,
    sample_size: int = 2000,
) -> IndexProfile:
    """Profile one vector column. Read-only; run it on a read-only connection."""
    ext = _extension_schema(conn)
    col = select_column(find_vector_columns(conn), table, column)
    columns = _columns(conn, col.relid)

    if text_column is not None:
        if text_column not in columns:
            raise TargetSelectionError(
                f"Column {text_column!r} doesn't exist in {col.schema}.{col.table}."
            )
        text_field: str | None = text_column
    else:
        text_field = _pick(columns, TEXT_COLUMNS, TEXT_TYPES)
    model_field, model_expr = _model_source(columns)

    reltuples = conn.execute(
        "SELECT reltuples::bigint FROM pg_class WHERE oid = %s", (col.relid,)
    ).fetchone()
    estimated = int(reltuples[0]) if reltuples and reltuples[0] >= 0 else None

    sample = _sample(conn, col, ext, estimated, sample_size, text_field, model_expr)
    if sample.method == "full table":
        estimated = sample.rows

    pk = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_index WHERE indrelid = %s AND indisprimary)",
        (col.relid,),
    ).fetchone()
    wal = conn.execute("SELECT current_setting('wal_level')").fetchone()

    return IndexProfile(
        store="pgvector",
        target=col.qualified,
        vector_type=col.type,
        declared_dimensions=col.dimensions,
        estimated_rows=estimated,
        sample=sample,
        text_field=text_field,
        model_field=sample.model_source or model_field,
        updated_at_field=_pick(columns, UPDATED_AT_COLUMNS, TIMESTAMP_TYPES),
        has_primary_key=bool(pk and pk[0]),
        logical_replication=bool(wal and wal[0] == "logical"),
        ann_indexes=_ann_indexes(conn, col),
        max_indexable_dimensions=MAX_INDEXABLE_DIMENSIONS.get(col.type),
        rows_hidden_by_access_rules=_rows_hidden(conn, col.relid),
    )
