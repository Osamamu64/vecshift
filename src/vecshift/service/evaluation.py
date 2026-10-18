"""Comparing old and new vectors on your data: search quality and latency. Read-only."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.migrate.engine import SETTLED_ROWS, SETTLED_SHARE
from vecshift.service.consent import Consent, Sending, ask
from vecshift.service.errors import ServiceError, first_line
from vecshift.service.jobs import load, pg_settings, read_job
from vecshift.service.plan import destination

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

    from vecshift.connectors.pgvector.search import PgSearcher
    from vecshift.embeddings import ModelSpec
    from vecshift.eval import EvalQuery, EvalReport, SideReport
    from vecshift.jobs import JobSpec

SETUP_TIMEOUT_MS = 10 * 60_000
GROUP_COLUMNS = ("document_id", "doc_id", "parent_id", "source_id", "file_id", "source", "url")
GROUP_KEYS = (*GROUP_COLUMNS, "file_name", "filename", "path", "title")
OTHER_LANGUAGE = {"arabic": "English", "latin": "Arabic"}


@dataclass(frozen=True, slots=True)
class EvalOptions:
    num_queries: int = 200
    queries: Path | None = None
    """JSONL of labeled queries."""
    generate_queries: str | None = None
    """A chat model that writes realistic queries, such as ``openai/gpt-4o-mini``."""
    cross_language: bool = False
    old_model: str | None = None
    """The model that made the old vectors, when the job doesn't say."""
    group_by: str | None = None
    neighbours: int = 100
    tolerance: float = 0.02
    min_slice: int = 30
    max_p95_ms: float | None = None
    max_slowdown: float | None = None
    dsn_env: str | None = None
    """Read the connection string from this variable instead, such as a replica's."""
    seed: int = 7


def run_eval(
    job_file: Path,
    options: EvalOptions | None = None,
    *,
    consent: Consent | None = None,
    on_step: Callable[[str], None] | None = None,
) -> EvalReport:
    """Search old and new vectors with the same queries and compare them.

    Query text goes to the models only after ``consent`` agrees, if it leaves the machine.
    ``on_step`` hears each step as it starts, such as "Embedding queries".
    """
    from vecshift.embeddings import parse_spec
    from vecshift.eval import Gates, SideReport

    o = options or EvalOptions()
    generator = parse_spec(o.generate_queries) if o.generate_queries else None
    if o.cross_language and generator is None:
        raise ServiceError("--cross-language needs --generate-queries.", code="options.invalid")
    if read_job(job_file).target.type == "qdrant":
        setup = _setup_qdrant(job_file, o.group_by, o.dsn_env)
    else:
        setup = _setup_pg(job_file, o.dsn_env, o.group_by)
    job, spec, searcher, text = setup.job, setup.spec, setup.searcher, setup.text
    old_raw = o.old_model or job.source.model
    old_spec = parse_spec(old_raw) if old_raw else None
    old = SideReport("old", setup.old_name, old_spec.name if old_spec else None)
    new = SideReport("new", setup.new_name, spec.name)
    try:
        rows = searcher.sample(text, max(o.num_queries * 3, o.neighbours))
        if not rows:
            raise ServiceError(
                "No rows have both an old and a new vector to compare.", code="eval.no_rows"
            )
        sends = [(spec.name, destination(spec))]
        if old_spec:
            sends.append((f"{old_spec.name} (old)", destination(old_spec)))
        if generator:
            sends.append((f"{generator.name} (writes queries)", destination(generator)))
        ask(consent, Sending("eval", tuple(sends), o.num_queries, "queries"))
        built, source, notes = _queries(
            searcher,
            text,
            rows,
            count=o.num_queries,
            seed=o.seed,
            file=o.queries,
            generator=generator,
            cross_language=o.cross_language,
        )
        report = asyncio.run(
            _evaluate(
                searcher,
                spec,
                old_spec,
                built,
                [k for k, _ in rows[: o.neighbours]],
                old=old,
                new=new,
                dims=setup.dims,
                source=source,
                partial=setup.partial,
                gates=Gates(o.tolerance, o.min_slice, o.max_p95_ms, o.max_slowdown),
                notes=notes,
                on_step=on_step or (lambda s: None),
            )
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    finally:
        setup.close()

    if setup.group_name:
        report.notes.append(f"Rows of the same document are matched by {setup.group_name}.")
    if setup.pending:
        report.notes.append(f"{setup.pending:,} {setup.unit} don't have a new vector yet.")
    return report


@dataclass(slots=True)
class _Columns:
    old: str
    new: str
    stage: str
    """``applied`` (before cutover) or ``cut_over``."""


def _columns(conn: psycopg.Connection, relid: int, live: str, target: str) -> _Columns:
    from vecshift.connectors.pgvector.inspect import _columns as table_columns

    names = table_columns(conn, relid)
    previous = f"{live}_old"[:63]
    if target in names:
        return _Columns(live, target, "applied")
    if previous in names:
        return _Columns(previous, live, "cut_over")
    raise ServiceError(
        f"There are no new vectors to evaluate yet: {target} doesn't exist.",
        "Run apply.",
        "eval.not_applied",
    )


def _metric(conn: psycopg.Connection, relid: int, column: str, configured: str | None) -> str:
    if configured:
        return configured
    rows = conn.execute(
        """
        SELECT opc.opcname FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
        JOIN pg_opclass opc ON opc.oid = i.indclass[0]
        WHERE i.indrelid = %s AND a.attname = %s
        """,
        (relid, column),
    ).fetchall()
    for (name,) in rows:
        for metric, suffix in (("cosine", "cosine_ops"), ("inner_product", "ip_ops")):
            if str(name).endswith(suffix):
                return metric
        if str(name).endswith("l2_ops"):
            return "l2"
    return "cosine"


def _group(
    conn: psycopg.Connection, relid: int, table: sql.Composable, option: str | None
) -> tuple[sql.Composable | None, str | None]:
    """How to tell which rows belong to the same document, and its name for the report."""
    from psycopg import sql

    from vecshift.connectors.pgvector.inspect import _columns as table_columns

    names = table_columns(conn, relid)
    if option:
        column, _, key = option.partition(".")
        if column not in names:
            raise ServiceError(f"--group-by: there's no column {column!r}.", code="options.invalid")
        if key:
            if names[column] not in {"jsonb", "json"}:
                raise ServiceError(
                    f"--group-by: {column} isn't a JSON column.", code="options.invalid"
                )
            return sql.SQL("{}->>{}").format(sql.Identifier(column), sql.Literal(key)), option
        return sql.Identifier(column), option
    for column in GROUP_COLUMNS:
        if column in names:
            return sql.Identifier(column), column
    for column, kind in names.items():
        if kind != "jsonb":
            continue
        rows = conn.execute(
            sql.SQL(
                "SELECT DISTINCT jsonb_object_keys({c}) FROM (SELECT {c} FROM {t} "
                "WHERE jsonb_typeof({c}) = 'object' LIMIT 200) s"
            ).format(c=sql.Identifier(column), t=table)
        ).fetchall()
        keys = {str(r[0]) for r in rows}
        for key in GROUP_KEYS:
            if key in keys:
                return (
                    sql.SQL("{}->>{}").format(sql.Identifier(column), sql.Literal(key)),
                    f"{column}.{key}",
                )
    return None, None


async def _working(spec: ModelSpec, dims: int | None) -> tuple[Any | None, str | None]:
    """An embedder for ``spec`` if it answers and matches the column, else why not."""
    from vecshift.embeddings import EmbeddingError, create_embedder

    embedder = create_embedder(spec)
    try:
        vector = (await embedder.embed(["a short test query"], "query"))[0]
    except EmbeddingError as exc:
        await embedder.aclose()
        return None, f"{spec.name} couldn't embed a test query: {exc}"
    if dims and len(vector) != dims:
        await embedder.aclose()
        return None, (
            f"{spec.name} returns {len(vector)} dimensions, but the column holds {dims}: "
            "is that the model that made these vectors?"
        )
    return embedder, None


def _queries(
    searcher: PgSearcher,
    text: str,
    rows: list[tuple[str, str]],
    *,
    count: int,
    seed: int,
    file: Path | None,
    generator: ModelSpec | None,
    cross_language: bool,
) -> tuple[list[EvalQuery], str, list[str]]:
    from vecshift.bench.corpus import CorpusError, Document, load_queries
    from vecshift.bench.generate import GenerationError, QueryGenerator
    from vecshift.eval import EvalQuery, proxy_queries, script

    if file is not None:
        try:
            labeled = load_queries(file)[:count]
        except CorpusError as exc:
            raise ServiceError(str(exc), code="queries.invalid") from exc
        # Labels name rows by their own key; eval may name them differently (point IDs).
        translate = getattr(searcher, "translate", None)
        given = sorted({k for q in labeled for k in q.relevant})
        names = translate(given) if translate else {k: k for k in given}
        relevant = [frozenset(names.get(k, k) for k in q.relevant) for q in labeled]
        keys = sorted({k for r in relevant for k in r})
        texts = searcher.texts(text, keys)
        queries = [
            EvalQuery(
                q.text,
                r,
                script(q.text),
                script(" ".join(texts.get(k, "") for k in sorted(r))),
            )
            for q, r in zip(labeled, relevant, strict=True)
        ]
        missing = [k for k in keys if k not in texts]
        notes = [f"{len(missing)} labeled answers aren't in the table."] if missing else []
        return queries, "labeled", notes
    if generator is not None:
        docs = [Document(key, body) for key, body in rows]
        scripts = {key: script(body) for key, body in rows}

        def language(doc: Document) -> str | None:
            return OTHER_LANGUAGE.get(scripts[doc.id]) if cross_language else None

        async def make() -> list[Any]:
            gen = QueryGenerator(generator)
            try:
                return await gen.generate(docs, count, seed, language)
            finally:
                await gen.aclose()

        try:
            made = asyncio.run(make())
        except GenerationError as exc:
            raise ServiceError(str(exc), code="model.failed") from exc
        queries = [
            EvalQuery(q.text, q.relevant, script(q.text), scripts[next(iter(q.relevant))])
            for q in made
        ]
        return queries, "cross-language" if cross_language else "generated", []
    queries, notes = proxy_queries(rows, count, seed)
    return queries, "proxy", notes


@dataclass(slots=True)
class _Setup:
    job: JobSpec
    spec: ModelSpec
    searcher: Any
    text: str
    old_name: str
    new_name: str
    dims: dict[str, int | None]
    partial: bool
    pending: int
    group_name: str | None
    close: Any
    unit: str = "rows"


def _setup_pg(job_file: Path, dsn_env: str | None, group_by: str | None) -> _Setup:
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.inspect import TEXT_COLUMNS, TEXT_TYPES, _pick
    from vecshift.connectors.pgvector.inspect import _columns as table_columns
    from vecshift.connectors.pgvector.search import PgSearcher
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column

    job, settings, spec = load(job_file, dsn_env, command="vecshift eval")
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    conn.rollback()
    try:
        # One read-only transaction for the setup queries, under a generous time limit.
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(SETUP_TIMEOUT_MS),))
        live = resolve_source_column(
            conn, job.pg_source.table, job.pg_source.vector_column, job.pg_target.column
        )
        target = inspect_target(conn, job.pg_source.table, live, job.pg_target.column)
        relid = target.source.relid
        names = table_columns(conn, relid)
        text = job.pg_source.text_column or _pick(names, TEXT_COLUMNS, TEXT_TYPES)
        if len(target.primary_key) != 1 or text is None:
            raise ServiceError(
                "eval needs a single-column primary key and a text column.",
                code="source.unsupported",
            )
        cols = _columns(conn, relid, live, job.pg_target.column)
        metric = _metric(
            conn, relid, cols.old, job.pg_target.metric.value if job.pg_target.metric else None
        )
        from psycopg import sql

        table = sql.Identifier(target.source.schema, target.source.table)
        group, group_name = _group(conn, relid, table, group_by)
        dims = {
            side: conn.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s AND attname = %s",
                (relid, name),
            ).fetchone()
            for side, name in (("old", cols.old), ("new", cols.new))
        }
        missing = conn.execute(
            sql.SQL(
                "SELECT count(*) FILTER (WHERE {new} IS NULL), count(*) FILTER "
                "(WHERE {new} IS NOT NULL) FROM {t} WHERE {txt} IS NOT NULL "
                "AND btrim({txt}::text) <> ''"
            ).format(new=sql.Identifier(cols.new), t=table, txt=sql.Identifier(text))
        ).fetchone()
        conn.rollback()
    except pgvector.TargetSelectionError as exc:
        conn.close()
        raise ServiceError(str(exc), code="source.not_found") from exc
    except ServiceError:
        conn.close()
        raise

    pending, done = (int(missing[0]), int(missing[1])) if missing else (0, 0)
    # A busy table always has a few rows waiting for a vector; that's not a partial migration.
    partial = pending > max(SETTLED_ROWS, SETTLED_SHARE * done)
    if not done:
        conn.close()
        raise ServiceError(
            "The new column has no vectors yet.",
            "Run apply (or apply --until 10).",
            "eval.not_applied",
        )
    searcher = PgSearcher(
        conn,
        schema=target.source.schema,
        table=target.source.table,
        pk=target.primary_key[0],
        columns={"old": cols.old, "new": cols.new},
        types={"old": names[cols.old], "new": names[cols.new]},
        extension_schema=target.extension_schema,
        metric=metric,
        partial=partial,
        group=group,
    )
    return _Setup(
        job=job,
        spec=spec,
        searcher=searcher,
        text=text,
        old_name=cols.old,
        new_name=cols.new,
        dims={k: (int(v[0]) if v and v[0] > 0 else None) for k, v in dims.items()},
        partial=partial,
        pending=pending,
        group_name=group_name,
        close=conn.close,
    )


def _qdrant_group(points: list[dict[str, Any]], option: str | None) -> str | None:
    """The payload field telling which points belong to one document."""
    from vecshift.connectors.qdrant.inspect import _get

    if option:
        return option
    for key in GROUP_KEYS:
        for prefix in ("", "metadata.", "meta."):
            if any(_get(p.get("payload") or {}, prefix + key) is not None for p in points):
                return prefix + key
    return None


def _store_error(exc: Exception) -> Exception:
    """A Qdrant or database error as a clean message; anything else unchanged."""
    import psycopg

    from vecshift.connectors.qdrant import QdrantError

    if isinstance(exc, QdrantError):
        return ServiceError(str(exc), exc.hint or "eval only reads; run it again.", "qdrant.error")
    if isinstance(exc, psycopg.Error):
        return ServiceError(
            f"The database reported an error: {first_line(exc)}",
            "eval only reads; run it again.",
            "database.error",
        )
    return exc


def _setup_qdrant(job_file: Path, group_by: str | None, dsn_env: str | None = None) -> _Setup:
    """Old and new vectors of a migration into Qdrant, before or after cutover."""
    from vecshift.connectors.pgvector.reader import point_id
    from vecshift.connectors.qdrant.inspect import _fields, _points
    from vecshift.connectors.qdrant.search import PgSide, QdrantRows, QdrantSide, SplitSearcher
    from vecshift.connectors.qdrant.writer import QdrantSource, QdrantWriter
    from vecshift.jobs.spec import QdrantSource as QdrantSourceSpec
    from vecshift.service.qdrant import locate

    found = locate(job_file, dsn_env)
    closers: list[Any] = [found.close]

    def close() -> None:
        for fn in reversed(closers):
            fn()

    try:
        client, lay = found.client, found.writer.layout
        if not client.exists(found.new):
            raise ServiceError(
                f"There are no new vectors to evaluate yet: {found.new} doesn't exist.",
                "Run apply.",
                "eval.not_applied",
            )
        counter = QdrantWriter(client, found.writer.source, lay, read_only=True)
        total = counter.total_count()
        pending = counter.pending_count(tuple(found.state.failed))
        if total - pending <= 0:
            raise ServiceError(
                "The new collection has no vectors yet.", "Run apply.", "eval.not_applied"
            )
        if pending > max(SETTLED_ROWS, SETTLED_SHARE * (total - pending)):
            raise ServiceError(
                f"{pending:,} points don't have a new vector yet.",
                "eval for Qdrant compares complete collections: run apply to finish first.",
                "eval.not_applied",
            )
        new_points, _ = _points(client, found.new, False, 200, 201)
        group = _qdrant_group(new_points, group_by)
        new_dims = lay.dims or None
        source = found.writer.source
        if isinstance(source, QdrantSource):
            reader = source.client
            info = reader.collection(source.collection)
            wanted = (
                found.job.source.vector if isinstance(found.job.source, QdrantSourceSpec) else None
            )
            fields = _fields(source.collection, info)
            field = next((f for f in fields if f.name == wanted), fields[0])
            old_side: Any = QdrantSide(
                reader,
                source.collection,
                field.name,
                field.distance or "Cosine",
                group_field=group,
            )
            new_side = QdrantSide(client, found.new, lay.vector, lay.distance, group_field=group)
            rows: Any = QdrantRows(new_side, found.text_field)
            old_name, old_dims, unit = source.collection, field.dimensions, "points"
        else:
            from vecshift.connectors.pgvector.reader import PgSource
            from vecshift.connectors.pgvector.search import PgSearcher

            if not isinstance(source, PgSource):
                raise ServiceError("Unknown source for this job.", code="source.unsupported")
            pk = source.pk
            settings = pg_settings(dsn_env or found.job.pg_source.dsn_env)
            prepared = _setup_pg_source(found, settings)
            conn, target, live, kind, metric = prepared
            closers.append(conn.close)
            pg = PgSearcher(
                conn,
                schema=target.source.schema,
                table=target.source.table,
                pk=pk,
                columns={"old": live, "new": live},
                types={"old": kind, "new": kind},
                extension_schema=target.extension_schema,
                metric=metric,
                key_fn=lambda value: str(point_id(value)),
            )
            old_side = PgSide(pg, "old")
            new_side = QdrantSide(
                client,
                found.new,
                lay.vector,
                lay.distance,
                group_field=group,
            )
            rows = pg
            old_name, old_dims, unit = (
                f"{target.source.table}.{live}",
                target.source.dimensions,
                "rows",
            )
        searcher = SplitSearcher(rows, {"old": old_side, "new": new_side})
    except Exception as exc:
        close()
        raise _store_error(exc) from exc
    except BaseException:
        close()
        raise
    return _Setup(
        job=found.job,
        spec=found.spec,
        searcher=searcher,
        text=found.text_field,
        old_name=old_name,
        new_name=found.new,
        dims={"old": old_dims, "new": new_dims},
        partial=False,
        pending=pending,
        group_name=group,
        close=close,
        unit=unit,
    )


def _setup_pg_source(found: Any, settings: Any) -> tuple[Any, Any, str, str, str]:
    """A read-only connection to a pgvector source, and its live vector column."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.inspect import _columns as table_columns
    from vecshift.connectors.pgvector.inspect import select_column
    from vecshift.connectors.pgvector.target import inspect_target

    source = found.job.pg_source
    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise ServiceError(f"Couldn't connect: {exc}", exc.hint, "connect.failed") from exc
    try:
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(SETUP_TIMEOUT_MS),))
        columns = pgvector.find_vector_columns(conn)
        chosen = select_column(columns, source.table, source.vector_column)
        target = inspect_target(conn, source.table, chosen.column, "")
        kind = table_columns(conn, target.source.relid)[chosen.column]
        metric = _metric(conn, target.source.relid, chosen.column, None)
        conn.rollback()
    except pgvector.TargetSelectionError as exc:
        conn.close()
        raise ServiceError(str(exc), code="source.not_found") from exc
    except BaseException:
        conn.close()
        raise
    return conn, target, chosen.column, kind, metric


async def _evaluate(
    searcher: PgSearcher,
    spec: ModelSpec,
    old_spec: ModelSpec | None,
    queries: list[EvalQuery],
    neighbour_keys: list[str],
    *,
    old: SideReport,
    new: SideReport,
    dims: dict[str, int | None],
    source: str,
    partial: bool,
    gates: Any,
    notes: list[str],
    on_step: Any,
) -> EvalReport:
    from vecshift.eval import run

    new_embedder, problem = await _working(spec, dims["new"])
    if new_embedder is None:
        raise ServiceError(problem or "The new model isn't working.", code="model.failed")
    old_embedder = None
    if old_spec is None:
        old.note = (
            "No old model given (source.model or --old-model), so the old side is judged "
            "from its stored vectors only."
        )
    else:
        old_embedder, old.note = await _working(old_spec, dims["old"])
    try:
        return await run(
            searcher,
            {"old": old_embedder, "new": new_embedder},
            queries,
            neighbour_keys,
            old=old,
            new=new,
            query_source=source,
            partial=partial,
            gates=gates,
            notes=notes,
            on_step=on_step,
        )
    finally:
        await new_embedder.aclose()
        if old_embedder is not None:
            await old_embedder.aclose()
