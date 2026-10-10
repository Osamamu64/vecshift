"""The ``vecshift eval`` command: is the new model better on your data, and how fast is it?"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import DEFAULT_JOB, _fail, load

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

    from vecshift.connectors.pgvector.search import PgSearcher
    from vecshift.embeddings import ModelSpec
    from vecshift.eval import EvalQuery, EvalReport, SideReport

EXIT_INCONCLUSIVE = 3
SETUP_TIMEOUT_MS = 10 * 60_000
GROUP_COLUMNS = ("document_id", "doc_id", "parent_id", "source_id", "file_id", "source", "url")
GROUP_KEYS = (*GROUP_COLUMNS, "file_name", "filename", "path", "title")
OTHER_LANGUAGE = {"arabic": "English", "latin": "Arabic"}


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
    raise _fail(f"There are no new vectors to evaluate yet: {target} doesn't exist.", "Run apply.")


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
            raise _fail(f"--group-by: there's no column {column!r}.")
        if key:
            if names[column] not in {"jsonb", "json"}:
                raise _fail(f"--group-by: {column} isn't a JSON column.")
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
            raise _fail(str(exc)) from exc
        keys = sorted({k for q in labeled for k in q.relevant})
        texts = searcher.texts(text, keys)
        queries = [
            EvalQuery(
                q.text,
                q.relevant,
                script(q.text),
                script(" ".join(texts.get(k, "") for k in sorted(q.relevant))),
            )
            for q in labeled
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
            raise _fail(str(exc)) from exc
        queries = [
            EvalQuery(q.text, q.relevant, script(q.text), scripts[next(iter(q.relevant))])
            for q in made
        ]
        return queries, "cross-language" if cross_language else "generated", []
    queries, notes = proxy_queries(rows, count, seed)
    return queries, "proxy", notes


def _confirm(sends: list[tuple[str, str]], count: int, yes: bool) -> None:
    remote = [(name, url) for name, url in sends if url]
    if not remote:
        return
    out = sys.stderr
    print("eval sends text to:", file=out)
    for name, url in remote:
        print(f"  {name:<40} {url}", file=out)
    print(
        f"About {count:,} short queries (sentences from your rows, or generated from them).",
        file=out,
    )
    if yes:
        return
    if not sys.stdin.isatty():
        raise _fail("Not sending data without confirmation.", "Pass --yes to proceed.")
    if not typer.confirm("Continue?", err=True):
        raise typer.Exit(1)


def eval_(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    num_queries: Annotated[int, typer.Option(min=5, max=10_000, help="Queries to evaluate.")] = 200,
    queries: Annotated[
        Path | None,
        typer.Option(help='JSONL of labeled queries: {"query": ..., "relevant": [ids]} per line.'),
    ] = None,
    generate_queries: Annotated[
        str | None,
        typer.Option(
            metavar="CHAT_MODEL",
            help="Write realistic queries with this chat model, e.g. openai/gpt-4o-mini.",
        ),
    ] = None,
    cross_language: Annotated[
        bool,
        typer.Option(
            help="With --generate-queries: ask in the other language (Arabic for Latin-script "
            "rows, English for Arabic ones)."
        ),
    ] = False,
    old_model: Annotated[
        str | None,
        typer.Option(help="The model that made the old vectors (default: source.model)."),
    ] = None,
    group_by: Annotated[
        str | None,
        typer.Option(
            metavar="COLUMN[.KEY]",
            help="Which rows belong to one document, e.g. document_id or metadata.source.",
        ),
    ] = None,
    neighbours: Annotated[
        int, typer.Option(min=0, max=5_000, help="Rows whose nearest rows are compared.")
    ] = 100,
    tolerance: Annotated[
        float, typer.Option(min=0, max=1, help="How far recall@10 may fall and still pass.")
    ] = 0.02,
    min_slice: Annotated[
        int, typer.Option(min=1, help="Fewest queries for a language pair to decide the verdict.")
    ] = 30,
    max_p95_ms: Annotated[
        float | None, typer.Option(help="Fail if the new p95 (embed + search) is above this.")
    ] = None,
    max_slowdown: Annotated[
        float | None,
        typer.Option(metavar="PERCENT", help="Fail if the new p95 is this much slower than old."),
    ] = None,
    dsn_env: Annotated[
        str | None,
        typer.Option(help="Read the connection string from this variable, e.g. a replica's."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for sampling.")] = 7,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Don't ask before sending data.")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    html: Annotated[
        Path | None,
        typer.Option("--html", dir_okay=False, help="Also write an HTML report to this file."),
    ] = None,
) -> None:
    """Compare the old and new vectors on your data: search quality and latency. Read-only."""
    from vecshift.connectors import pgvector
    from vecshift.connectors.pgvector.inspect import TEXT_COLUMNS, TEXT_TYPES, _pick
    from vecshift.connectors.pgvector.inspect import _columns as table_columns
    from vecshift.connectors.pgvector.search import PgSearcher
    from vecshift.connectors.pgvector.target import inspect_target, resolve_source_column
    from vecshift.embeddings import parse_spec
    from vecshift.eval import Gates, SideReport

    job, settings, spec = load(job_file, dsn_env)
    old_raw = old_model or job.source.model
    old_spec = parse_spec(old_raw) if old_raw else None
    generator = parse_spec(generate_queries) if generate_queries else None
    if cross_language and generator is None:
        raise _fail("--cross-language needs --generate-queries.")

    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    conn.rollback()
    try:
        # One read-only transaction for the setup queries, under a generous time limit.
        conn.execute("SELECT set_config('statement_timeout', %s, true)", (str(SETUP_TIMEOUT_MS),))
        live = resolve_source_column(
            conn, job.source.table, job.source.vector_column, job.target.column
        )
        target = inspect_target(conn, job.source.table, live, job.target.column)
        relid = target.source.relid
        names = table_columns(conn, relid)
        text = job.source.text_column or _pick(names, TEXT_COLUMNS, TEXT_TYPES)
        if len(target.primary_key) != 1 or text is None:
            raise _fail("eval needs a single-column primary key and a text column.")
        cols = _columns(conn, relid, live, job.target.column)
        metric = _metric(
            conn, relid, cols.old, job.target.metric.value if job.target.metric else None
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
        raise _fail(str(exc)) from exc
    except typer.Exit:
        conn.close()
        raise

    pending, done = (int(missing[0]), int(missing[1])) if missing else (0, 0)
    if not done:
        conn.close()
        raise _fail("The new column has no vectors yet.", "Run apply (or apply --until 10).")
    searcher = PgSearcher(
        conn,
        schema=target.source.schema,
        table=target.source.table,
        pk=target.primary_key[0],
        columns={"old": cols.old, "new": cols.new},
        types={"old": names[cols.old], "new": names[cols.new]},
        extension_schema=target.extension_schema,
        metric=metric,
        partial=pending > 0,
        group=group,
    )
    old = SideReport("old", cols.old, old_spec.name if old_spec else None)
    new = SideReport("new", cols.new, spec.name)
    try:
        rows = searcher.sample(text, max(num_queries * 3, neighbours))
        if not rows:
            raise _fail("No rows have both an old and a new vector to compare.")
        sends = [(spec.name, "" if spec.is_local else spec.url or "")]
        if old_spec:
            sends.append(
                (f"{old_spec.name} (old)", "" if old_spec.is_local else old_spec.url or "")
            )
        if generator:
            sends.append(
                (
                    f"{generator.name} (writes queries)",
                    "" if generator.is_local else generator.url or "",
                )
            )
        _confirm(sends, num_queries, yes)
        built, source, notes = _queries(
            searcher,
            text,
            rows,
            count=num_queries,
            seed=seed,
            file=queries,
            generator=generator,
            cross_language=cross_language,
        )
        report = asyncio.run(
            _evaluate(
                searcher,
                spec,
                old_spec,
                built,
                [k for k, _ in rows[:neighbours]],
                old=old,
                new=new,
                dims={k: (int(v[0]) if v and v[0] > 0 else None) for k, v in dims.items()},
                source=source,
                partial=pending > 0,
                gates=Gates(tolerance, min_slice, max_p95_ms, max_slowdown),
                notes=notes,
                on_step=(lambda s: None) if output_json else _step,
            )
        )
    finally:
        conn.close()

    if group_name:
        report.notes.append(f"Rows of the same document are matched by {group_name}.")
    if pending:
        report.notes.append(f"{pending:,} rows don't have a new vector yet.")
    if output_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _render(report, job.name)
    if html is not None:
        from vecshift import __version__
        from vecshift.eval.html import render_html

        html.write_text(render_html(report, job=job.name, version=__version__), encoding="utf-8")
        if not output_json:
            typer.echo(f"\nWrote {html}")
    if report.verdict.status == "no_go":
        raise typer.Exit(1)
    if report.verdict.status == "inconclusive":
        raise typer.Exit(EXIT_INCONCLUSIVE)


def _step(message: str) -> None:
    typer.secho(f"  {message}…", dim=True, err=True)


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
        raise _fail(problem or "The new model isn't working.")
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


# --- Rendering


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return ""
    delta = after - before
    color = typer.colors.GREEN if delta > 0.005 else typer.colors.RED if delta < -0.005 else None
    return typer.style(f"{delta:+.3f}", fg=color)


def _render(report: EvalReport, name: str) -> None:
    old, new = report.old, report.new
    typer.secho(f"vecshift eval · {name}", bold=True)
    typer.echo(f"  Old  {old.column} · {old.model or 'model unknown'}")
    typer.echo(f"  New  {new.column} · {new.model}")
    kind = "partial migration" if report.partial else "complete migration"
    typer.echo(f"  {report.queries:,} {report.query_source} queries · {kind}")
    for side in (old, new):
        if side.note:
            typer.secho(f"  ⚠ {side.note}", fg=typer.colors.YELLOW)

    if new.scores:
        typer.secho("\nSearch quality", bold=True)
        typer.echo(f"  {'':<30}{'old':>8}{'new':>8}  change")
        for key, after in new.scores.items():
            before = old.scores.get(key)
            label = f"{'all queries' if key == 'all' else key} ({after.queries})"
            typer.echo(
                f"  Recall@10 {label:<20}{_num(before.recall_at_10 if before else None):>8}"
                f"{_num(after.recall_at_10):>8}  "
                f"{_change(before.recall_at_10 if before else None, after.recall_at_10)}"
            )
        a_old, a_new = old.scores.get("all"), new.scores.get("all")
        typer.echo(
            f"  {'Recall@1 (all)':<30}{_num(a_old.recall_at_1 if a_old else None):>8}"
            f"{_num(a_new.recall_at_1 if a_new else None):>8}"
        )
        typer.echo(
            f"  {'MRR@10 (all)':<30}{_num(a_old.mrr_at_10 if a_old else None):>8}"
            f"{_num(a_new.mrr_at_10 if a_new else None):>8}"
        )
        if report.result_overlap is not None:
            typer.echo(f"  Top-10 results in common: {report.result_overlap:.0%}")

    if old.same_group_at_10 is not None or report.neighbour_overlap is not None:
        typer.secho("\nNearest rows (from stored vectors)", bold=True)
        if old.same_group_at_10 is not None or new.same_group_at_10 is not None:
            typer.echo(
                f"  {'Same document in top 10':<30}{_num(old.same_group_at_10):>8}"
                f"{_num(new.same_group_at_10):>8}  "
                f"{_change(old.same_group_at_10, new.same_group_at_10)}"
            )
        if report.neighbour_overlap is not None:
            typer.echo(f"  Nearest rows in common: {report.neighbour_overlap:.0%}")

    if old.sweep or new.sweep or old.embed_latency or new.embed_latency:
        typer.secho("\nLatency vs accuracy", bold=True)
        typer.echo(f"  {'Query embedding p50 / p95':<26}{_lat(old)}   {_lat(new)}")
        for side in (old, new):
            if not side.sweep:
                continue
            setting = {"hnsw": "ef_search", "ivfflat": "probes"}.get(side.method or "", "exact")
            typer.echo(f"  {side.side} search ({side.method or 'no index'}):")
            for point in side.sweep:
                mark = " *" if point.current else ""
                lat = point.latency
                p50, p95, p99 = (lat.p50_ms, lat.p95_ms, lat.p99_ms) if lat else (None, None, None)
                value = point.setting if point.setting is not None else "—"
                typer.echo(
                    f"    {setting}={value}{mark:<3} index recall {_num(point.index_recall)}"
                    f"  p50 {_ms(p50)}  p95 {_ms(p95)}  p99 {_ms(p99)}"
                )
        samples = next((p.latency.samples for p in new.sweep if p.latency), 0)
        typer.secho(
            f"  * current setting. {samples} queries per setting; search times include the "
            "network round trip.",
            dim=True,
        )
        typer.echo(
            f"  End-to-end p95 (embed + search): old {_ms(old.end_to_end_p95)} · "
            f"new {_ms(new.end_to_end_p95)}"
        )

    for note in report.notes:
        typer.secho(f"\n  {note}", dim=True)
    verdict = report.verdict
    label, color = {
        "go": ("GO", typer.colors.GREEN),
        "no_go": ("NO-GO", typer.colors.RED),
        "inconclusive": ("INCONCLUSIVE", typer.colors.YELLOW),
    }[verdict.status]
    typer.secho(f"\nVerdict: {label}", fg=color, bold=True)
    for reason in verdict.reasons:
        typer.echo(f"  ✖ {reason}" if verdict.status != "go" else f"  {reason}")
    for warning in verdict.warnings:
        typer.secho(f"  ⚠ {warning}", fg=typer.colors.YELLOW)


def _lat(side: SideReport) -> str:
    lat = side.embed_latency
    text = f"{_ms(lat.p50_ms)} / {_ms(lat.p95_ms)}" if lat else "—"
    return f"{side.side} {text}"
