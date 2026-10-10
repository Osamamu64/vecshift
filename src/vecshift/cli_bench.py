"""The ``vecshift bench`` command."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift import __version__
from vecshift.cli_style import warn_if_password_on_command_line

if TYPE_CHECKING:
    from vecshift.bench import BenchResult, PlanItem
    from vecshift.embeddings import ModelSpec

DEFAULT_MODELS = ["hash/256", "hash/1024"]


def _fail(message: str, hint: str | None = None) -> typer.Exit:
    typer.secho(message, err=True, fg=typer.colors.RED)
    if hint:
        typer.echo(f"→ {hint}", err=True)
    return typer.Exit(2)


def _note(message: str) -> None:
    from vecshift import ui

    if ui.fancy():
        ui.note(f"  · {message}")
    else:
        typer.secho(message, err=True, dim=True)


def money(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "free"
    if value < 0.01:
        return "<$0.01"
    return f"${value:,.2f}" if value < 1000 else f"${value:,.0f}"


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f} ms" if value < 1 else f"{value:,.1f} ms"


def _confirm(items: list[PlanItem], generator: ModelSpec | None, n_docs: int, yes: bool) -> None:
    from vecshift import ui

    remote = [i for i in items if not i.spec.is_local]
    remote_gen = generator is not None and not generator.is_local
    if ui.fancy():
        ui.title("bench")
        rows: list[list[str | Any]] = [
            [
                item.spec.name,
                f"~{item.est_tokens:,}",
                money(item.est_cost) if item.est_cost is not None else "unknown price",
                "local" if item.spec.is_local else (item.spec.url or ""),
            ]
            for item in items
        ]
        if generator is not None:
            rows.append([generator.name, "generates queries", "", generator.url or ""])
        ui.table(["Model", "Tokens", "Cost", "Where"], rows, right=frozenset({1, 2}))
        if not remote and not remote_gen:
            return
        ui.warn(f"  This sends the text of {n_docs:,} sampled documents to the services above.")
        if not yes and not ui.confirm("Continue?", default=False):
            raise typer.Exit(1)
        return
    typer.echo("Plan", err=True)
    for item in items:
        where = "local" if item.spec.is_local else (item.spec.url or "")
        cost = money(item.est_cost) if item.est_cost is not None else "unknown price"
        typer.echo(
            f"  {item.spec.name:<34} ~{item.est_tokens:,} tokens  {cost:>10}  {where}", err=True
        )
    if generator is not None:
        typer.echo(f"  {generator.name:<34} generates queries  {generator.url}", err=True)
    if not remote and not remote_gen:
        return
    typer.echo(
        f"\nThis sends the text of {n_docs:,} sampled documents to the remote services above.",
        err=True,
    )
    if yes:
        return
    if not sys.stdin.isatty():
        raise _fail("Not sending data without confirmation.", "Pass --yes to proceed.")
    if not typer.confirm("Continue?", err=True):
        raise typer.Exit(1)


def _render(result: BenchResult) -> None:
    from vecshift import ui
    from vecshift.bench.runner import ModelResult

    fancy = ui.fancy()

    if not fancy:
        typer.secho(
            f"vecshift bench · {result.documents:,} documents, {result.queries:,} queries "
            f"({result.query_source}) from {result.source}",
            bold=True,
        )
    rows: list[tuple[str, ...]] = []
    ranked = result.ranked()
    for rank, m in enumerate(ranked, 1):
        if m.scores is None:
            continue
        s = m.scores
        speed = f"{m.docs_per_second:,.0f}" if m.docs_per_second else "cached"
        cost = money(m.cost_per_million_docs)
        if m.tokens_estimated and m.cost_per_million_docs:
            cost = "~" + cost
        rows.append(
            (
                str(rank),
                m.name,
                f"{s.recall_at_10:.3f}",
                f"{s.recall_at_1:.3f}",
                f"{s.mrr_at_10:.3f}",
                _ms(m.query_ms_p50),
                speed,
                cost,
                f"{m.gb_per_million_vectors:,.1f} GB" if m.gb_per_million_vectors else "—",
            )
        )
    header = (
        "#",
        "Model",
        "Recall@10",
        "Recall@1",
        "MRR@10",
        "Query p50",
        "Docs/s",
        "Per 1M docs",
        "Per 1M vecs",
    )
    if rows and fancy:
        from rich.text import Text

        ui.section(
            f"Results · {result.documents:,} documents, {result.queries:,} queries "
            f"({result.query_source}) from {result.source}"
        )
        styled: list[list[str | Text]] = [
            [
                Text(r[0], style="bold green" if r[0] == "1" else "dim"),
                Text(r[1], style="bold"),
                *r[2:],
            ]
            for r in rows
        ]
        ui.table(header, styled, right=frozenset({0, 2, 3, 4, 5, 6, 7, 8}))
    elif rows:
        widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
        right = {0, 2, 3, 4, 5, 6, 7, 8}

        def line(cells: tuple[str, ...]) -> str:
            return "  ".join(
                c.rjust(w) if i in right else c.ljust(w)
                for i, (c, w) in enumerate(zip(cells, widths, strict=True))
            )

        typer.echo()
        typer.secho(line(header), bold=True)
        for r in rows:
            typer.echo(line(r))

    failed: list[ModelResult] = [m for m in ranked if m.error]
    for m in failed:
        if fancy:
            ui.error(f"{m.name} failed: {m.error}")
        else:
            typer.secho(f"\n✖ {m.name} failed: {m.error}", fg=typer.colors.RED)
    for note in result.notes:
        if fancy:
            ui.note(f"  {note}")
        else:
            typer.secho(f"\n{note}", dim=True)
    if fancy:
        ui.console.print()


def bench(
    ctx: typer.Context,
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to compare, as provider/model[,option=value]. Repeat for each model. "
            "Defaults to two free hashing baselines.",
            show_default=False,
        ),
    ] = None,
    docs: Annotated[
        Path | None,
        typer.Option(help='JSONL file of documents, one {"id": ..., "text": ...} per line.'),
    ] = None,
    dsn: Annotated[
        str | None,
        typer.Option(
            envvar=["VECSHIFT_DSN", "DATABASE_URL"],
            help="PostgreSQL connection string, to sample documents from a pgvector table.",
            show_default=False,
        ),
    ] = None,
    table: Annotated[
        str | None, typer.Option(help="Table to sample, as table or schema.table.")
    ] = None,
    column: Annotated[
        str | None, typer.Option(help="Vector column, if the table has several.")
    ] = None,
    text_column: Annotated[str | None, typer.Option(help="Column holding the text.")] = None,
    sample_size: Annotated[
        int, typer.Option("--sample", min=10, max=100_000, help="Documents to sample.")
    ] = 1000,
    num_queries: Annotated[int, typer.Option(min=5, max=10_000, help="Queries to evaluate.")] = 100,
    queries: Annotated[
        Path | None,
        typer.Option(help='JSONL of labeled queries: {"query": ..., "relevant": [ids]} per line.'),
    ] = None,
    generate_queries: Annotated[
        str | None,
        typer.Option(
            help="Write realistic queries with this chat model, e.g. openai/gpt-4o-mini.",
            show_default=False,
        ),
    ] = None,
    save_queries: Annotated[
        Path | None, typer.Option(help="Save generated queries here, to reuse with --queries.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed for sampling.")] = 7,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Don't read or write the embedding cache.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Don't ask before sending data.")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    html: Annotated[
        Path | None,
        typer.Option("--html", dir_okay=False, help="Also write an HTML leaderboard to this file."),
    ] = None,
) -> None:
    """Compare embedding models on a sample of your own data."""
    warn_if_password_on_command_line(ctx, dsn)
    try:
        from vecshift.bench import Benchmark, CorpusError, Query, plan, run
        from vecshift.bench import corpus as corpus_mod
        from vecshift.embeddings import EmbeddingCache, SpecError, parse_spec
    except ImportError as exc:  # pragma: no cover - a broken install
        raise _fail(
            "A package vecshift needs is missing.",
            "Reinstall: pip install --force-reinstall vecshift",
        ) from exc

    if queries and generate_queries:
        raise _fail("Use either --queries or --generate-queries, not both.")
    if save_queries and not generate_queries:
        raise _fail(
            "--save-queries only works with --generate-queries.",
            "Proxy queries change the documents they come from, so they can't be reused.",
        )
    try:
        specs = [parse_spec(m) for m in (model or DEFAULT_MODELS)]
        generator_spec = parse_spec(generate_queries) if generate_queries else None
    except SpecError as exc:
        raise _fail(str(exc)) from exc
    if not model:
        _note(
            "No --model given, so comparing two free hashing baselines. Add e.g. "
            "-m openai/text-embedding-3-small -m ollama/nomic-embed-text"
        )

    try:
        labeled = corpus_mod.load_queries(queries) if queries else None
        wanted = {r for q in labeled for r in q.relevant} if labeled else set()
        if docs is not None:
            documents = corpus_mod.sample(
                corpus_mod.load_documents(docs), sample_size, wanted, seed
            )
            source = docs.name
        elif dsn:
            documents, source = _from_database(dsn, table, column, text_column, sample_size, wanted)
        else:
            raise _fail(
                "No documents to benchmark.",
                "Pass --docs FILE, or a database with --dsn (or VECSHIFT_DSN) and --table.",
            )
    except CorpusError as exc:
        raise _fail(str(exc)) from exc

    notes: list[str] = []
    if labeled is not None:
        ids = {d.id for d in documents}
        kept = [Query(q.text, q.relevant & ids) for q in labeled if q.relevant & ids]
        if len(kept) < len(labeled):
            notes.append(
                f"{len(labeled) - len(kept)} labeled queries had no relevant document "
                "in the corpus and were skipped."
            )
        if not kept:
            raise _fail("None of the labeled queries' documents are in the corpus.")
        benchmark = Benchmark(documents, kept, "labeled", notes)
    elif generator_spec is not None:
        benchmark = Benchmark(documents, [], "generated", notes)
    else:
        try:
            edited, proxy, proxy_notes = corpus_mod.proxy_queries(documents, num_queries, seed)
        except CorpusError as exc:
            raise _fail(str(exc)) from exc
        benchmark = Benchmark(edited, proxy, "proxy", proxy_notes)

    if generator_spec is not None:
        est = Benchmark(documents, [Query("x" * 60, frozenset())] * num_queries, "", [])
        _confirm(plan(est, specs), generator_spec, len(documents), yes)
        benchmark.queries = asyncio.run(_generate(generator_spec, documents, num_queries, seed))
        if not benchmark.queries:
            raise _fail("The chat model didn't return any queries.")
        benchmark.notes.append(f"Queries were written by {generator_spec.name}.")
        if save_queries:
            corpus_mod.save_queries(save_queries, benchmark.queries)
            _note(f"Saved {len(benchmark.queries)} queries to {save_queries}")
    else:
        _confirm(plan(benchmark, specs), None, len(benchmark.documents), yes)

    cache = None if no_cache else EmbeddingCache()
    result = asyncio.run(
        run(benchmark, specs, cache, source, on_model=lambda s: _note(f"Embedding with {s.name}…"))
    )
    if cache:
        cache.close()

    if html is not None:
        from vecshift.bench.html import render_html

        try:
            html.write_text(render_html(result, version=__version__), encoding="utf-8")
        except OSError as exc:
            raise _fail(f"Couldn't write {html}: {exc.strerror}") from exc
    if output_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _render(result)
    if html is not None:
        typer.echo(f"HTML leaderboard written to {html}", err=output_json)
    if all(m.error for m in result.models):
        raise typer.Exit(1)


async def _generate(spec: ModelSpec, documents: list, n: int, seed: int) -> list:  # type: ignore[type-arg]
    from vecshift.bench.generate import GenerationError, QueryGenerator

    try:
        generator = QueryGenerator(spec)
    except GenerationError as exc:
        raise _fail(str(exc)) from exc
    try:
        return await generator.generate(documents, n, seed)
    except GenerationError as exc:
        raise _fail(str(exc)) from exc
    finally:
        await generator.aclose()


def _from_database(
    dsn: str,
    table: str | None,
    column: str | None,
    text_column: str | None,
    size: int,
    wanted: set[str],
) -> tuple[list, str]:  # type: ignore[type-arg]
    try:
        from vecshift.connectors import pgvector
        from vecshift.connectors.pgvector.documents import sample_documents
    except ImportError as exc:  # pragma: no cover
        raise _fail(
            "The PostgreSQL driver is missing.",
            "Reinstall: pip install --force-reinstall vecshift",
        ) from exc
    from vecshift.bench import Document
    from vecshift.bench.corpus import clip

    try:
        settings = pgvector.prepare(dsn)
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        rows, name = sample_documents(conn, table, column, text_column, size, wanted)
    except pgvector.TargetSelectionError as exc:
        raise _fail(str(exc)) from exc
    finally:
        conn.rollback()
        conn.close()
    return [Document(i, clip(t)) for i, t in rows], name
