"""Render a benchmark as a self-contained HTML leaderboard."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime

from vecshift.bench.runner import BenchResult, ModelResult
from vecshift.html_kit import asset, compact, e, page, stamp

VALUE_TOLERANCE = 0.02
"""A model within this much recall@10 of the best counts as "as good" when picking value."""

QUERY_SOURCES = {
    "proxy": "proxy queries taken from the documents",
    "generated": "written by an LLM",
    "labeled": "your labeled queries",
}


def _money(value: float | None) -> str:
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


def _gb(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f} GB" if value >= 1 else f"{value * 1000:,.0f} MB"


def _value_pick(models: list[ModelResult]) -> ModelResult | None:
    best = max(m.scores.recall_at_10 for m in models if m.scores)
    close = [m for m in models if m.scores and m.scores.recall_at_10 >= best - VALUE_TOLERANCE]
    priced = [m for m in close if m.cost_per_million_docs is not None]
    pool = priced or close
    return min(
        pool,
        key=lambda m: (
            m.cost_per_million_docs if priced else 0,
            m.gb_per_million_vectors or math.inf,
        ),
    )


def _picks(models: list[ModelResult]) -> str:
    scored = [m for m in models if m.scores]
    if not scored:
        return ""
    best = max(scored, key=lambda m: m.scores.recall_at_10 if m.scores else 0)
    value = _value_pick(scored)
    timed = [m for m in scored if m.query_ms_p50 is not None]
    fastest = min(timed, key=lambda m: m.query_ms_p50 or 0) if timed else None

    def card(label: str, m: ModelResult | None, why: str) -> str:
        if m is None:
            return ""
        return (
            f'<div class="card pick"><div class="tile-label">{e(label)}</div>'
            f'<div class="who">{e(m.name)}</div><div class="why">{e(why)}</div></div>'
        )

    if best.scores is None:  # pragma: no cover - scored models always have scores
        return ""
    value_why = ""
    if value:
        cost = value.cost_per_million_docs
        price = "free to embed" if cost == 0 else f"{_money(cost)} per 1M documents"
        value_why = f"{price}, {_gb(value.gb_per_million_vectors)} per 1M vectors"
    return (
        '<section aria-label="Picks"><div class="picks">'
        + card("Best retrieval", best, f"recall@10 {best.scores.recall_at_10:.3f}")
        + card("Best value", value, value_why)
        + card(
            "Fastest queries",
            fastest,
            f"{_ms(fastest.query_ms_p50)} median per query" if fastest else "",
        )
        + "</div></section>"
    )


def _tiles(result: BenchResult) -> str:
    tiles = [
        ("Documents", compact(result.documents), f"sampled from {result.source}"),
        ("Queries", compact(result.queries), QUERY_SOURCES.get(result.query_source, "")),
        ("Models", str(len(result.models)), "compared on the same data"),
    ]
    return (
        '<section aria-label="Key numbers"><div class="tiles">'
        + "".join(
            f'<div class="card tile"><div class="tile-label">{e(label)}</div>'
            f'<div class="tile-value">{e(value)}</div><div class="tile-caption">{e(caption)}</div>'
            "</div>"
            for label, value, caption in tiles
        )
        + "</div></section>"
    )


def _leaderboard(ranked: list[ModelResult]) -> str:
    scored = [m for m in ranked if m.scores]

    def best(get: Callable[[ModelResult], float | None], low: bool = False) -> float | None:
        values = [v for m in scored if (v := get(m)) is not None]
        if len(set(values)) < 2:  # nothing stands out when every model ties
            return None
        return min(values) if low else max(values)

    def cell(value: float | None, target: float | None, text: str) -> str:
        cls = "num best" if value is not None and value == target and len(scored) > 1 else "num"
        return f'<td class="{cls}">{e(text)}</td>'

    tops = {
        "r10": best(lambda m: m.scores.recall_at_10 if m.scores else None),
        "r1": best(lambda m: m.scores.recall_at_1 if m.scores else None),
        "mrr": best(lambda m: m.scores.mrr_at_10 if m.scores else None),
        "lat": best(lambda m: m.query_ms_p50, low=True),
        "dps": best(lambda m: m.docs_per_second),
        "cost": best(lambda m: m.cost_per_million_docs, low=True),
        "gb": best(lambda m: m.gb_per_million_vectors, low=True),
    }
    rows = []
    for rank, m in enumerate(ranked, 1):
        name = f'<td class="model"><b>{e(m.name)}</b><code>{e(m.model_tag or m.spec)}</code></td>'
        if not m.scores:
            rows.append(
                f'<tr class="failed"><td class="rank">&ndash;</td>{name}'
                f'<td colspan="7">Failed: {e(m.error or "unknown error")}</td></tr>'
            )
            continue
        s = m.scores
        r10 = s.recall_at_10
        r10_cls = "best" if r10 == tops["r10"] and len(scored) > 1 else ""
        cost = _money(m.cost_per_million_docs)
        if m.tokens_estimated and m.cost_per_million_docs:
            cost = "~" + cost
        rows.append(
            f'<tr><td class="rank">{rank}</td>{name}'
            f'<td class="num"><div class="recall"><span class="{r10_cls}">{r10:.3f}</span>'
            f'<span class="track" aria-hidden="true"><span style="width:{r10 * 100:.1f}%">'
            "</span></span></div></td>"
            + cell(s.recall_at_1, tops["r1"], f"{s.recall_at_1:.3f}")
            + cell(s.mrr_at_10, tops["mrr"], f"{s.mrr_at_10:.3f}")
            + cell(m.query_ms_p50, tops["lat"], _ms(m.query_ms_p50))
            + cell(
                m.docs_per_second,
                tops["dps"],
                f"{m.docs_per_second:,.0f}" if m.docs_per_second else "cached",
            )
            + cell(m.cost_per_million_docs, tops["cost"], cost)
            + cell(m.gb_per_million_vectors, tops["gb"], _gb(m.gb_per_million_vectors))
            + "</tr>"
        )
    head = (
        '<tr><th class="rank">#</th><th>Model</th><th class="num">Recall@10</th>'
        '<th class="num">Recall@1</th><th class="num">MRR@10</th>'
        '<th class="num">Query p50</th><th class="num">Docs/s</th>'
        '<th class="num">Per 1M docs</th><th class="num">Per 1M vectors</th></tr>'
    )
    return (
        '<section aria-labelledby="lb-title"><h2 id="lb-title">Leaderboard</h2>'
        '<div class="card lb"><div class="table-scroll"><table>'
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div></div></section>"
    )


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for exp in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for step in (1, 2, 5):
            t = step * 10.0**exp
            if lo <= t <= hi:
                ticks.append(t)
    if len(ticks) > 6:
        ticks = [t for t in ticks if str(t).lstrip("0.").startswith("1")] or ticks[::2]
    return ticks


def _scatter(
    ranked: list[ModelResult],
    title: str,
    subtitle: str,
    get_x: Callable[[ModelResult], float | None],
    fmt_x: Callable[[float], str],
    x_name: str,
) -> str:
    points = [
        (rank, m, x, m.scores.recall_at_10)
        for rank, m in enumerate(ranked, 1)
        if m.scores and (x := get_x(m)) is not None and x > 0
    ]
    head = f"<figcaption><b>{e(title)}</b><span>{e(subtitle)}</span></figcaption>"
    if len(points) < 2:
        return (
            f'<figure class="card">{head}<p class="statement">Needs two or more models</p></figure>'
        )

    xs = [p[2] for p in points]
    lo, hi = min(xs) / 1.6, max(xs) * 1.6
    span = math.log10(hi) - math.log10(lo)

    def xpos(v: float) -> float:
        return 100 * (math.log10(v) - math.log10(lo)) / span

    top = min(1.0, math.ceil(max(p[3] for p in points) * 10 + 0.5) / 10)
    ygrid = "".join(
        f'<div class="gridline" style="bottom:{f * 100:.0f}%"></div>' for f in (0.5, 1.0)
    )
    yaxis = "".join(
        f'<span style="bottom:{f * 100:.0f}%">{top * f:.2f}</span>' for f in (0.0, 0.5, 1.0)
    )
    ticks = _log_ticks(lo, hi)
    vgrid = "".join(f'<div class="vline" style="left:{xpos(t):.2f}%"></div>' for t in ticks)
    xlabels = "".join(f'<span style="left:{xpos(t):.2f}%">{e(fmt_x(t))}</span>' for t in ticks)
    dots = "".join(
        f'<div class="pt" tabindex="0" style="left:{xpos(x):.2f}%;bottom:{100 * y / top:.2f}%" '
        f'data-tip-value="{y:.3f} recall@10" data-tip-label="{e(m.name)} · {e(fmt_x(x))}" '
        f'aria-label="{rank}. {e(m.name)}: recall@10 {y:.3f}, {e(fmt_x(x))}">{rank}</div>'
        for rank, m, x, y in points
    )
    rows = "".join(
        f'<tr><td>{rank}. {e(m.name)}</td><td class="num">{e(fmt_x(x))}</td>'
        f'<td class="num">{y:.3f}</td></tr>'
        for rank, m, x, y in points
    )
    table = (
        '<details class="table-view"><summary>View as table</summary><div class="table-scroll">'
        f'<table><thead><tr><th>Model</th><th class="num">{e(x_name)}</th>'
        f'<th class="num">Recall@10</th></tr></thead><tbody>{rows}</tbody></table></div></details>'
    )
    plot = (
        f'<div class="scatter" role="img" aria-label="{e(title)}">'
        f'<div class="yaxis" aria-hidden="true" style="height:220px">{yaxis}</div>'
        f'<div class="plot">{ygrid}{vgrid}{dots}</div>'
        f'<div class="xaxis log" aria-hidden="true">{xlabels}</div>'
        f'<div class="axis-title" aria-hidden="true">{e(x_name)}, log scale</div></div>'
    )
    return f'<figure class="card">{head}{plot}{table}</figure>'


def _charts(ranked: list[ModelResult]) -> str:
    storage = _scatter(
        ranked,
        "Quality vs storage",
        "Recall@10 against float32 storage for a million vectors. Up and left is better.",
        lambda m: m.gb_per_million_vectors,
        _gb,
        "Storage per 1M vectors",
    )
    latency = _scatter(
        ranked,
        "Quality vs query speed",
        "Recall@10 against median time to embed one query. Up and left is better.",
        lambda m: m.query_ms_p50,
        _ms,
        "Query latency",
    )
    return (
        '<section aria-labelledby="charts-title"><h2 id="charts-title">Trade-offs</h2>'
        f'<div class="charts">{storage}{latency}</div>'
        '<p class="chart-note">Points are numbered by leaderboard rank.</p></section>'
    )


def _method(result: BenchResult) -> str:
    parts = [
        f"<p>Each model embedded the same {result.documents:,} documents and "
        f"{result.queries:,} queries ({e(QUERY_SOURCES.get(result.query_source, ''))}). "
        "Search is exact cosine similarity over all documents, so the numbers measure the "
        "models, not an index. Recall@10 is the share of relevant documents found in the top "
        "10 results.</p>",
        *(f"<p>{e(n)}</p>" for n in result.notes),
        *(
            f'<p class="err">✖ {e(m.name)} failed: {e(m.error)}</p>'
            for m in result.models
            if m.error
        ),
        "<p>Costs use the provider's reported token counts and the price per million tokens "
        "in each model spec; a ~ marks an estimate.</p>",
    ]
    return (
        '<section aria-labelledby="method-title"><h2 id="method-title">How this was measured</h2>'
        f'<div class="card method">{"".join(parts)}</div></section>'
    )


def render_html(
    result: BenchResult, *, version: str | None = None, generated_at: datetime | None = None
) -> str:
    when = stamp(generated_at)
    ranked = result.ranked()
    by = f"vecshift {version}" if version else "vecshift"
    body = (
        _tiles(result)
        + _picks(result.models)
        + _leaderboard(ranked)
        + _charts(ranked)
        + _method(result)
    )
    footer = (
        f"<p>Generated by {e(by)} on {e(when)}.</p>"
        "<p>This page holds scores and model names only, no document text. It loads nothing "
        "from the network.</p>"
    )
    return page(
        title=f"Embedding benchmark · {result.source}",
        product="vecshift bench",
        heading=f"Embedding models on {result.source}",
        meta=[f"{len(result.models)} models", when],
        body=body,
        footer=footer,
        generator=by,
        extra_css=asset("bench.css"),
    )
