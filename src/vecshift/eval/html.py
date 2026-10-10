"""The ``vecshift eval --html`` report: one self-contained page, no network access."""

from __future__ import annotations

import math
from datetime import datetime

from vecshift.doctor.findings import Severity
from vecshift.doctor.html import _icon
from vecshift.eval.metrics import ALL
from vecshift.eval.runner import EvalReport, SideReport, SweepPoint
from vecshift.html_kit import asset, e, page, stamp

QUERY_SOURCES = {
    "proxy": "sentences taken from the rows (no LLM)",
    "labeled": "your labeled queries",
    "generated": "written by an LLM from sampled rows",
    "cross-language": "written by an LLM in the other language",
}
SETTING_NAMES = {"hnsw": "ef_search", "ivfflat": "probes"}
VERDICTS = {
    "go": ("ok", Severity.OK, "GO: the new vectors are ready"),
    "no_go": ("error", Severity.ERROR, "NO-GO: don't cut over yet"),
    "inconclusive": ("warning", Severity.WARNING, "Inconclusive: not enough evidence"),
}


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def _slice_name(name: str) -> str:
    return "All queries" if name == ALL else name.replace("→", " → ").capitalize()


def _delta(
    before: float | None, after: float | None, *, higher_is_better: bool, ms: bool = False
) -> str:
    if before is None or after is None:
        return ""
    change = after - before
    text = f"{change:+.1f} ms" if ms else f"{change:+.3f}"
    if abs(change) < (0.05 if ms else 0.0005):
        return (
            f'<span class="delta">no change vs old {e(_ms(before) if ms else _num(before))}</span>'
        )
    better = (change > 0) == higher_is_better
    arrow = "▲" if change > 0 else "▼"
    tone = "good" if better else "bad"
    was = _ms(before) if ms else _num(before)
    return (
        f'<span class="delta {tone}"><i aria-hidden="true">{arrow}</i>{e(text)} '
        f"vs old {e(was)}</span>"
    )


# --- Sections


def _verdict(report: EvalReport) -> str:
    tone, severity, title = VERDICTS[report.verdict.status]
    verdict = report.verdict
    lead = verdict.reasons[0] if verdict.reasons else ""
    rest = "".join(f"<li>{e(r)}</li>" for r in verdict.reasons[1:])
    warnings = "".join(
        f'<li class="warn">{_icon(Severity.WARNING)}<span>{e(w)}</span></li>'
        for w in verdict.warnings
    )
    items = f'<ul class="reasons">{rest}{warnings}</ul>' if rest or warnings else ""
    return (
        f'<section class="card verdict" data-tone="{tone}" aria-labelledby="verdict-title">'
        f"{_icon(severity, large=True)}"
        f'<div><h2 id="verdict-title">{e(title)}</h2><p>{e(lead)}</p>{items}</div></section>'
    )


def _tile(label: str, value: str, caption: str) -> str:
    return (
        f'<div class="card tile"><div class="tile-label">{e(label)}</div>'
        f'<div class="tile-value">{e(value)}</div><div class="tile-caption">{caption}</div></div>'
    )


def _tiles(report: EvalReport) -> str:
    old, new = report.old, report.new
    tiles = []
    after = new.scores.get(ALL)
    before = old.scores.get(ALL)
    if after:
        caption = (
            _delta(before.recall_at_10, after.recall_at_10, higher_is_better=True)
            if before
            else "the old model wasn't available to compare"
        )
        tiles.append(_tile("Recall@10, new", _num(after.recall_at_10), caption))
    if new.same_group_at_10 is not None:
        tiles.append(
            _tile(
                "Same-document neighbours",
                _num(new.same_group_at_10),
                _delta(old.same_group_at_10, new.same_group_at_10, higher_is_better=True)
                or "of each row's 10 nearest rows",
            )
        )
    if new.end_to_end_p95 is not None:
        tiles.append(
            _tile(
                "End-to-end p95, new",
                _ms(new.end_to_end_p95),
                _delta(old.end_to_end_p95, new.end_to_end_p95, higher_is_better=False, ms=True)
                or "query embedding plus search",
            )
        )
    point = new.current
    if point and point.index_recall is not None:
        setting = SETTING_NAMES.get(new.method or "", "")
        where = f"at {setting}={point.setting}" if setting else "exact search"
        tiles.append(
            _tile(
                "Index recall, new", f"{point.index_recall:.0%}", e(f"of the exact top 10, {where}")
            )
        )
    tiles.append(
        _tile(
            "Queries",
            f"{report.queries:,}",
            e(QUERY_SOURCES.get(report.query_source, report.query_source)),
        )
    )
    return (
        '<section aria-label="Key numbers" style="margin-top:12px">'
        f'<div class="tiles">{"".join(tiles)}</div></section>'
    )


def _legend(*, ring: bool = False, line: bool = False) -> str:
    shape = "key line" if line else "key"
    items = [
        f'<li><span class="{shape} old"></span>Old</li>',
        f'<li><span class="{shape} new"></span>New</li>',
    ]
    if ring:
        items.append('<li><span class="key ring"></span>Current setting</li>')
    return f'<ul class="legend">{"".join(items)}</ul>'


def _quality(report: EvalReport) -> str:
    old, new = report.old, report.new
    head = (
        "<figcaption><b>Search quality by language</b><span>Recall@10: how often the right "
        "row is in the top 10. Rows are query → document scripts.</span></figcaption>"
    )
    if not new.scores:
        return (
            f'<figure class="card wide">{head}<p class="statement">No queries were run</p></figure>'
        )
    ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    grid = "".join(f'<div class="db-grid" style="left:{t * 100:.0f}%"></div>' for t in ticks[1:-1])
    rows = []
    for name, after in new.scores.items():
        before = old.scores.get(name)
        label = _slice_name(name)
        dots = []
        if before:
            lo, hi = sorted((before.recall_at_10, after.recall_at_10))
            dots.append(
                f'<div class="db-line" style="left:{lo * 100:.2f}%;'
                f'width:{(hi - lo) * 100:.2f}%"></div>'
            )
            dots.append(
                f'<span class="db-dot old" tabindex="0" '
                f'style="left:{before.recall_at_10 * 100:.2f}%" '
                f'data-tip-value="{before.recall_at_10:.3f}" data-tip-label="{e(label)} · old" '
                f'aria-label="{e(label)}, old: recall@10 {before.recall_at_10:.3f}"></span>'
            )
        dots.append(
            f'<span class="db-dot new" tabindex="0" style="left:{after.recall_at_10 * 100:.2f}%" '
            f'data-tip-value="{after.recall_at_10:.3f}" data-tip-label="{e(label)} · new" '
            f'aria-label="{e(label)}, new: recall@10 {after.recall_at_10:.3f}"></span>'
        )
        value = (
            f"{_num(before.recall_at_10)} → {_num(after.recall_at_10)}"
            if before
            else _num(after.recall_at_10)
        )
        rows.append(
            f'<div class="db-row"><div class="db-label">{e(label)}<span>{after.queries:,} queries'
            f'</span></div><div class="db-track">{grid}{"".join(dots)}</div>'
            f'<div class="db-value">{e(value)}</div></div>'
        )
    axis = (
        '<div class="db-axis" aria-hidden="true"><span></span><div class="ticks">'
        + "".join(f'<span style="left:{t * 100:.0f}%">{t:g}</span>' for t in ticks)
        + '</div><span class="spacer"></span></div>'
    )
    table_rows = "".join(
        f'<tr><td>{e(_slice_name(name))}</td><td class="num">{after.queries:,}</td>'
        + "".join(
            f'<td class="num">{_num(getattr(s, metric) if s else None)}</td>'
            for s in (old.scores.get(name), after)
            for metric in ("recall_at_1", "recall_at_10", "mrr_at_10")
        )
        + "</tr>"
        for name, after in new.scores.items()
    )
    table = (
        '<details class="table-view"><summary>View as table</summary><div class="table-scroll">'
        '<table><thead><tr><th>Queries</th><th class="num">Count</th>'
        '<th class="num">Old R@1</th><th class="num">Old R@10</th><th class="num">Old MRR</th>'
        '<th class="num">New R@1</th><th class="num">New R@10</th><th class="num">New MRR</th>'
        f"</tr></thead><tbody>{table_rows}</tbody></table></div></details>"
    )
    overlap = (
        f'<p class="chart-note">The two sides share {report.result_overlap:.0%} of their '
        "top-10 results.</p>"
        if report.result_overlap is not None
        else ""
    )
    legend = _legend() if old.scores else ""
    return (
        f'<figure class="card wide">{head}{legend}<div class="db" role="img" '
        f'aria-label="Recall@10 by language, old and new">{"".join(rows)}</div>'
        f"{axis}{overlap}{table}</figure>"
    )


def _nice(value: float) -> float:
    """A round number at or above ``value``, for an axis maximum."""
    if value <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if step * magnitude >= value:
            return step * magnitude
    return 10 * magnitude  # pragma: no cover


def _latency_chart(report: EvalReport) -> str:
    sides = [s for s in (report.old, report.new) if s.sweep]
    head = (
        "<figcaption><b>Latency vs accuracy</b><span>Each point is one index setting: search p95 "
        "against the share of the exact top 10 the index finds. Up and left is better."
        "</span></figcaption>"
    )
    if not sides:
        why = (
            "Measured once the migration is complete"
            if report.partial
            else "No latency was measured"
        )
        return f'<figure class="card wide">{head}<p class="statement">{e(why)}</p></figure>'

    points = [(s, p) for s in sides for p in s.sweep if p.latency and p.index_recall is not None]
    slowest = max(p.latency.p95_ms for _, p in points if p.latency)
    x_step = _nice(slowest * 1.1 / 4)
    x_max = x_step * math.ceil(slowest * 1.1 / x_step)
    low = min(p.index_recall for _, p in points if p.index_recall is not None)
    # Fit the data, but always show at least five points of recall.
    y_min = max(0.0, min(0.95, math.floor(low * 20) / 20))

    def x(v: float) -> float:
        return float(100 * v / x_max)

    def y(v: float) -> float:
        return float(100 * (v - y_min) / (1 - y_min))

    y_ticks = [y_min, (y_min + 1) / 2, 1.0]
    x_ticks = [x_step * i for i in range(round(x_max / x_step) + 1)]
    grid = "".join(f'<div class="gridline" style="bottom:{y(t):.2f}%"></div>' for t in y_ticks[1:])
    vgrid = "".join(f'<div class="vline" style="left:{x(t):.2f}%"></div>' for t in x_ticks[1:])
    lines, marks, labels = [], [], []
    ends: list[tuple[float, float, str]] = []
    for side in sides:
        coords = [
            (x(p.latency.p95_ms), y(p.index_recall), p)
            for p in side.sweep
            if p.latency and p.index_recall is not None
        ]
        # In setting order, so the line shows what each step up buys and costs.
        coords.sort(key=lambda c: c[2].setting or 0)
        path = " ".join(f"{cx:.2f},{100 - cy:.2f}" for cx, cy, _ in coords)
        lines.append(f'<polyline class="{side.side}" points="{path}"/>')
        setting = SETTING_NAMES.get(side.method or "", "setting")
        for cx, cy, point in coords:
            name = f"{setting}={point.setting}" if point.setting is not None else "exact"
            current = " current" if point.current else ""
            lat = point.latency
            p95 = _ms(lat.p95_ms if lat else None)
            marks.append(
                f'<span class="lp {side.side}{current}" tabindex="0" '
                f'style="left:{cx:.2f}%;bottom:{cy:.2f}%" '
                f'data-tip-value="{point.index_recall:.0%} · p95 {e(p95)}" '
                f'data-tip-label="{side.side}, {e(name)}{" (current)" if point.current else ""}" '
                f'aria-label="{side.side}, {e(name)}: index recall {point.index_recall:.0%}, '
                f'search p95 {e(_ms(lat.p95_ms if lat else None))}"></span>'
            )
        if coords:
            ends.append((coords[-1][0], coords[-1][1], side.side))
    # End labels help only when they're apart; otherwise the legend and tooltips do the job.
    crowded = len(ends) == 2 and (
        abs(ends[0][0] - ends[1][0]) < 18 and abs(ends[0][1] - ends[1][1]) < 12
    )
    if not crowded:
        for ex, ey, name in ends:
            flip = " flip" if ex > 80 else ""
            labels.append(
                f'<span class="lc-label{flip}" style="left:{ex:.2f}%;bottom:{ey:.2f}%">'
                f"{name}</span>"
            )
    yaxis = "".join(f'<span style="bottom:{y(t):.2f}%">{t * 100:g}%</span>' for t in y_ticks)
    xaxis = "".join(f'<span style="left:{x(t):.2f}%">{round(t, 6):g} ms</span>' for t in x_ticks)
    plot = (
        f'<div class="lc" role="img" aria-label="Index recall against search p95, old and new">'
        f'<div class="yaxis" aria-hidden="true" style="height:220px">{yaxis}</div>'
        f'<div class="plot">{grid}{vgrid}<svg viewBox="0 0 100 100" preserveAspectRatio="none" '
        f'aria-hidden="true">{"".join(lines)}</svg>{"".join(marks)}{"".join(labels)}</div>'
        f'<div class="xaxis" aria-hidden="true">{xaxis}</div>'
        '<div class="axis-title" aria-hidden="true">Search p95</div></div>'
    )
    rows = "".join(
        f"<tr><td>{s.side}</td><td>{e(_setting(s, p))}</td>"
        f'<td class="num">{_num(p.index_recall)}</td>'
        + "".join(
            f'<td class="num">{e(_ms(getattr(p.latency, f) if p.latency else None))}</td>'
            for f in ("p50_ms", "p95_ms", "p99_ms")
        )
        + "</tr>"
        for s in sides
        for p in s.sweep
    )
    table = (
        '<details class="table-view"><summary>View as table</summary><div class="table-scroll">'
        '<table><thead><tr><th>Side</th><th>Setting</th><th class="num">Index recall</th>'
        '<th class="num">p50</th><th class="num">p95</th><th class="num">p99</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></details>"
    )
    samples = next((p.latency.samples for _, p in points if p.latency), 0)
    note = (
        f'<p class="chart-note">{samples} queries per setting. Search times include the network '
        "round trip from where vecshift ran.</p>"
    )
    legend = _legend(ring=True, line=True)
    return f'<figure class="card wide">{head}{legend}{plot}{note}{table}</figure>'


def _setting(side: SideReport, point: SweepPoint) -> str:
    if point.setting is None:
        return "exact (no index)"
    name = SETTING_NAMES.get(side.method or "", "setting")
    return f"{name}={point.setting}" + (" (current)" if point.current else "")


def _latency_table(report: EvalReport) -> str:
    old, new = report.old, report.new

    def search(side: SideReport, field: str) -> str:
        point = side.current
        return _ms(getattr(point.latency, field) if point and point.latency else None)

    def embed(side: SideReport, field: str) -> str:
        return _ms(getattr(side.embed_latency, field) if side.embed_latency else None)

    rows = [
        ("Query embedding p50", embed(old, "p50_ms"), embed(new, "p50_ms"), ""),
        ("Query embedding p95", embed(old, "p95_ms"), embed(new, "p95_ms"), ""),
        ("Search p50 (current setting)", search(old, "p50_ms"), search(new, "p50_ms"), ""),
        ("Search p95 (current setting)", search(old, "p95_ms"), search(new, "p95_ms"), ""),
        ("Search p99 (current setting)", search(old, "p99_ms"), search(new, "p99_ms"), ""),
        ("End-to-end p95", _ms(old.end_to_end_p95), _ms(new.end_to_end_p95), "total"),
    ]
    body = "".join(
        f'<tr class="{cls}"><td>{e(label)}</td><td class="num">{e(a)}</td>'
        f'<td class="num">{e(b)}</td></tr>'
        for label, a, b, cls in rows
    )
    return (
        '<section aria-labelledby="lat-title"><h2 id="lat-title">Latency</h2>'
        '<div class="card lat"><div class="table-scroll"><table><thead><tr><th>Step</th>'
        f'<th class="num">Old</th><th class="num">New</th></tr></thead><tbody>{body}</tbody>'
        "</table></div></div></section>"
    )


def _method(report: EvalReport) -> str:
    old, new = report.old, report.new
    parts = [
        f"<p>Old vectors: <code>{e(old.column)}</code>, {e(old.model or 'model unknown')}. "
        f"New vectors: <code>{e(new.column)}</code>, {e(new.model or '')}.</p>",
        "<p>Each query was embedded by each side's model and searched in that side's column, "
        "through its index. Documents weren't re-embedded: the vectors already in the database "
        "were searched. Languages are told apart by script (Arabic or Latin letters).</p>",
        *(f"<p>{e(side.note)}</p>" for side in (old, new) if side.note),
        *(f"<p>{e(note)}</p>" for note in report.notes),
    ]
    if report.neighbour_overlap is not None:
        parts.append(
            f"<p>From stored vectors, each sampled row's 10 nearest rows overlap "
            f"{report.neighbour_overlap:.0%} between the two sides.</p>"
        )
    return (
        '<section aria-labelledby="method-title"><h2 id="method-title">How this was measured</h2>'
        f'<div class="card method">{"".join(parts)}</div></section>'
    )


def render_html(
    report: EvalReport,
    *,
    job: str,
    version: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    when = stamp(generated_at)
    by = f"vecshift {version}" if version else "vecshift"
    kind = "partial migration" if report.partial else "complete migration"
    body = (
        _verdict(report)
        + _tiles(report)
        + '<section aria-labelledby="charts-title"><h2 id="charts-title">Old vs new</h2>'
        + f'<div class="charts">{_quality(report)}{_latency_chart(report)}</div></section>'
        + _latency_table(report)
        + _method(report)
    )
    footer = (
        f"<p>Generated by {e(by)} on {e(when)}.</p>"
        "<p>This page holds scores, timings, and column and model names, no document or query "
        "text. It loads nothing from the network.</p>"
    )
    return page(
        title=f"Migration eval · {job}",
        product="vecshift eval",
        heading=f"Old vs new vectors · {job}",
        meta=[f"{report.queries:,} queries", kind, when],
        body=body,
        footer=footer,
        generator=by,
        extra_css=asset("bench.css") + asset("eval.css"),
    )
