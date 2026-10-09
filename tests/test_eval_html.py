"""The eval HTML report."""

import asyncio
from typing import Any

from tests.test_eval import FakeModel, FakeSearcher, run_fake
from vecshift.eval import EvalQuery, SideReport, SweepPoint, run
from vecshift.eval.html import render_html
from vecshift.eval.metrics import Latency
from vecshift.eval.runner import EvalReport


def html(report: EvalReport) -> str:
    return render_html(report, job="docs", version="9.9")


def test_full_report() -> None:
    page = html(run_fake())
    assert 'data-tone="ok"' in page and "GO: the new vectors are ready" in page
    assert "Search quality by language" in page and "Latency vs accuracy" in page
    assert page.count('class="db-row"') == 2, "all queries, latin→latin"
    assert 'class="lp new current"' in page and "ef_search=40 (current)" in page
    assert "<polyline" in page and "End-to-end p95" in page
    assert "vecshift 9.9" in page


def test_untrusted_names_are_escaped() -> None:
    report = run_fake(new=SideReport("new", "emb<b>", "<script>alert(1)</script>"))
    page = html(report)
    assert "<script>alert(1)" not in page and "&lt;script&gt;alert(1)" in page
    assert "emb<b>" not in page


def test_no_query_text_in_the_page() -> None:
    secret = "patient 4471 diagnosis"
    queries = [EvalQuery(f"{i} {secret}", frozenset({str(i)}), "latin", "latin") for i in range(40)]

    class Model(FakeModel):
        async def embed(self, texts: Any, mode: str = "query") -> list[list[float]]:
            return [[float(t.split()[0])] for t in texts]

    report = asyncio.run(
        run(
            FakeSearcher(),
            {"old": Model(), "new": Model()},
            queries,
            ["1"],
            old=SideReport("old", "a", "m"),
            new=SideReport("new", "b", "m"),
            query_source="proxy",
            partial=False,
            sweep_queries=5,
            latency_samples=2,
        )
    )
    assert secret not in html(report)


def test_partial_and_vectors_only_reports() -> None:
    partial = html(run_fake(partial=True))
    assert "Measured once the migration is complete" in partial

    blind = html(run_fake(embedders={"old": None, "new": FakeModel()}))
    assert "the old model wasn" in blind
    quality = blind.split("Search quality by language")[1].split("</figure>")[0]
    assert 'class="db-dot old"' not in quality and 'class="legend"' not in quality


def test_crowded_end_labels_are_dropped() -> None:
    def side(name: str, ms: float, recall: float) -> SideReport:
        result = SideReport(name, name, "m")  # type: ignore[arg-type]
        result.method = "hnsw"
        result.sweep = [SweepPoint(40, recall, Latency(10, ms, ms, ms), current=True)]
        return result

    def page(new_ms: float) -> str:
        report = EvalReport(
            "queries", False, "proxy", 10, side("old", 2.0, 0.99), side("new", new_ms, 0.99)
        )
        return html(report)

    assert 'class="lc-label' not in page(2.05)
    assert page(9.0).count('class="lc-label') == 2
