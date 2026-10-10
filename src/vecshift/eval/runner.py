"""Compare the old and new vectors: search quality per language, and latency against accuracy.

The runner talks to the store through a small searcher interface and to the models through
the embedding contract, so it can be tested with fakes and reused for other stores. Every
search it makes is a read.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from vecshift.eval.metrics import ALL, K, Latency, Scores, by_slice, latency, overlap
from vecshift.eval.queries import EvalQuery

Side = Literal["old", "new"]
SIDES: tuple[Side, Side] = ("old", "new")
INDEX_RECALL_WARNING = 0.9


class Searcher(Protocol):
    def search(
        self, side: Side, vector: Sequence[float], k: int, setting: int | None = None
    ) -> list[str]:
        """Top ``k`` row keys using the index, at a search setting (None: the default)."""
        ...

    def exact(self, side: Side, vector: Sequence[float], k: int) -> list[str]:
        """Top ``k`` row keys by exact search, without the index."""
        ...

    def index(self, side: Side) -> tuple[str | None, int | None, list[int]]:
        """(index method, current search setting, settings to sweep)."""
        ...

    def neighbours(self, side: Side, key: str, k: int) -> tuple[str | None, list[tuple[str, str]]]:
        """A row's own group and its ``k`` nearest other rows with their groups."""
        ...

    def vectors(self, side: Side, keys: Sequence[str]) -> list[list[float]]:
        """Stored vectors of the given rows, to use as queries when no model can embed."""
        ...


class Embedder(Protocol):
    async def embed(
        self, texts: Sequence[str], mode: Literal["document", "query"] = ...
    ) -> list[list[float]]: ...


@dataclass(slots=True)
class SweepPoint:
    setting: int | None
    """``hnsw.ef_search`` or ``ivfflat.probes``; None when there's no index."""
    index_recall: float | None
    """Share of the exact top 10 that the index found."""
    latency: Latency | None
    current: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": self.setting,
            "index_recall": self.index_recall,
            "latency": self.latency.to_dict() if self.latency else None,
            "current": self.current,
        }


@dataclass(slots=True)
class SideReport:
    side: Side
    column: str
    model: str | None
    embedded: bool = False
    """Whether queries could be embedded with this side's model."""
    note: str | None = None
    scores: dict[str, Scores] = field(default_factory=dict)
    embed_latency: Latency | None = None
    method: str | None = None
    sweep: list[SweepPoint] = field(default_factory=list)
    same_group_at_10: float | None = None
    """Share of a row's nearest rows that come from the same document (or group)."""

    @property
    def current(self) -> SweepPoint | None:
        return next((p for p in self.sweep if p.current), self.sweep[0] if self.sweep else None)

    @property
    def end_to_end_p95(self) -> float | None:
        point = self.current
        if not point or not point.latency:
            return None
        embed = self.embed_latency.p95_ms if self.embed_latency else 0.0
        return embed + point.latency.p95_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "model": self.model,
            "embedded": self.embedded,
            "note": self.note,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "embed_latency": self.embed_latency.to_dict() if self.embed_latency else None,
            "index_method": self.method,
            "sweep": [p.to_dict() for p in self.sweep],
            "same_group_at_10": self.same_group_at_10,
        }


@dataclass(frozen=True, slots=True)
class Gates:
    tolerance: float = 0.02
    """How far recall@10 may fall (absolute) before the verdict is no-go."""
    min_slice: int = 30
    """Slices with fewer queries are reported but don't decide the verdict."""
    max_p95_ms: float | None = None
    max_slowdown_pct: float | None = None


@dataclass(slots=True)
class Verdict:
    status: Literal["go", "no_go", "inconclusive"]
    reasons: list[str] = field(default_factory=list)
    """Why it isn't go (or, for go, what was checked)."""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": self.reasons, "warnings": self.warnings}


@dataclass(slots=True)
class EvalReport:
    mode: Literal["queries", "vectors"]
    """``queries``: both models embedded the queries. ``vectors``: the old model couldn't,
    so the old side is judged from its stored vectors alone."""
    partial: bool
    query_source: str
    queries: int
    old: SideReport
    new: SideReport
    result_overlap: float | None = None
    neighbour_overlap: float | None = None
    notes: list[str] = field(default_factory=list)
    verdict: Verdict = field(default_factory=lambda: Verdict("inconclusive"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "partial": self.partial,
            "query_source": self.query_source,
            "queries": self.queries,
            "old": self.old.to_dict(),
            "new": self.new.to_dict(),
            "result_overlap_at_10": self.result_overlap,
            "neighbour_overlap_at_10": self.neighbour_overlap,
            "notes": self.notes,
            "verdict": self.verdict.to_dict(),
        }


async def _embed(
    embedder: Embedder, texts: list[str], samples: int
) -> tuple[list[list[float]], Latency | None]:
    vectors = await embedder.embed(texts, "query")
    timings = []
    for text in texts[:samples]:
        started = time.perf_counter()
        await embedder.embed([text], "query")
        timings.append((time.perf_counter() - started) * 1000)
    return vectors, latency(timings)


def _sweep(
    searcher: Searcher, side: Side, vectors: list[list[float]], warmup: int = 5
) -> tuple[str | None, list[SweepPoint]]:
    method, current, settings = searcher.index(side)
    for vector in vectors[:warmup]:
        searcher.search(side, vector, K)
    exact = [searcher.exact(side, v, K) for v in vectors]
    if method is None:
        scans = []
        for vector in vectors:
            started = time.perf_counter()
            searcher.exact(side, vector, K)
            scans.append((time.perf_counter() - started) * 1000)
        return None, [SweepPoint(None, 1.0, latency(scans), current=True)]
    found: dict[int, list[list[str]]] = {s: [] for s in settings}
    timings: dict[int, list[float]] = {s: [] for s in settings}
    # Each query runs at every setting in turn, so caches warming up during the run
    # don't flatter whichever setting happens to be measured last.
    for i, vector in enumerate(vectors):
        order = settings[i % len(settings) :] + settings[: i % len(settings)]
        for setting in order:
            started = time.perf_counter()
            found[setting].append(searcher.search(side, vector, K, setting))
            timings[setting].append((time.perf_counter() - started) * 1000)
    points = [
        SweepPoint(s, overlap(found[s], exact), latency(timings[s]), s == current) for s in settings
    ]
    return method, points


def _neighbours(
    searcher: Searcher, side: Side, keys: Sequence[str]
) -> tuple[list[list[str]], float | None]:
    lists, shares = [], []
    for key in keys:
        own, found = searcher.neighbours(side, key, K)
        lists.append([k for k, _ in found])
        if own is not None and found:
            shares.append(sum(1 for _, group in found if group == own) / len(found))
    return lists, (sum(shares) / len(shares) if shares else None)


def decide(report: EvalReport, gates: Gates) -> Verdict:
    """Go, no-go, or inconclusive, with the reasons."""
    verdict = Verdict("go")
    old, new = report.old, report.new
    evidence = False
    if report.mode == "queries":
        for name, before in old.scores.items():
            after = new.scores.get(name)
            if after is None:
                continue
            label = "overall" if name == ALL else name
            if before.queries < gates.min_slice:
                verdict.warnings.append(
                    f"Only {before.queries} {label} queries: shown, but too few to decide on."
                )
                continue
            evidence = True
            if after.recall_at_10 < before.recall_at_10 - gates.tolerance:
                verdict.reasons.append(
                    f"Recall@10 {label} fell from {before.recall_at_10:.3f} to "
                    f"{after.recall_at_10:.3f}."
                )
    if old.same_group_at_10 is not None and new.same_group_at_10 is not None:
        evidence = True
        if new.same_group_at_10 < old.same_group_at_10 - gates.tolerance:
            verdict.reasons.append(
                "Fewer nearest rows come from the same document: "
                f"{old.same_group_at_10:.3f} before, {new.same_group_at_10:.3f} after."
            )
    if report.mode == "vectors":
        verdict.warnings.append(
            "The old model wasn't available to embed queries, so the old side was judged from "
            "its stored vectors only. That's a weaker check than comparing search results."
        )

    old_p95, new_p95 = old.end_to_end_p95, new.end_to_end_p95
    if gates.max_p95_ms is not None and new_p95 is not None and new_p95 > gates.max_p95_ms:
        verdict.reasons.append(
            f"New p95 latency is {new_p95:.1f} ms, over the {gates.max_p95_ms:g} ms limit."
        )
    if gates.max_slowdown_pct is not None and old_p95 and new_p95 is not None:
        limit = old_p95 * (1 + gates.max_slowdown_pct / 100)
        if new_p95 > limit:
            verdict.reasons.append(
                f"New p95 latency is {new_p95:.1f} ms against {old_p95:.1f} ms before, more "
                f"than {gates.max_slowdown_pct:g}% slower."
            )
    for side in (old, new):
        point = side.current
        if point and point.index_recall is not None and point.index_recall < INDEX_RECALL_WARNING:
            verdict.warnings.append(
                f"The {side.side} index finds {point.index_recall:.0%} of the exact top 10 at "
                "its current setting; a higher setting trades speed for accuracy."
            )

    if verdict.reasons:
        verdict.status = "no_go"
    elif evidence:
        decided = [
            "overall" if name == ALL else name
            for name, scores in old.scores.items()
            if scores.queries >= gates.min_slice
        ]
        if report.mode == "queries" and decided:
            verdict.reasons.append(
                f"Recall@10 held within {gates.tolerance:g} or improved: {', '.join(decided)}."
            )
        if old.same_group_at_10 is not None and new.same_group_at_10 is not None:
            verdict.reasons.append("Nearest rows from the same document held or improved.")
    else:
        verdict.status = "inconclusive"
        verdict.reasons.append(
            "Not enough evidence to decide: use more queries, give the old model "
            "(source.model), or a way to group rows by document (--group-by)."
        )
    return verdict


async def run(
    searcher: Searcher,
    embedders: dict[Side, Embedder | None],
    queries: Sequence[EvalQuery],
    neighbour_keys: Sequence[str],
    *,
    old: SideReport,
    new: SideReport,
    query_source: str,
    partial: bool,
    gates: Gates | None = None,
    sweep_queries: int = 30,
    latency_samples: int = 20,
    notes: Sequence[str] = (),
    on_step: Callable[[str], None] = lambda s: None,
) -> EvalReport:
    """Evaluate both sides. ``embedders['old']`` is None when the old model is unavailable."""
    sides = {"old": old, "new": new}
    report = EvalReport(
        mode="queries" if embedders.get("old") is not None else "vectors",
        partial=partial,
        query_source=query_source,
        queries=len(queries),
        old=old,
        new=new,
        notes=list(notes),
    )
    texts = [q.text for q in queries]
    ranked: dict[Side, list[list[str]]] = {}
    for name in SIDES:
        embedder = embedders.get(name)
        side = sides[name]
        if embedder is None or not texts:
            continue
        on_step(f"Embedding {len(texts)} queries with the {name} model")
        vectors, side.embed_latency = await _embed(embedder, texts, latency_samples)
        side.embedded = True
        on_step(f"Searching the {name} column")
        if partial:
            ranked[name] = [searcher.exact(name, v, K) for v in vectors]
        else:
            ranked[name] = [searcher.search(name, v, K) for v in vectors]
            on_step(f"Measuring {name} search latency and index accuracy")
            side.method, side.sweep = _sweep(searcher, name, vectors[:sweep_queries])
        side.scores = by_slice(ranked[name], queries)
    if len(ranked) == 2:
        report.result_overlap = overlap(ranked["old"], ranked["new"])
    for name in SIDES:
        side = sides[name]
        if side.sweep or partial or not neighbour_keys:
            continue
        # Search speed doesn't depend on the model, so stored vectors stand in for queries.
        on_step(f"Measuring {name} search latency with stored vectors")
        stored = searcher.vectors(name, neighbour_keys[:sweep_queries])
        if stored:
            side.method, side.sweep = _sweep(searcher, name, stored)
            report.notes.append(
                f"The {name} side's latency was measured with stored row vectors as queries, "
                "since its model couldn't embed queries."
            )

    if neighbour_keys:
        on_step("Comparing each sampled row's nearest rows")
        old_lists, old.same_group_at_10 = _neighbours(searcher, "old", neighbour_keys)
        new_lists, new.same_group_at_10 = _neighbours(searcher, "new", neighbour_keys)
        report.neighbour_overlap = overlap(old_lists, new_lists)
    if partial:
        report.notes.append(
            "The migration isn't finished, so both sides were searched exactly, over the rows "
            "that have both vectors. Latency is measured once it's complete."
        )
    report.verdict = decide(report, gates or Gates())
    return report
