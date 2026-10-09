"""Scores from ranked result lists, and latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from vecshift.eval.queries import EvalQuery

K = 10
ALL = "all"


@dataclass(frozen=True, slots=True)
class Scores:
    queries: int
    recall_at_1: float
    recall_at_10: float
    mrr_at_10: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Latency:
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile, so small samples report a value that was measured."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def latency(values: Sequence[float]) -> Latency | None:
    if not values:
        return None
    return Latency(
        len(values), percentile(values, 50), percentile(values, 95), percentile(values, 99)
    )


def score(ranked: Sequence[Sequence[str]], relevant: Sequence[frozenset[str]]) -> Scores:
    r1 = r10 = mrr = 0.0
    for row, rel in zip(ranked, relevant, strict=True):
        hits = [rank for rank, key in enumerate(row[:K], 1) if key in rel]
        r1 += (1 if hits and hits[0] == 1 else 0) / len(rel)
        r10 += len(hits) / len(rel)
        mrr += 1 / hits[0] if hits else 0.0
    n = len(relevant) or 1
    return Scores(len(relevant), r1 / n, r10 / n, mrr / n)


def by_slice(ranked: Sequence[Sequence[str]], queries: Sequence[EvalQuery]) -> dict[str, Scores]:
    """Scores for all queries, then for each query→document script pair."""
    result = {ALL: score(ranked, [q.relevant for q in queries])}
    for name in sorted({q.slice for q in queries}):
        picked = [i for i, q in enumerate(queries) if q.slice == name]
        result[name] = score([ranked[i] for i in picked], [queries[i].relevant for i in picked])
    return result


def overlap(a: Sequence[Sequence[str]], b: Sequence[Sequence[str]], k: int = K) -> float:
    """Average share of the top ``k`` results two searches have in common."""
    if not a:
        return 0.0
    total = 0.0
    for x, y in zip(a, b, strict=True):
        size = min(k, max(len(x), len(y)))
        total += len(set(x[:k]) & set(y[:k])) / size if size else 1.0
    return total / len(a)
