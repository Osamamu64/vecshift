"""Retrieval metrics computed with exact (brute-force) cosine search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

K = 10

Matrix = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Scores:
    recall_at_1: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float


def normalize(vectors: Sequence[Sequence[float]]) -> Matrix:
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def evaluate(docs: Matrix, queries: Matrix, relevant: Sequence[set[int]]) -> Scores:
    """Score each query's top 10 documents against its relevant document indexes."""
    k = min(K, docs.shape[0])
    sims = queries @ docs.T
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
    ranked = np.take_along_axis(top, order, axis=1)

    r1 = r10 = mrr = ndcg = 0.0
    for row, rel in zip(ranked.tolist(), relevant, strict=True):
        hits = [rank for rank, doc in enumerate(row, 1) if doc in rel]
        r1 += (1 if hits and hits[0] == 1 else 0) / len(rel)
        r10 += len(hits) / len(rel)
        mrr += 1 / hits[0] if hits else 0.0
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
        ndcg += dcg / ideal
    n = len(relevant)
    return Scores(r1 / n, r10 / n, mrr / n, ndcg / n)
