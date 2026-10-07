"""``vecshift bench``: compare embedding models on a sample of your own data."""

from vecshift.bench.corpus import Benchmark, CorpusError, Document, Query
from vecshift.bench.metrics import Scores
from vecshift.bench.runner import BenchResult, ModelResult, PlanItem, plan, run

__all__ = [
    "BenchResult",
    "Benchmark",
    "CorpusError",
    "Document",
    "ModelResult",
    "PlanItem",
    "Query",
    "Scores",
    "plan",
    "run",
]
