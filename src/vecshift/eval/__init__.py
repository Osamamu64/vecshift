"""Compare a migration's old and new vectors before cutover: quality and latency."""

from vecshift.eval.metrics import Latency, Scores
from vecshift.eval.queries import EvalQuery, proxy_queries, script
from vecshift.eval.runner import EvalReport, Gates, SideReport, SweepPoint, Verdict, decide, run

__all__ = [
    "EvalQuery",
    "EvalReport",
    "Gates",
    "Latency",
    "Scores",
    "SideReport",
    "SweepPoint",
    "Verdict",
    "decide",
    "proxy_queries",
    "run",
    "script",
]
