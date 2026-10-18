"""What vecshift does, without a terminal: the layer under the CLI and the API.

Each operation takes a job file and plain options, and returns a result or raises
``ServiceError``. Nothing here prints, prompts, or exits: callers are told what's about to
happen through callbacks (``Consent`` before text leaves the machine, ``on_step`` for
progress) and decide how to show it.

This package must not import typer, rich, ``vecshift.ui``, or any ``vecshift.cli*`` module;
``tests/test_architecture.py`` checks.
"""

from vecshift.service.consent import Consent, Sending
from vecshift.service.errors import Declined, ServiceError
from vecshift.service.evaluation import EvalOptions, run_eval
from vecshift.service.jobs import into_qdrant, read_job
from vecshift.service.plan import PlanResult, Prepared, QdrantPrepared, prepare_plan
from vecshift.service.progress import Status, status

__all__ = [
    "Consent",
    "Declined",
    "EvalOptions",
    "PlanResult",
    "Prepared",
    "QdrantPrepared",
    "Sending",
    "ServiceError",
    "Status",
    "into_qdrant",
    "prepare_plan",
    "read_job",
    "run_eval",
    "status",
]
