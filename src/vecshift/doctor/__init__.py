"""``vecshift doctor``: read-only diagnosis of an existing vector index."""

from vecshift.doctor.checks import run_checks
from vecshift.doctor.findings import Finding, Report, Severity
from vecshift.doctor.profile import AnnIndex, IndexProfile, Sample

__all__ = [
    "AnnIndex",
    "Finding",
    "IndexProfile",
    "Report",
    "Sample",
    "Severity",
    "run_checks",
]
