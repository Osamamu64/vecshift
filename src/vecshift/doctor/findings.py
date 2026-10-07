"""Findings and the report that collects them."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {Severity.OK: 0, Severity.INFO: 1, Severity.WARNING: 2, Severity.ERROR: 3}


@dataclass(frozen=True, slots=True)
class Finding:
    id: str
    """Stable identifier such as ``dims.mixed``, for scripts and CI."""
    severity: Severity
    title: str
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": str(self.severity),
            "title": self.title,
            "detail": self.detail,
            "hint": self.hint,
        }


@dataclass(slots=True)
class Report:
    store: str
    target: str
    vector_type: str
    declared_dimensions: int | None
    estimated_rows: int | None
    sample_rows: int
    sample_method: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst(self) -> Severity:
        return max((f.severity for f in self.findings), key=lambda s: s.rank, default=Severity.OK)

    def count(self, severity: Severity) -> int:
        return sum(1 for f in self.findings if f.severity is severity)

    def sorted_findings(self) -> list[Finding]:
        """Most severe first; ties keep check order."""
        return sorted(self.findings, key=lambda f: -f.severity.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "target": self.target,
            "vector_type": self.vector_type,
            "declared_dimensions": self.declared_dimensions,
            "estimated_rows": self.estimated_rows,
            "sample": {"rows": self.sample_rows, "method": self.sample_method},
            "summary": {str(s): self.count(s) for s in Severity},
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }
