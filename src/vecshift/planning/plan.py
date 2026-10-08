"""The result of planning a migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from vecshift.doctor.findings import Finding, Severity


@dataclass(frozen=True, slots=True)
class Change:
    kind: str
    """``add_column``, ``embed``, ``index``, or ``cutover``."""
    summary: str
    sql: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Measured by embedding a few sample documents with the real model."""

    documents: int
    dimensions: int
    tokens_per_char: float
    docs_per_second: float


@dataclass(slots=True)
class Estimates:
    rows: int | None = None
    rows_exact: bool = False
    tokens: int | None = None
    tokens_method: str | None = None
    cost_usd: float | None = None
    requests: int | None = None
    seconds: float | None = None
    seconds_method: str | None = None
    new_bytes: int | None = None
    old_bytes: int | None = None
    index_memory_bytes: int | None = None
    maintenance_work_mem: int | None = None


@dataclass(slots=True)
class Plan:
    job: str
    source: str
    target_column: str
    model: str
    dimensions: int | None
    dimensions_source: str | None
    """``spec``, ``known``, ``probe``, or ``None`` when unknown."""
    vector_type: str
    changes: list[Change] = field(default_factory=list)
    estimates: Estimates = field(default_factory=Estimates)
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: -f.severity.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job,
            "ok": self.ok,
            "source": self.source,
            "target_column": self.target_column,
            "model": self.model,
            "dimensions": self.dimensions,
            "dimensions_source": self.dimensions_source,
            "vector_type": self.vector_type,
            "changes": [asdict(c) for c in self.changes],
            "estimates": asdict(self.estimates),
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }
