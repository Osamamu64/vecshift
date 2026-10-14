"""Cutover and rollback for Qdrant: point the application's alias at another collection.

Qdrant applies the actions of one alias request atomically, so searches through the alias
move from the old collection to the new one at once, and rollback moves them back. Neither
collection is changed by the switch itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vecshift.connectors.qdrant.client import QdrantClient
from vecshift.connectors.qdrant.writer import QdrantWriter
from vecshift.doctor.findings import Finding, Severity

Stage = Literal["not_applied", "applied", "cut_over"]


@dataclass(frozen=True, slots=True)
class Readiness:
    stage: Stage
    findings: list[Finding]
    pending: int

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)


class QdrantSwitch:
    def __init__(
        self,
        client: QdrantClient,
        writer: QdrantWriter,
        alias: str | None,
        previous: str | None,
    ) -> None:
        """``previous`` is the collection searches use before cutover: the source collection,
        or whatever the alias pointed at for a pgvector source (``None`` if nothing)."""
        self.client = client
        self.writer = writer
        self.new = writer.layout.collection
        self.alias = alias
        self.previous = previous

    def points_at(self) -> str | None:
        """The collection the alias names now."""
        if self.alias is None:
            return None
        return dict(self.client.aliases()).get(self.alias)

    def stage(self) -> Stage:
        if self.alias is not None and self.points_at() == self.new:
            return "cut_over"
        return "applied" if self.client.exists(self.new) else "not_applied"

    def check(self) -> Readiness:
        findings: list[Finding] = []
        stage = self.stage()
        if stage == "not_applied":
            findings.append(
                Finding(
                    "cutover.not_applied",
                    Severity.ERROR,
                    "Nothing to cut over yet",
                    f"Collection {self.new} doesn't exist.",
                    hint="Run vecshift apply first.",
                )
            )
        elif stage == "cut_over":
            findings.append(
                Finding(
                    "cutover.done",
                    Severity.ERROR,
                    "Already cut over",
                    f"The alias {self.alias} already points at {self.new}.",
                    hint="Run vecshift rollback to undo it.",
                )
            )
        if self.alias is None:
            findings.append(
                Finding(
                    "cutover.no_alias",
                    Severity.ERROR,
                    "No alias to switch",
                    "Cutover moves the alias your application searches, and the job doesn't "
                    "name one.",
                    hint="Set target.alias in the job file and have the application search it.",
                )
            )
        elif self.alias in self.client.collections():
            findings.append(
                Finding(
                    "cutover.alias_conflict",
                    Severity.ERROR,
                    f"{self.alias} is a collection, not an alias",
                    "Qdrant can't create an alias with the name of an existing collection.",
                    hint="Pick another target.alias, and have the application search it.",
                )
            )
        if stage != "applied":
            return Readiness(stage, findings, 0)

        now = self.points_at()
        if self.previous is not None and now is not None and now != self.previous:
            findings.append(
                Finding(
                    "cutover.alias_moved",
                    Severity.ERROR,
                    f"The alias points at {now}, not {self.previous}",
                    "Something else moved the alias since the migration started.",
                    hint="Check which collection your application should search.",
                )
            )
        pending = self.writer.pending_count()
        if pending:
            findings.append(
                Finding(
                    "cutover.pending",
                    Severity.ERROR,
                    f"{pending:,} points have no new vector yet",
                    "After cutover they wouldn't be found by searches.",
                    hint="Cutover embeds them first; or run vecshift apply.",
                )
            )
        status = str(self.client.collection(self.new).get("status", ""))
        if status not in {"green", "grey"}:
            findings.append(
                Finding(
                    "cutover.indexing",
                    Severity.WARNING,
                    f"Qdrant is still indexing {self.new}",
                    "Searches work meanwhile, but can be slower until it's done.",
                    hint="Wait for vecshift status to show the index as ready.",
                )
            )
        sizes = self.writer.dimensions_in_use()
        if sizes and sizes != {self.writer.layout.dims}:
            findings.append(
                Finding(
                    "cutover.mixed_sizes",
                    Severity.ERROR,
                    "The new collection holds vectors of different sizes",
                    f"Found sizes {sorted(sizes)}; expected {self.writer.layout.dims}.",
                )
            )
        findings.append(
            Finding(
                "cutover.model_switch",
                Severity.INFO,
                "Switch your application's model at cutover",
                "Searches through the alias need queries embedded with the new model.",
            )
        )
        return Readiness(stage, findings, pending)

    def point(self, collection: str) -> None:
        """Point the alias at ``collection``, atomically, creating it if needed."""
        if self.alias is None:
            raise ValueError("no alias")
        actions: list[dict[str, Any]] = []
        if self.points_at() is not None:
            actions.append({"delete_alias": {"alias_name": self.alias}})
        actions.append({"create_alias": {"alias_name": self.alias, "collection_name": collection}})
        self.client.update_aliases(actions)

    def remove_alias(self) -> None:
        if self.alias is not None and self.points_at() is not None:
            self.client.update_aliases([{"delete_alias": {"alias_name": self.alias}}])
