"""Plans for migrations into Qdrant: from a Qdrant collection, or from a pgvector table."""

from __future__ import annotations

from vecshift.connectors.pgvector.target import TargetState
from vecshift.connectors.qdrant.inspect import MAX_DIMENSIONS
from vecshift.connectors.qdrant.target import QdrantTargetState
from vecshift.connectors.qdrant.writer import DISTANCES
from vecshift.doctor.findings import Finding, Severity
from vecshift.doctor.profile import IndexProfile
from vecshift.embeddings.spec import parse_spec
from vecshift.jobs.spec import JobSpec, QdrantSource
from vecshift.planning.plan import Change, Plan, ProbeResult
from vecshift.planning.planner import (
    SampleStats,
    _model_findings,
    _resolve_dimensions,
    budget_findings,
    dims_unknown,
    estimate,
    same_model_findings,
    text_findings,
)

BYTES_PER_DIMENSION = 4
"""Qdrant stores float32 vectors unless told otherwise."""


def metric(job: JobSpec, profile: IndexProfile) -> str:
    target = job.qdrant_target
    if target.metric:
        return target.metric.value
    for index in profile.ann_indexes:
        if index.metric in DISTANCES:
            return index.metric
    return "cosine"


def _base(job: JobSpec) -> str:
    if isinstance(job.source, QdrantSource):
        return job.source.collection
    return job.pg_source.table.split(".")[-1]


def build_qdrant_plan(
    job: JobSpec,
    profile: IndexProfile,
    target: QdrantTargetState,
    sample: SampleStats,
    probe: ProbeResult | None = None,
    pg: TargetState | None = None,
) -> Plan:
    """``pg`` describes the source table when the source is pgvector."""
    spec = parse_spec(job.model)
    settings = job.qdrant_target
    from_qdrant = isinstance(job.source, QdrantSource)
    dims, dims_source = _resolve_dimensions(spec, probe)
    distance = metric(job, profile)
    findings: list[Finding] = []

    old_size = profile.declared_dimensions
    source_type = f"{profile.vector_type}({old_size})" if old_size else profile.vector_type
    rows_known = f", ~{profile.estimated_rows:,} rows" if profile.estimated_rows is not None else ""
    where = target.collection if target.vector is None else f"{target.collection}:{target.vector}"
    plan = Plan(
        job=job.name,
        source=f"{profile.target} {source_type}{rows_known}",
        target_column=where,
        model=spec.name,
        dimensions=dims,
        dimensions_source=dims_source,
        vector_type="float32 vector",
        metric=distance,
        store="qdrant",
    )

    # --- The source.
    text_findings(
        profile,
        findings,
        "Set source.text_field" if from_qdrant else "Set source.text_column",
        dropped=True,
    )
    if pg is not None:
        if len(pg.primary_key) != 1:
            findings.append(
                Finding(
                    "plan.no_primary_key",
                    Severity.ERROR,
                    "Needs a single-column primary key",
                    "Each row becomes a point whose ID comes from the row's primary key, so "
                    "runs can resume and later edits find their point.",
                    hint="Add a single-column primary key (or unique ID) to the table first.",
                )
            )
        if pg.row_security:
            findings.append(
                Finding(
                    "plan.row_security",
                    Severity.ERROR,
                    "Row-level security limits which rows apply can read",
                    "The connecting role is subject to the table's policies, so the new "
                    "collection could silently miss rows it can't see.",
                    hint="Connect as a role with BYPASSRLS for the migration.",
                )
            )
    if target.other_vectors:
        names = ", ".join(f"`{n}`" for n in target.other_vectors)
        findings.append(
            Finding(
                "plan.other_vectors",
                Severity.ERROR,
                "The collection has other vectors",
                f"Besides the one being re-embedded it holds {names}. The new collection "
                "would only carry the new vector, so those would be lost at cutover.",
                hint="Collections with several vectors aren't supported yet.",
            )
        )

    # --- The new collection and the alias.
    if target.alias is not None and target.alias_target == target.collection:
        findings.append(
            Finding(
                "plan.cut_over",
                Severity.ERROR,
                "Already cut over",
                f"The alias `{target.alias}` already points at `{target.collection}`.",
                hint="If you've cut over, pick a new target.collection to start another "
                "migration; otherwise target.collection names the collection in use.",
            )
        )
    elif target.live is not None and target.collection == target.live:
        findings.append(
            Finding(
                "plan.target_is_source",
                Severity.ERROR,
                "The new collection is the current one",
                f"`{target.collection}` already holds the vectors being replaced.",
                hint="Set target.collection to a new name.",
            )
        )
    elif target.exists:
        want = DISTANCES.get(distance)
        if target.dimensions is not None and target.dimensions == dims and target.distance == want:
            findings.append(
                Finding(
                    "plan.collection_exists",
                    Severity.INFO,
                    "The new collection already exists",
                    f"`{where}` already has matching vectors, so apply will add the points "
                    "it's missing and update the ones whose source changed.",
                )
            )
        else:
            have = f"{target.dimensions or '?'} dims, {target.distance or 'no such vector'}"
            findings.append(
                Finding(
                    "plan.collection_conflict",
                    Severity.ERROR,
                    "The new collection exists with different vectors",
                    f"`{where}` has {have}, but this job needs {dims or '?'} dims, {want}.",
                    hint="Pick another target.collection, or delete that one if it's unused.",
                )
            )
    if target.alias is None:
        findings.append(
            Finding(
                "plan.no_alias",
                Severity.WARNING,
                "No alias to switch",
                (
                    f"The application searches the collection `{target.live}` directly, so "
                    "cutover has no alias to point at the new collection."
                    if from_qdrant
                    else "Cutover creates an alias for the new collection, and none is set."
                ),
                hint=f"Set target.alias (for example `{_base(job)}_live`) and have the "
                "application search that alias; Qdrant switches aliases atomically.",
            )
        )
    elif target.alias in target.collections:
        findings.append(
            Finding(
                "plan.alias_conflict",
                Severity.ERROR,
                "The alias name is taken by a collection",
                f"`{target.alias}` is a collection, and Qdrant can't create an alias with "
                "the same name.",
                hint="Pick another target.alias.",
            )
        )

    # --- The model.
    _model_findings(spec, findings)
    if dims is None:
        dims_unknown(spec, findings, "collection")
    same_model_findings(spec, profile, findings)
    if dims and dims > MAX_DIMENSIONS:
        findings.append(
            Finding(
                "plan.too_large_to_index",
                Severity.ERROR,
                "Too many dimensions",
                f"Qdrant vectors hold up to {MAX_DIMENSIONS:,} dimensions, and this model "
                f"produces {dims:,}.",
                hint="Add dims= to the model spec to shorten the vectors, if the model allows.",
            )
        )

    # --- Estimates.
    est = estimate(job, spec, profile, sample, probe, dims * BYTES_PER_DIMENSION if dims else None)
    plan.estimates = est
    budget_findings(job, spec, est, findings)

    # --- What apply would do.
    if dims and not target.exists:
        vector = f" ({target.vector})" if target.vector else ""
        plan.changes.append(
            Change(
                "create_collection",
                f"create collection {target.collection}{vector}: {dims} dims, "
                f"{DISTANCES.get(distance, distance)}",
                note=(
                    "With the source's payload indexes ("
                    + ", ".join(sorted(target.payload_indexes))
                    + "), so filters stay fast."
                    if target.payload_indexes
                    else None
                ),
            )
        )
    rows_text = f"~{est.rows:,}" if est.rows is not None else "all"
    noun = "points" if from_qdrant else "rows"
    plan.changes.append(
        Change(
            "embed",
            f"embed {rows_text} {noun} from {profile.text_field or '?'} with {spec.name}, "
            "copying their payloads",
            note="In batches, resumable. The source is only read. Run apply again any time "
            "to catch up points added, changed, or deleted since.",
        )
    )
    if settings.index != "none":
        plan.changes.append(
            Change(
                "index",
                f"wait for Qdrant to index {target.collection} ({distance})",
                note="Qdrant builds the HNSW index itself as points arrive.",
            )
        )
    if target.alias:
        now = f" (now {target.alias_target})" if target.alias_target else ""
        plan.changes.append(
            Change(
                "cutover",
                f"later, with vecshift cutover: alias {target.alias}{now} → {target.collection}",
                note="Qdrant switches the alias atomically, so searches move at once. "
                + (
                    "Rollback points it back; cleanup deletes the old collection."
                    if from_qdrant
                    else "The table is left as it is."
                ),
            )
        )

    plan.findings = findings
    return plan
