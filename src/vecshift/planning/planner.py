"""Turn a job, what's in the database, and what's known about the model into a plan."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from vecshift.connectors.pgvector.inspect import MAX_INDEXABLE_DIMENSIONS
from vecshift.connectors.pgvector.target import TargetState, add_column_sql, index_sql
from vecshift.doctor.findings import Finding, Severity
from vecshift.doctor.profile import IndexProfile
from vecshift.embeddings.spec import ModelSpec, parse_spec
from vecshift.jobs.spec import IndexMethod, JobSpec
from vecshift.planning.plan import Change, Estimates, Plan, ProbeResult

CHARS_PER_TOKEN = 4.0
"""Rough English average, used when there's no probe to measure the real ratio."""

HNSW_BYTES_PER_ELEMENT = 200
"""Rough per-row overhead of an HNSW graph (m=16) on top of the vector itself."""

# Doctor findings that matter for a migration, carried into the plan unchanged.
# Apply keeps the new column in sync with a trigger, so doctor's change-tracking findings
# don't apply; row-level security still matters.
CARRIED = {"access.row_security"}


@dataclass(frozen=True, slots=True)
class SampleStats:
    documents: int
    """Sampled rows that have text."""
    average_chars: float


def _bytes_per_vector(vector_type: str, dims: int) -> int:
    return dims * (2 if vector_type == "halfvec" else 4) + 8


def _metric(job: JobSpec, profile: IndexProfile) -> str:
    if job.target.metric:
        return job.target.metric.value
    for index in profile.ann_indexes:
        if index.metric in {"cosine", "inner_product", "l2"}:
            return index.metric
    return "cosine"


def _resolve_dimensions(
    spec: ModelSpec, probe: ProbeResult | None
) -> tuple[int | None, str | None]:
    if probe:
        return probe.dimensions, "probe"
    if spec.dimensions:
        return spec.dimensions, "spec"
    if spec.native_dimensions:
        return spec.native_dimensions, "known"
    return None, None


def _model_findings(spec: ModelSpec, findings: list[Finding]) -> None:
    native = spec.native_dimensions
    if (
        spec.dimensions
        and spec.provider == "openai"
        and spec.base_model
        not in {
            "text-embedding-3-small",
            "text-embedding-3-large",
        }
    ):
        findings.append(
            Finding(
                "plan.dims_unsupported",
                Severity.ERROR,
                "This model can't shorten its vectors",
                f"{spec.model} doesn't accept a dims= setting; only the text-embedding-3 "
                "models do.",
                hint="Remove dims= from the model spec.",
            )
        )
    elif spec.dimensions and native and spec.dimensions > native:
        findings.append(
            Finding(
                "plan.dims_too_large",
                Severity.ERROR,
                "dims= is larger than the model's output",
                f"{spec.model} produces {native:,} dimensions, so it can't return "
                f"{spec.dimensions:,}.",
            )
        )

    if spec.provider == "hash":
        findings.append(
            Finding(
                "plan.baseline_model",
                Severity.WARNING,
                "This is a test model",
                f"{spec.name} is vecshift's built-in hashing baseline: it matches words, not "
                "meaning. It's for trying vecshift out, not for real searches.",
                hint="Use a real embedding model for production data.",
            )
        )

    if spec.provider == "openai" and spec.key_env and not os.environ.get(spec.key_env):
        findings.append(
            Finding(
                "plan.missing_key",
                Severity.ERROR,
                "No API key for the model",
                f"{spec.key_env} isn't set, so apply couldn't call {spec.name}.",
                hint=f"Export {spec.key_env}, or add it to the .env file vecshift reads.",
            )
        )


def build_plan(
    job: JobSpec,
    profile: IndexProfile,
    target: TargetState,
    sample: SampleStats,
    probe: ProbeResult | None = None,
) -> Plan:
    spec = parse_spec(job.model)
    vector_type = job.target.vector_type.value
    dims, dims_source = _resolve_dimensions(spec, probe)
    metric = _metric(job, profile)
    findings: list[Finding] = []

    old_size = profile.declared_dimensions
    source_type = f"{profile.vector_type}({old_size})" if old_size else profile.vector_type
    rows_known = f", ~{profile.estimated_rows:,} rows" if profile.estimated_rows is not None else ""
    plan = Plan(
        job=job.name,
        source=f"{profile.target} {source_type}{rows_known}",
        target_column=job.target.column,
        model=spec.name,
        dimensions=dims,
        dimensions_source=dims_source,
        vector_type=vector_type,
        metric=metric,
    )

    # --- The source has to be re-embeddable and addressable.
    if profile.text_field is None:
        findings.append(
            Finding(
                "plan.no_text",
                Severity.ERROR,
                "No source text to embed",
                "No text column was found next to the vectors, so there's nothing to re-embed.",
                hint="Set source.text_column in the job file if the text has an unusual name.",
            )
        )
    elif (
        profile.sample.texts_present is not None
        and profile.sample.texts_present < profile.sample.rows
    ):
        missing = profile.sample.rows - profile.sample.texts_present
        findings.append(
            Finding(
                "plan.partial_text",
                Severity.WARNING,
                "Some rows have no text",
                f"{missing:,} of {profile.sample.rows:,} sampled rows have no text in "
                f"`{profile.text_field}`. Their new vector stays empty, so search won't find "
                "them after cutover.",
            )
        )
    if not profile.has_primary_key:
        findings.append(
            Finding(
                "plan.no_primary_key",
                Severity.ERROR,
                "No primary key",
                "Apply writes each new vector back to its row and resumes after "
                "interruptions, which needs a stable ID per row.",
                hint="Add a primary key to the table first.",
            )
        )

    if profile.has_primary_key and len(target.primary_key) > 1:
        findings.append(
            Finding(
                "plan.composite_primary_key",
                Severity.ERROR,
                "Composite primary key",
                f"The primary key spans {', '.join(target.primary_key)}. Apply writes each "
                "vector back by a single ID column.",
                hint="Add a single-column unique ID, or migrate a view-free copy of the table.",
            )
        )

    # --- The target column.
    if not target.owner:
        findings.append(
            Finding(
                "plan.not_owner",
                Severity.ERROR,
                "Can't add a column to this table",
                "Adding the new vector column needs the table's owner (or a member of the "
                "owning role), and the connecting role isn't.",
                hint="Connect as the table owner. On Supabase that's usually postgres.",
            )
        )
    if target.partitioned:
        findings.append(
            Finding(
                "plan.partitioned",
                Severity.ERROR,
                "Partitioned tables aren't supported yet",
                "PostgreSQL can't build an index concurrently on a partitioned table, and apply "
                "only builds indexes that way so writes never block.",
                hint="Migrate each partition as its own table for now.",
            )
        )
    if target.row_security:
        findings.append(
            Finding(
                "plan.row_security",
                Severity.ERROR,
                "Row-level security limits which rows apply can reach",
                "The connecting role is subject to the table's policies (the table forces "
                "row-level security, or the role isn't its owner), so apply could skip rows "
                "it can't see and still report success.",
                hint="Connect as a role with BYPASSRLS, or turn off FORCE ROW LEVEL SECURITY "
                "for the migration.",
            )
        )
    if target.previous_column_exists and not target.column_exists:
        findings.append(
            Finding(
                "plan.cut_over",
                Severity.ERROR,
                "This table was already cut over",
                f"`{target.source.column}_old` holds the vectors from before the last cutover.",
                hint="Run vecshift rollback to undo it, or vecshift cleanup when you're sure, "
                "before starting another migration.",
            )
        )
    if target.column_exists:
        same = target.column_type == vector_type and (
            dims is None or target.column_dimensions == dims
        )
        if same:
            findings.append(
                Finding(
                    "plan.column_exists",
                    Severity.INFO,
                    "Target column already exists",
                    f"`{job.target.column}` is already a matching `{vector_type}` column, so "
                    "apply will fill in the rows it's missing.",
                )
            )
        else:
            have = f"{target.column_type}({target.column_dimensions or '?'})"
            want = f"{vector_type}({dims or '?'})"
            findings.append(
                Finding(
                    "plan.column_conflict",
                    Severity.ERROR,
                    "Target column exists with a different type",
                    f"`{job.target.column}` is `{have}`, but this job needs `{want}`.",
                    hint="Pick another target.column, or drop the existing one if it's unused.",
                )
            )

    # --- The model.
    _model_findings(spec, findings)
    if dims is None:
        findings.append(
            Finding(
                "plan.dims_unknown",
                Severity.WARNING,
                "Vector size unknown",
                f"vecshift doesn't know how many dimensions {spec.name} produces, so it can't "
                "size the new column or estimate storage.",
                hint="Add dims= to the model spec, or run plan with --probe to measure it.",
            )
        )
    recorded = set(profile.sample.models)
    lowered = {m.lower() for m in recorded}
    tag_prefix = f"{spec.provider}/{spec.model}@".lower()
    same = spec.model in recorded or spec.base_model in lowered
    same = same or any(m.startswith(tag_prefix) for m in lowered)
    if len(recorded) == 1 and same:
        findings.append(
            Finding(
                "plan.same_model",
                Severity.WARNING,
                "The table already uses this model",
                f"Rows record `{next(iter(recorded))}` in `{profile.model_field}`, the same "
                "model this job would switch to.",
                hint="Check the model in the job file. Re-embedding with the same model "
                "only makes sense to change dims= or the prefixes.",
            )
        )

    limit = MAX_INDEXABLE_DIMENSIONS.get(vector_type)
    if dims and limit and dims > limit and job.target.index is not IndexMethod.NONE:
        halfvec_ok = vector_type == "vector" and dims <= MAX_INDEXABLE_DIMENSIONS["halfvec"]
        findings.append(
            Finding(
                "plan.too_large_to_index",
                Severity.ERROR,
                "Too many dimensions to index",
                f"pgvector can index `{vector_type}` columns up to {limit:,} dimensions, and "
                f"this model produces {dims:,}.",
                hint="Set target.vector_type: halfvec (up to 4,000 dimensions, half the storage)."
                if halfvec_ok
                else "Add dims= to the model spec to shorten the vectors, if the model allows.",
            )
        )

    # --- Doctor's view of change tracking and access.
    from vecshift.doctor.checks import run_checks

    for finding in run_checks(profile).findings:
        if finding.id in CARRIED and not target.row_security:
            findings.append(finding)

    # --- Estimates.
    est = Estimates(maintenance_work_mem=target.maintenance_work_mem)
    share = (
        (profile.sample.texts_present or 0) / profile.sample.rows if profile.sample.rows else 0.0
    )
    if profile.estimated_rows is not None:
        est.rows = round(profile.estimated_rows * share)
        est.rows_exact = profile.sample.method == "full table"
    if est.rows is not None:
        tokens_per_char = probe.tokens_per_char if probe else 1 / CHARS_PER_TOKEN
        est.tokens = round(est.rows * sample.average_chars * tokens_per_char)
        est.tokens_method = (
            f"measured on {probe.documents} documents"
            if probe
            else f"about {CHARS_PER_TOKEN:g} characters per token"
        )
        est.requests = math.ceil(est.rows / spec.batch_size) if est.rows else 0
        if spec.price is not None:
            est.cost_usd = est.tokens * spec.price / 1_000_000

        durations: list[tuple[float, str]] = []
        if job.limits.tokens_per_minute:
            durations.append(
                (60 * est.tokens / job.limits.tokens_per_minute, "your tokens-per-minute limit")
            )
        if job.limits.requests_per_minute:
            durations.append(
                (
                    60 * est.requests / job.limits.requests_per_minute,
                    "your requests-per-minute limit",
                )
            )
        if probe and probe.docs_per_second > 0:
            durations.append((est.rows / probe.docs_per_second, "speed measured by the probe"))
        if durations:
            est.seconds, est.seconds_method = max(durations)

        if dims:
            per_vector = _bytes_per_vector(vector_type, dims)
            est.new_bytes = est.rows * per_vector
            if job.target.index is IndexMethod.HNSW:
                est.index_memory_bytes = est.rows * (per_vector + HNSW_BYTES_PER_ELEMENT)

    old_dims = profile.declared_dimensions or (
        profile.sample.dimensions.most_common(1)[0][0] if profile.sample.dimensions else None
    )
    if old_dims and profile.estimated_rows is not None:
        with_vectors = profile.sample.vectors / profile.sample.rows if profile.sample.rows else 0
        est.old_bytes = round(
            profile.estimated_rows * with_vectors * _bytes_per_vector(profile.vector_type, old_dims)
        )
    plan.estimates = est

    if job.limits.budget_usd is not None and est.cost_usd is not None:
        if est.cost_usd > job.limits.budget_usd:
            findings.append(
                Finding(
                    "plan.over_budget",
                    Severity.ERROR,
                    "Estimated cost is over budget",
                    f"Embedding is estimated at ${est.cost_usd:,.2f}, above the "
                    f"${job.limits.budget_usd:,.2f} budget.",
                    hint="Raise limits.budget_usd, or use a cheaper model or fewer dims.",
                )
            )
    elif est.cost_usd and job.limits.budget_usd is None:
        findings.append(
            Finding(
                "plan.no_budget",
                Severity.INFO,
                "No budget set",
                "Without limits.budget_usd, nothing stops a run that costs more than expected.",
                hint=f"Add limits.budget_usd, for example {max(1, math.ceil(est.cost_usd * 1.5))}.",
            )
        )
    if spec.price is None:
        findings.append(
            Finding(
                "plan.cost_unknown",
                Severity.INFO,
                "Cost unknown",
                f"There's no price for {spec.name}, so the plan can't estimate cost or "
                "enforce a budget.",
                hint="Add price= (USD per million tokens) to the model spec.",
            )
        )
    if (
        est.index_memory_bytes
        and target.maintenance_work_mem
        and est.index_memory_bytes > target.maintenance_work_mem
    ):
        need_mb = math.ceil(est.index_memory_bytes * 1.25 / 1024**2 / 64) * 64
        need = f"{need_mb}MB" if need_mb < 1024 else f"{math.ceil(need_mb / 1024)}GB"
        findings.append(
            Finding(
                "plan.index_memory",
                Severity.WARNING,
                "The index build won't fit in memory",
                f"An HNSW index this size needs about {_size(est.index_memory_bytes)}, and "
                f"maintenance_work_mem is {_size(target.maintenance_work_mem)}. pgvector builds "
                "much more slowly once the graph no longer fits.",
                hint=f"Raise it for the build, e.g. SET maintenance_work_mem = '{need}', if "
                "the server has the memory to spare. In Docker, also give the container that "
                "much shared memory (--shm-size), which parallel builds use.",
            )
        )

    # --- What apply would do.
    if dims and not target.column_exists:
        plan.changes.append(
            Change(
                "add_column",
                f"add column {target.source.schema}.{target.source.table}."
                f"{job.target.column} {vector_type}({dims})",
                add_column_sql(target, job.target.column, vector_type, dims),
                "Instant for any table size: a nullable column without a default doesn't "
                "rewrite rows. It takes a brief exclusive lock.",
            )
        )
    if profile.text_field:
        plan.changes.append(
            Change(
                "trigger",
                f"add a trigger that clears {job.target.column} when {profile.text_field} changes",
                note="So rows edited during the migration get re-embedded. Cutover moves it to "
                "the old column, and cleanup removes it.",
            )
        )
    rows_text = f"~{est.rows:,}" if est.rows is not None else "all"
    plan.changes.append(
        Change(
            "embed",
            f"embed {rows_text} rows from {profile.text_field or '?'} with {spec.name}",
            note="In batches, resumable, without blocking reads or writes. Run apply again "
            "any time to catch up rows added or changed since.",
        )
    )
    if job.target.index is not IndexMethod.NONE and dims:
        plan.changes.append(
            Change(
                "index",
                f"build a {job.target.index.value} index ({metric}) on {job.target.column}",
                index_sql(
                    target,
                    job.target.column,
                    vector_type,
                    job.target.index.value,
                    metric,
                    est.rows or 0,
                ),
                "Built after the backfill and concurrently, so writes continue.",
            )
        )
    old = target.source.column
    plan.changes.append(
        Change(
            "cutover",
            f"later, with vecshift cutover: {old} → {old}_old, {job.target.column} → {old}",
            note="Renames in one transaction, so searches switch atomically and your SQL keeps "
            "working. Rollback renames them back; cleanup drops the old column.",
        )
    )

    plan.findings = findings
    return plan


def _size(n: int) -> str:
    for unit, size in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:,.1f} {unit}"
    return f"{n} B"
