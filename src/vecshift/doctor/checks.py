"""The diagnostic checks. Each takes an :class:`IndexProfile` and returns findings."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

from vecshift.doctor.findings import Finding, Report, Severity
from vecshift.doctor.profile import IndexProfile

NORM_TOLERANCE = 0.01
"""How far from 1.0 a vector's L2 norm may be and still count as normalized."""

ZERO_NORM = 1e-9
DUPLICATE_WARNING_RATIO = 0.01
UNINDEXED_ROWS_WARNING = 100_000
HISTOGRAM_BINS = 20

Check = Callable[[IndexProfile], list[Finding]]


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0%"
    value = 100 * part / whole
    return f"{value:.0f}%" if value >= 1 or value == 0 else "<1%"


def _verb(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _breakdown(counts: dict[int, int] | dict[str, int]) -> str:
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{k} ({v:,})" for k, v in items)


def check_access(p: IndexProfile) -> list[Finding]:
    if not p.rows_hidden_by_access_rules:
        return []
    return [
        Finding(
            "access.row_security",
            Severity.WARNING,
            "Row-level security may hide rows",
            "Row-level security is on for this table and the connecting role doesn't bypass "
            "it, so this report only covers the rows that role can see.",
            hint="Connect as the table owner, or a role with BYPASSRLS (on Supabase, the "
            "postgres user), for a complete picture.",
        )
    ]


def check_text(p: IndexProfile) -> list[Finding]:
    if p.text_field is None:
        return [
            Finding(
                "text.missing",
                Severity.ERROR,
                "No source text: this index can't be re-embedded from itself",
                "No text column was found next to the vectors. Re-embedding with a new model "
                "needs the original text.",
                hint="If the text is in a column with an unusual name, pass --text-column. If "
                "it lives elsewhere (another table, object storage), keep that source "
                "available for the migration.",
            )
        ]
    rows = p.sample.rows
    present = p.sample.texts_present or 0
    if present < rows:
        missing = rows - present
        return [
            Finding(
                "text.partial",
                Severity.WARNING,
                "Some rows have no text",
                f"{missing:,} of {rows:,} sampled rows ({_pct(missing, rows)}) "
                f"{_verb(missing, 'has', 'have')} empty or missing `{p.text_field}`, so "
                f"{_verb(missing, 'it', 'they')} can't be re-embedded.",
            )
        ]
    return [
        Finding(
            "text.ok",
            Severity.OK,
            "Source text available",
            f"Every sampled row has text in `{p.text_field}`, so the index can be re-embedded.",
        )
    ]


def check_null_vectors(p: IndexProfile) -> list[Finding]:
    s = p.sample
    if s.null_vectors == 0:
        return []
    return [
        Finding(
            "vectors.null",
            Severity.WARNING,
            "Some rows have no vector",
            f"{s.null_vectors:,} of {s.rows:,} sampled rows ({_pct(s.null_vectors, s.rows)}) "
            f"{_verb(s.null_vectors, 'has', 'have')} no embedding, so search can't find "
            f"{_verb(s.null_vectors, 'it', 'them')}.",
            hint="These are often rows whose embedding failed or hasn't run yet.",
        )
    ]


def check_dimensions(p: IndexProfile) -> list[Finding]:
    findings: list[Finding] = []
    dims = p.sample.dimensions
    if len(dims) > 1:
        findings.append(
            Finding(
                "dims.mixed",
                Severity.ERROR,
                "Vectors of different sizes in one index",
                f"Sampled vectors have {len(dims)} different sizes: {_breakdown(dict(dims))}. "
                "They come from different models and can't be compared with each other.",
                hint="Find which rows came from which model, and re-embed the minority "
                "with the majority model.",
            )
        )
    if p.declared_dimensions is None:
        findings.append(
            Finding(
                "dims.undeclared",
                Severity.WARNING,
                "Vector size isn't fixed by the schema",
                f"The field is declared as `{p.vector_type}` without a size, so nothing stops "
                "vectors of different sizes being written to it, and it can't be given an ANN "
                "index for fast search.",
                hint=f"Declare the size, for example `{p.vector_type}(1536)`, once every "
                "vector has the same size.",
            )
        )
    elif len(dims) <= 1:
        findings.append(
            Finding(
                "dims.ok",
                Severity.OK,
                "Consistent vector size",
                f"The schema fixes vectors at {p.declared_dimensions:,} dimensions.",
            )
        )

    size = p.declared_dimensions or (max(dims) if dims else None)
    limit = p.max_indexable_dimensions
    if size and limit and size > limit and not p.ann_indexes:
        findings.append(
            Finding(
                "dims.too_large_to_index",
                Severity.WARNING,
                "Too many dimensions for an ANN index",
                f"{p.vector_type} vectors with {size:,} dimensions are over the "
                f"{limit:,}-dimension limit for an index, so every search is an exact scan.",
                hint="Index a halfvec cast instead (up to 4,000 dimensions), or truncate a "
                "Matryoshka-trained model to fewer dimensions.",
            )
        )
    return findings


def check_norms(p: IndexProfile) -> list[Finding]:
    norms = p.sample.norms
    if not norms:
        return []
    findings: list[Finding] = []
    zero = sum(1 for n in norms if n < ZERO_NORM)
    if zero:
        findings.append(
            Finding(
                "norms.zero",
                Severity.ERROR,
                "Zero vectors",
                f"{zero:,} sampled {_verb(zero, 'vector is', 'vectors are')} all zeros. Cosine "
                "distance is undefined for zero vectors, so they never rank sensibly.",
                hint="Zero vectors usually mean a failed embedding call stored a placeholder. "
                "Re-embed those rows.",
            )
        )

    nonzero = [n for n in norms if n >= ZERO_NORM]
    if not nonzero:
        return findings
    unit = sum(1 for n in nonzero if abs(n - 1.0) <= NORM_TOLERANCE)
    metrics = {i.metric for i in p.ann_indexes}

    if 0 < unit < len(nonzero):
        findings.append(
            Finding(
                "norms.mixed",
                Severity.WARNING,
                "Mix of normalized and unnormalized vectors",
                f"{unit:,} of {len(nonzero):,} sampled vectors are unit length and the rest "
                "aren't. That usually means two different pipelines or models wrote to "
                "this index.",
            )
        )
    elif unit == len(nonzero):
        findings.append(
            Finding(
                "norms.ok",
                Severity.OK,
                "Vectors are normalized",
                "Every sampled vector has unit length, so cosine, inner product, and L2 "
                "distance all rank results the same way.",
            )
        )
    else:
        lo, hi = min(nonzero), max(nonzero)
        findings.append(
            Finding(
                "norms.unnormalized",
                Severity.INFO,
                "Vectors aren't normalized",
                f"Sampled vector lengths range from {lo:.3g} to {hi:.3g}. That's fine with "
                "cosine distance; inner product and L2 will favour longer vectors.",
            )
        )

    if unit < len(nonzero) and "inner_product" in metrics:
        findings.append(
            Finding(
                "norms.inner_product_unnormalized",
                Severity.WARNING,
                "Inner-product index over unnormalized vectors",
                "An inner-product index ranks longer vectors higher regardless of meaning "
                "when vectors aren't normalized.",
                hint="Normalize vectors before writing them, or switch the index to cosine.",
            )
        )
    return findings


def check_duplicates(p: IndexProfile) -> list[Finding]:
    s = p.sample
    findings: list[Finding] = []
    for kind, count, base, noun in (
        ("text", s.duplicate_texts, s.texts_present or 0, "texts"),
        ("vectors", s.duplicate_vectors, s.vectors, "vectors"),
    ):
        if base and count / base > DUPLICATE_WARNING_RATIO:
            findings.append(
                Finding(
                    f"duplicates.{kind}",
                    Severity.WARNING,
                    f"Duplicate {noun}",
                    f"{count:,} of {base:,} sampled rows ({_pct(count, base)}) "
                    f"{_verb(count, 'repeats', 'repeat')} a "
                    f"{noun[:-1]} that's already in the sample. Duplicates crowd out other "
                    "results and cost money to re-embed.",
                    hint="Deduplicate before migrating: you only pay to embed each text once.",
                )
            )
    return findings


def check_models(p: IndexProfile) -> list[Finding]:
    models = p.sample.models
    if not models:
        return [
            Finding(
                "models.untracked",
                Severity.INFO,
                "No record of which model made these vectors",
                "Nothing next to the vectors says which embedding model produced them, so "
                "a mixed or outdated index can't be detected directly.",
                hint="Store a model tag with each row, for example the output of "
                "`vecshift fingerprint`, in a column or in metadata under `model_tag`.",
            )
        ]
    findings: list[Finding] = []
    if len(models) > 1:
        findings.append(
            Finding(
                "models.mixed",
                Severity.ERROR,
                "Vectors from more than one model",
                f"`{p.model_field}` records {len(models)} different models: "
                f"{_breakdown(dict(models))}. Vectors from different models aren't "
                "comparable, so searches across them return poor results.",
                hint="Re-embed the rows from the older model(s) with the current one.",
            )
        )
    else:
        (model,) = models
        findings.append(
            Finding(
                "models.ok",
                Severity.OK,
                "One embedding model",
                f"Every sampled row that records a model in `{p.model_field}` says `{model}`.",
            )
        )
    recorded = sum(models.values())
    if recorded < p.sample.vectors:
        missing = p.sample.vectors - recorded
        findings.append(
            Finding(
                "models.partial",
                Severity.WARNING,
                "Some rows don't record their model",
                f"{missing:,} of {p.sample.vectors:,} sampled vectors "
                f"{_verb(missing, 'has', 'have')} nothing in `{p.model_field}`, so "
                f"{_verb(missing, 'its', 'their')} model is unknown.",
            )
        )
    return findings


def check_indexes(p: IndexProfile) -> list[Finding]:
    if p.ann_indexes:
        described = ", ".join(
            f"{i.name} ({i.method}{', ' + i.metric if i.metric else ''})" for i in p.ann_indexes
        )
        return [Finding("index.ok", Severity.OK, "ANN index present", described)]
    rows = p.estimated_rows or 0
    large = rows > UNINDEXED_ROWS_WARNING
    return [
        Finding(
            "index.missing",
            Severity.WARNING if large else Severity.INFO,
            "No ANN index",
            f"Searches scan every row exactly{f' (about {rows:,} rows)' if rows else ''}. "
            + (
                "That gets slow at this size."
                if large
                else "That's fine for small tables but slows down as the table grows."
            ),
            hint="Create an HNSW index with the operator class for your distance, for "
            "example `vector_cosine_ops`.",
        )
    ]


def check_sync(p: IndexProfile) -> list[Finding]:
    findings: list[Finding] = []
    if not p.has_primary_key:
        findings.append(
            Finding(
                "sync.no_primary_key",
                Severity.ERROR,
                "No primary key",
                "Without a stable ID per row, a migration can't resume safely, avoid "
                "duplicates, or match updates and deletes to the right row.",
                hint="Add a primary key before migrating.",
            )
        )
    if p.logical_replication and p.has_primary_key:
        findings.append(
            Finding(
                "sync.logical_replication",
                Severity.OK,
                "Live writes can be streamed",
                "Logical replication is enabled, so inserts, updates, and deletes made during "
                "a migration can be streamed to the new index.",
            )
        )
    elif p.updated_at_field:
        findings.append(
            Finding(
                "sync.updated_at",
                Severity.OK,
                "Live writes can be tracked by timestamp",
                f"`{p.updated_at_field}` lets a migration catch up on rows changed while it "
                "ran. Deletes don't show up this way, so they need a separate reconciliation "
                "pass by ID.",
            )
        )
    elif p.store == "pgvector" and p.has_primary_key:
        findings.append(
            Finding(
                "sync.trigger",
                Severity.OK,
                "vecshift keeps up with writes",
                "There's no logical replication or updated-at column, and none is needed: "
                "`vecshift apply` adds a trigger that clears a row's new vector when its text "
                "changes, so rows edited during a migration are re-embedded before cutover. "
                "Inserts are picked up the same way, and deleted rows simply drop out.",
            )
        )
    else:
        findings.append(
            Finding(
                "sync.none",
                Severity.WARNING,
                "No way to track writes during a migration",
                "There's no logical replication and no updated-at timestamp column, so "
                "changes made while a migration runs would be missed.",
                hint="Add an `updated_at` column maintained by a trigger, or plan to pause "
                "writes or dual-write during the migration.",
            )
        )
    return findings


CHECKS: tuple[Check, ...] = (
    check_access,
    check_text,
    check_null_vectors,
    check_dimensions,
    check_norms,
    check_duplicates,
    check_models,
    check_indexes,
    check_sync,
)

SAMPLE_CHECKS = {
    check_text,
    check_null_vectors,
    check_dimensions,
    check_norms,
    check_duplicates,
    check_models,
}


def _histogram(values: list[float], bins: int = HISTOGRAM_BINS) -> list[dict[str, float]]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [{"start": lo, "end": hi, "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / width), bins - 1)] += 1
    return [
        {"start": lo + i * width, "end": lo + (i + 1) * width, "count": c}
        for i, c in enumerate(counts)
    ]


def facts(profile: IndexProfile) -> dict[str, Any]:
    """The measurements behind the findings, in a JSON-friendly shape."""
    s = profile.sample
    nonzero = [n for n in s.norms if n >= ZERO_NORM]
    norms: dict[str, Any] | None = None
    if s.norms:
        norms = {
            "count": len(s.norms),
            "zero": len(s.norms) - len(nonzero),
            "unit": sum(1 for n in nonzero if abs(n - 1.0) <= NORM_TOLERANCE),
            "min": min(s.norms),
            "median": statistics.median(s.norms),
            "max": max(s.norms),
            "tolerance": NORM_TOLERANCE,
            "histogram": _histogram(s.norms),
        }
    return {
        "text_field": profile.text_field,
        "model_field": profile.model_field,
        "updated_at_field": profile.updated_at_field,
        "has_primary_key": profile.has_primary_key,
        "logical_replication": profile.logical_replication,
        "rows_hidden_by_access_rules": profile.rows_hidden_by_access_rules,
        "max_indexable_dimensions": profile.max_indexable_dimensions,
        "ann_indexes": [
            {"name": i.name, "method": i.method, "metric": i.metric} for i in profile.ann_indexes
        ],
        "sample": {
            "rows": s.rows,
            "vectors": s.vectors,
            "null_vectors": s.null_vectors,
            "texts_present": s.texts_present,
            "duplicate_texts": s.duplicate_texts,
            "duplicate_vectors": s.duplicate_vectors,
        },
        "dimensions": {str(d): n for d, n in s.dimensions.most_common()},
        "models": dict(s.models.most_common()),
        "norms": norms,
    }


def run_checks(profile: IndexProfile) -> Report:
    """Run every check against ``profile`` and collect the findings."""
    report = Report(
        store=profile.store,
        target=profile.target,
        vector_type=profile.vector_type,
        declared_dimensions=profile.declared_dimensions,
        estimated_rows=profile.estimated_rows,
        sample_rows=profile.sample.rows,
        sample_method=profile.sample.method,
        facts=facts(profile),
    )
    empty = profile.sample.rows == 0
    if empty:
        report.findings.append(
            Finding(
                "sample.empty",
                Severity.INFO,
                "No rows to inspect",
                "The table is empty (or no rows are visible), so only schema checks ran.",
            )
        )
    for check in CHECKS:
        if empty and check in SAMPLE_CHECKS and check is not check_dimensions:
            continue
        report.findings.extend(check(profile))
    return report
