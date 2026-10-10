from collections import Counter
from typing import Any

import pytest

from vecshift.doctor import AnnIndex, IndexProfile, Sample, Severity, run_checks


def make_profile(**overrides: Any) -> IndexProfile:
    """A healthy profile; tests break one thing at a time."""
    sample_fields = {
        k: overrides.pop(k)
        for k in list(overrides)
        if k in Sample.__dataclass_fields__ and k != "rows"
    }
    rows = overrides.pop("rows", 100)
    sample = Sample(
        rows=rows,
        method="full table",
        dimensions=Counter({3: rows}),
        norms=[1.0] * rows,
        texts_present=rows,
        models=Counter({"openai/text-embedding-3-small@1536#abc": rows}),
    )
    for k, v in sample_fields.items():
        setattr(sample, k, v)
    defaults: dict[str, Any] = {
        "store": "pgvector",
        "target": "public.documents.embedding",
        "vector_type": "vector",
        "declared_dimensions": 3,
        "estimated_rows": rows,
        "sample": sample,
        "text_field": "content",
        "model_field": "metadata->>'model_tag'",
        "updated_at_field": "updated_at",
        "has_primary_key": True,
        "ann_indexes": [AnnIndex("documents_embedding_idx", "hnsw", "cosine")],
        "max_indexable_dimensions": 2000,
    }
    defaults.update(overrides)
    return IndexProfile(**defaults)


def ids(profile: IndexProfile) -> dict[str, Severity]:
    return {f.id: f.severity for f in run_checks(profile).findings}


def test_healthy_profile_is_all_ok() -> None:
    report = run_checks(make_profile())
    assert report.worst is Severity.OK
    assert set(ids(make_profile())) == {
        "text.ok",
        "dims.ok",
        "norms.ok",
        "models.ok",
        "index.ok",
        "sync.updated_at",
    }


def test_missing_text_is_an_error() -> None:
    assert ids(make_profile(text_field=None, texts_present=None))["text.missing"] is Severity.ERROR


def test_partial_text_is_a_warning() -> None:
    assert ids(make_profile(texts_present=90))["text.partial"] is Severity.WARNING


def test_null_vectors() -> None:
    found = ids(make_profile(null_vectors=5))
    assert found["vectors.null"] is Severity.WARNING


def test_mixed_dimensions() -> None:
    profile = make_profile(declared_dimensions=None, dimensions=Counter({1536: 90, 3072: 10}))
    found = ids(profile)
    assert found["dims.mixed"] is Severity.ERROR
    assert found["dims.undeclared"] is Severity.WARNING
    assert "dims.ok" not in found


def test_too_many_dimensions_to_index_only_without_an_index() -> None:
    big = {"declared_dimensions": 3072, "dimensions": Counter({3072: 100})}
    assert "dims.too_large_to_index" not in ids(make_profile(**big))
    found = ids(make_profile(ann_indexes=[], **big))
    assert found["dims.too_large_to_index"] is Severity.WARNING


def test_zero_vectors() -> None:
    found = ids(make_profile(norms=[1.0] * 99 + [0.0]))
    assert found["norms.zero"] is Severity.ERROR
    assert found["norms.ok"] is Severity.OK


def test_mixed_norms() -> None:
    assert ids(make_profile(norms=[1.0] * 50 + [3.0] * 50))["norms.mixed"] is Severity.WARNING


def test_unnormalized_is_info_unless_indexed_by_inner_product() -> None:
    norms = [2.0, 5.0] * 50
    assert ids(make_profile(norms=norms))["norms.unnormalized"] is Severity.INFO
    ip = [AnnIndex("idx", "hnsw", "inner_product")]
    found = ids(make_profile(norms=norms, ann_indexes=ip))
    assert found["norms.inner_product_unnormalized"] is Severity.WARNING


def test_normalized_vectors_with_inner_product_are_fine() -> None:
    ip = [AnnIndex("idx", "hnsw", "inner_product")]
    assert "norms.inner_product_unnormalized" not in ids(make_profile(ann_indexes=ip))


@pytest.mark.parametrize(("dupes", "expected"), [(1, False), (2, True)])
def test_duplicate_threshold(dupes: int, expected: bool) -> None:
    found = ids(make_profile(duplicate_texts=dupes, duplicate_vectors=dupes))
    assert ("duplicates.text" in found) is expected
    assert ("duplicates.vectors" in found) is expected


def test_mixed_models() -> None:
    found = ids(make_profile(models=Counter({"ada-002": 60, "3-small": 40})))
    assert found["models.mixed"] is Severity.ERROR


def test_untracked_and_partially_tracked_models() -> None:
    assert ids(make_profile(models=Counter()))["models.untracked"] is Severity.INFO
    found = ids(make_profile(models=Counter({"m": 80})))
    assert found["models.partial"] is Severity.WARNING


@pytest.mark.parametrize(
    ("rows", "severity"), [(1_000, Severity.INFO), (500_000, Severity.WARNING)]
)
def test_missing_index_severity_grows_with_size(rows: int, severity: Severity) -> None:
    found = ids(make_profile(ann_indexes=[], estimated_rows=rows))
    assert found["index.missing"] is severity


def test_sync_strategies() -> None:
    assert "sync.logical_replication" in ids(make_profile(logical_replication=True))
    assert "sync.updated_at" in ids(make_profile())
    other = ids(make_profile(store="qdrant", updated_at_field=None))
    assert other["sync.none"] is Severity.WARNING, "stores vecshift has no trigger for"
    # pgvector needs neither: vecshift's own trigger tracks edits during apply.
    pg = ids(make_profile(store="pgvector", updated_at_field=None))
    assert pg["sync.trigger"] is Severity.OK and "sync.none" not in pg
    no_key = ids(make_profile(store="pgvector", updated_at_field=None, has_primary_key=False))
    assert "sync.none" in no_key and "sync.trigger" not in no_key
    found = ids(make_profile(has_primary_key=False, logical_replication=True))
    assert found["sync.no_primary_key"] is Severity.ERROR
    assert "sync.logical_replication" not in found


def test_row_security_warning() -> None:
    found = ids(make_profile(rows_hidden_by_access_rules=True))
    assert found["access.row_security"] is Severity.WARNING


def test_empty_table_skips_sample_checks() -> None:
    profile = make_profile(
        rows=0, dimensions=Counter(), norms=[], texts_present=0, models=Counter()
    )
    found = ids(profile)
    assert found["sample.empty"] is Severity.INFO
    assert not any(k.startswith(("text.", "norms.", "models.")) for k in found)
    assert "index.ok" in found


def test_report_sorting_and_json() -> None:
    report = run_checks(make_profile(text_field=None, texts_present=None, updated_at_field=None))
    severities = [f.severity.rank for f in report.sorted_findings()]
    assert severities == sorted(severities, reverse=True)
    data = report.to_dict()
    assert data["summary"]["error"] == 1
    assert data["findings"][0]["id"] == "text.missing"
