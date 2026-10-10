from collections import Counter
from typing import Any

import pytest

from tests.test_doctor_checks import make_profile
from vecshift.connectors.pgvector.inspect import VectorColumn
from vecshift.connectors.pgvector.target import TargetState
from vecshift.doctor import AnnIndex, Severity
from vecshift.jobs import JobSpec
from vecshift.planning import ProbeResult, SampleStats, build_plan

COLUMN = VectorColumn("public", "documents", "embedding", "vector", 1536, relid=1, attnum=3)
SAMPLE = SampleStats(documents=100, average_chars=400.0)
MB = 1024**2


def job(**overrides: Any) -> JobSpec:
    data: dict[str, Any] = {
        "source": {"table": "public.documents"},
        "model": "ollama/nomic-embed-text",
        "limits": {"budget_usd": 100},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return JobSpec.model_validate(data)


def target(**overrides: Any) -> TargetState:
    values: dict[str, Any] = {
        "source": COLUMN,
        "extension_schema": "extensions",
        "owner": True,
        "column_exists": False,
        "column_type": None,
        "column_dimensions": None,
        "maintenance_work_mem": 1024 * MB,
    }
    values.update(overrides)
    return TargetState(**values)


def plan_for(
    j: JobSpec | None = None,
    t: TargetState | None = None,
    probe: ProbeResult | None = None,
    **profile: Any,
) -> Any:
    defaults: dict[str, Any] = {
        "estimated_rows": 10_000,
        "rows": 100,
        "models": Counter({"text-embedding-ada-002": 100}),
    }
    defaults.update(profile)
    return build_plan(j or job(), make_profile(**defaults), t or target(), SAMPLE, probe)


def ids(p: Any) -> dict[str, Severity]:
    return {f.id: f.severity for f in p.findings}


def test_happy_path() -> None:
    p = plan_for()
    assert p.ok and p.dimensions == 768 and p.dimensions_source == "known"
    assert [c.kind for c in p.changes] == ["add_column", "trigger", "embed", "index", "cutover"]
    add, _, _, index, _ = p.changes
    assert add.sql == (
        "ALTER TABLE public.documents ADD COLUMN embedding_v2 extensions.vector(768);"
    )
    assert index.sql and "USING hnsw (embedding_v2 extensions.vector_cosine_ops)" in index.sql
    e = p.estimates
    assert e.rows == 10_000 and e.tokens == 1_000_000 and e.requests == 157
    assert e.cost_usd == 0 and e.new_bytes == 10_000 * (768 * 4 + 8)
    assert e.old_bytes == 10_000 * (3 * 4 + 8)  # the profile's vectors are 3-dimensional


def test_source_problems() -> None:
    found = ids(plan_for(text_field=None, texts_present=None, has_primary_key=False))
    assert found["plan.no_text"] is Severity.ERROR
    assert found["plan.no_primary_key"] is Severity.ERROR
    assert ids(plan_for(texts_present=90))["plan.partial_text"] is Severity.WARNING


def test_target_problems() -> None:
    assert ids(plan_for(t=target(owner=False)))["plan.not_owner"] is Severity.ERROR
    same = plan_for(t=target(column_exists=True, column_type="vector", column_dimensions=768))
    assert ids(same)["plan.column_exists"] is Severity.INFO
    assert "add_column" not in [c.kind for c in same.changes]
    clash = plan_for(t=target(column_exists=True, column_type="vector", column_dimensions=1536))
    assert ids(clash)["plan.column_conflict"] is Severity.ERROR


def test_model_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    found = ids(plan_for(job(model="openai/text-embedding-ada-002,dims=512")))
    assert found["plan.dims_unsupported"] is Severity.ERROR
    assert found["plan.missing_key"] is Severity.ERROR
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    found = ids(plan_for(job(model="openai/text-embedding-3-small,dims=4096")))
    assert found["plan.dims_too_large"] is Severity.ERROR
    assert "plan.missing_key" not in found
    unknown = plan_for(job(model="compat/mystery,url=http://localhost:1/v1,price=0"))
    assert ids(unknown)["plan.dims_unknown"] is Severity.WARNING
    assert unknown.dimensions is None and unknown.estimates.new_bytes is None


@pytest.mark.parametrize(
    "recorded",
    ["nomic-embed-text", "ollama/nomic-embed-text@768#abc123"],
)
def test_same_model(recorded: str) -> None:
    found = ids(plan_for(models=Counter({recorded: 100})))
    assert found["plan.same_model"] is Severity.WARNING


def test_index_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    big = job(model="openai/text-embedding-3-large")
    p = plan_for(big)
    finding = next(f for f in p.findings if f.id == "plan.too_large_to_index")
    assert finding.severity is Severity.ERROR and finding.hint and "halfvec" in finding.hint
    assert plan_for(
        job(model="openai/text-embedding-3-large", target={"vector_type": "halfvec"})
    ).ok
    assert plan_for(job(model="openai/text-embedding-3-large", target={"index": "none"})).ok


def test_budget_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    pricey = job(model="openai/text-embedding-3-small", limits={"budget_usd": 0.001})
    p = plan_for(pricey)
    assert p.estimates.cost_usd == pytest.approx(1_000_000 * 0.02 / 1e6)
    assert ids(p)["plan.over_budget"] is Severity.ERROR
    no_budget = job(model="openai/text-embedding-3-small", limits=None)
    assert ids(plan_for(no_budget))["plan.no_budget"] is Severity.INFO
    unpriced = job(model="compat/m,url=http://localhost:1/v1,dims=8")
    assert ids(plan_for(unpriced))["plan.cost_unknown"] is Severity.INFO


def test_index_memory() -> None:
    p = plan_for(t=target(maintenance_work_mem=8 * MB))
    finding = next(f for f in p.findings if f.id == "plan.index_memory")
    assert finding.hint and "SET maintenance_work_mem = '64MB'" in finding.hint
    assert "plan.index_memory" not in ids(
        plan_for(job(target={"index": "ivfflat"}), t=target(maintenance_work_mem=8 * MB))
    )


def test_durations_take_the_slowest_limit() -> None:
    limited = job(limits={"tokens_per_minute": 100_000, "requests_per_minute": 1_000})
    p = plan_for(limited)
    assert p.estimates.seconds == pytest.approx(600)  # 1M tokens at 100K/min beats 157 requests
    probe = ProbeResult(documents=16, dimensions=768, tokens_per_char=0.5, docs_per_second=5)
    p = plan_for(limited, probe=probe)
    assert p.estimates.tokens == 2_000_000 and p.dimensions_source == "probe"
    assert p.estimates.seconds == pytest.approx(2000) and "probe" in (
        p.estimates.seconds_method or ""
    )
    assert plan_for().estimates.seconds is None


def test_metric_and_ivfflat() -> None:
    ip = plan_for(
        job(target={"index": "ivfflat"}), ann_indexes=[AnnIndex("old", "hnsw", "inner_product")]
    )
    index = next(c for c in ip.changes if c.kind == "index")
    assert index.sql and "vector_ip_ops" in index.sql and "WITH (lists = 10)" in index.sql
    explicit = plan_for(job(target={"metric": "l2"}), ann_indexes=[AnnIndex("o", "hnsw", "cosine")])
    assert "vector_l2_ops" in (next(c for c in explicit.changes if c.kind == "index").sql or "")


def test_no_updated_at_needed() -> None:
    """The sync trigger tracks changes, so a table without updated_at is fine to migrate."""
    p = plan_for(updated_at_field=None, logical_replication=False)
    assert p.ok and not any(f.id.startswith("sync.") for f in p.findings)


def test_composite_primary_key() -> None:
    found = ids(plan_for(t=target(primary_key=("tenant_id", "id"))))
    assert found["plan.composite_primary_key"] is Severity.ERROR
    assert "plan.composite_primary_key" not in ids(plan_for(t=target(primary_key=("id",))))


def test_json_shape() -> None:
    data = plan_for().to_dict()
    assert data["ok"] is True and data["dimensions"] == 768
    assert {c["kind"] for c in data["changes"]} == {
        "add_column",
        "trigger",
        "embed",
        "index",
        "cutover",
    }
    assert data["estimates"]["rows"] == 10_000


def test_hash_model_is_allowed_with_a_warning() -> None:
    p = plan_for(job(model="hash/256"))
    assert ids(p)["plan.baseline_model"] is Severity.WARNING
    assert p.ok and p.dimensions == 256 and p.estimates.cost_usd == 0
