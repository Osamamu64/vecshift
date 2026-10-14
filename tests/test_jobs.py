from pathlib import Path

import pytest
import yaml

from vecshift.jobs import JobError, JobSpec, load_job, template


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "vecshift.yaml"
    path.write_text(text)
    return path


def test_template_round_trips() -> None:
    for model in ("openai/text-embedding-3-large,dims=1024", "ollama/nomic-embed-text:latest"):
        job = JobSpec.model_validate(yaml.safe_load(template("public.docs", model)))
        assert job.model == model
        assert job.pg_source.table == "public.docs"
        assert job.pg_target.column == "embedding_v2"
        assert job.name == "docs-reembed"
        assert job.limits.budget_usd is None


def test_minimal_job(tmp_path: Path) -> None:
    job = load_job(write(tmp_path, "source: {table: docs}\nmodel: ollama/nomic-embed-text\n"))
    assert job.pg_source.dsn_env == "VECSHIFT_DSN"
    assert job.pg_target.vector_type == "vector" and job.pg_target.index == "hnsw"


def test_empty_sections_mean_defaults(tmp_path: Path) -> None:
    text = "source:\n  table: docs\nmodel: ollama/nomic-embed-text\ntarget:\nlimits:\n"
    job = load_job(write(tmp_path, text))
    assert job.pg_target.column == "embedding_v2" and job.limits.budget_usd is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("model: ollama/x\n", "source: required"),
        ("source: {table: docs}\n", "model: required"),
        ("source: {table: docs}\nmodel: ollama/x\nsurprise: 1\n", "surprise: unknown setting"),
        ("source: {table: 'docs; drop'}\nmodel: ollama/x\n", "plain identifier"),
        ("source: {table: a.b.c}\nmodel: ollama/x\n", "schema.table"),
        ("source: {table: docs}\nmodel: acme/x\n", "Unknown provider"),
        ("source: {table: docs}\nmodel: ollama/x\nlimits: {budget_usd: -1}\n", "budget_usd"),
        ("source: {table: docs}\nmodel: ollama/x\ntarget: {index: btree}\n", "target.index"),
        ("- just\n- a list\n", "should contain settings"),
        ("source: [unclosed\n", "isn't valid YAML at line"),
    ],
)
def test_invalid_jobs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(JobError, match=message.replace("(", r"\(")):
        load_job(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(JobError, match="Couldn't read"):
        load_job(tmp_path / "nope.yaml")


# --- Qdrant


def test_qdrant_to_qdrant(tmp_path: Path) -> None:
    text = "source: {type: qdrant, collection: docs}\nmodel: ollama/x\n"
    job = load_job(write(tmp_path, text))
    assert job.source.type == "qdrant" and job.target.type == "qdrant"
    assert job.target_collection == "docs_v2"
    assert job.target_url_env == "VECSHIFT_QDRANT_URL" and job.target_key_env == "QDRANT_API_KEY"
    with pytest.raises(TypeError):
        _ = job.pg_source


def test_qdrant_target_inherits_connection(tmp_path: Path) -> None:
    text = (
        "source: {type: qdrant, collection: docs, url_env: SRC_URL, key_env: SRC_KEY}\n"
        "target: {collection: docs_2026, alias: live}\nmodel: ollama/x\n"
    )
    job = load_job(write(tmp_path, text))
    assert job.target_url_env == "SRC_URL" and job.target_key_env == "SRC_KEY"
    assert job.target_collection == "docs_2026" and job.qdrant_target.alias == "live"


def test_pgvector_to_qdrant(tmp_path: Path) -> None:
    text = "source: {table: public.articles}\ntarget: {type: qdrant}\nmodel: ollama/x\n"
    job = load_job(write(tmp_path, text))
    assert job.source.type == "pgvector" and job.target_collection == "articles_v2"
    assert job.target_url_env == "VECSHIFT_QDRANT_URL"
    text = "source: {table: articles}\ntarget: {type: qdrant, alias: art}\nmodel: ollama/x\n"
    assert load_job(write(tmp_path, text)).target_collection == "art_v2"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("source: {type: qdrant}\nmodel: ollama/x\n", "source.collection: required"),
        (
            "source: {type: qdrant, collection: 'a/b'}\nmodel: ollama/x\n",
            "source.collection: must be a Qdrant collection",
        ),
        (
            "source: {type: qdrant, collection: a, url_env: 'x y'}\nmodel: ollama/x\n",
            "environment variable",
        ),
        (
            "source: {type: qdrant, collection: a}\ntarget: {type: pgvector}\nmodel: ollama/x\n",
            "needs a Qdrant target",
        ),
        (
            "source: {type: qdrant, collection: a}\ntarget: {column: x}\nmodel: ollama/x\n",
            "target.column: unknown setting",
        ),
        ("source: {type: milvus, table: a}\nmodel: ollama/x\n", "source"),
        (
            "source: {table: a}\ntarget: {type: qdrant, index: ivfflat}\nmodel: ollama/x\n",
            "target.index",
        ),
    ],
)
def test_invalid_qdrant_jobs(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(JobError, match=message):
        load_job(write(tmp_path, text))
