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
        assert job.source.table == "public.docs"
        assert job.target.column == "embedding_v2"
        assert job.name == "docs-reembed"
        assert job.limits.budget_usd is None


def test_minimal_job(tmp_path: Path) -> None:
    job = load_job(write(tmp_path, "source: {table: docs}\nmodel: ollama/nomic-embed-text\n"))
    assert job.source.dsn_env == "VECSHIFT_DSN"
    assert job.target.vector_type == "vector" and job.target.index == "hnsw"


def test_empty_sections_mean_defaults(tmp_path: Path) -> None:
    text = "source:\n  table: docs\nmodel: ollama/nomic-embed-text\ntarget:\nlimits:\n"
    job = load_job(write(tmp_path, text))
    assert job.target.column == "embedding_v2" and job.limits.budget_usd is None


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
