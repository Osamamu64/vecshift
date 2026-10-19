"""The service layer on its own: errors with codes, and how the CLI shows them."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from vecshift.cli_style import reports_errors
from vecshift.service import Declined, EvalOptions, ServiceError, read_job, run_eval, status

PG_JOB = """\
name: docs
source: {type: pgvector, table: public.docs}
target: {type: pgvector, column: embedding_v2}
model: hash/8
"""
QDRANT_JOB = """\
name: docs
source: {type: qdrant, collection: docs}
target: {type: qdrant}
model: hash/8
"""


@pytest.fixture
def bare(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("VECSHIFT_DSN", "DATABASE_URL", "VECSHIFT_QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_job_problems(tmp_path: Path) -> None:
    with pytest.raises(ServiceError) as missing:
        read_job(tmp_path / "none.yaml")
    assert missing.value.code == "job.missing" and "vecshift init" in (missing.value.hint or "")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [", encoding="utf-8")
    with pytest.raises(ServiceError) as invalid:
        read_job(bad)
    assert invalid.value.code == "job.invalid"


@pytest.mark.parametrize(
    ("job", "variable"), [(PG_JOB, "VECSHIFT_DSN"), (QDRANT_JOB, "VECSHIFT_QDRANT_URL")]
)
def test_missing_settings_name_the_variable(
    tmp_path: Path, bare: None, job: str, variable: str
) -> None:
    path = tmp_path / "j.yaml"
    path.write_text(job, encoding="utf-8")
    with pytest.raises(ServiceError) as exc:
        status(path)
    assert exc.value.code == "env.missing" and exc.value.message == f"{variable} isn't set."


def test_options_are_checked_before_connecting(tmp_path: Path, bare: None) -> None:
    path = tmp_path / "j.yaml"
    path.write_text(PG_JOB, encoding="utf-8")
    with pytest.raises(ServiceError) as exc:
        run_eval(path, EvalOptions(cross_language=True))
    assert exc.value.code == "options.invalid"


def test_the_cli_shows_service_errors_as_before() -> None:
    app = typer.Typer()

    @app.command()
    @reports_errors
    def go(what: str) -> None:
        if what == "fail":
            raise ServiceError("It broke.", "Fix it.", "x.broken")
        raise Declined()

    failed = CliRunner().invoke(app, ["fail"])
    assert failed.exit_code == 2
    assert failed.output == "It broke.\n→ Fix it.\n"
    declined = CliRunner().invoke(app, ["no"])
    assert declined.exit_code == 1 and declined.output == ""


def bench_documents() -> tuple[tuple[str, str], ...]:
    sentences = [
        "Document {i} explains how the vector index is rebuilt after a model change.",
        "Its second part covers topic {t} and the budget that the migration needs.",
        "The last sentence of document {i} compares recall before and after cutover.",
    ]
    return tuple(
        (str(i), " ".join(s.format(i=i, t=i % 5) for s in sentences)) for i in range(1, 31)
    )


def test_bench_asks_before_text_leaves_the_machine() -> None:
    from vecshift.service.bench import BenchNotice, BenchOptions, Documents, run_bench

    source = Documents(bench_documents())
    options = BenchOptions(models=("hash/8", "openai/text-embedding-3-small"), num_queries=5)
    with pytest.raises(ServiceError) as caught:
        run_bench(source, options)
    assert caught.value.code == "consent.required"
    assert "https://api.openai.com" in (caught.value.hint or "")

    seen: list[BenchNotice] = []

    def no(notice: BenchNotice) -> bool:
        seen.append(notice)
        return False

    with pytest.raises(Declined):
        run_bench(source, options, confirm=no)
    assert [name for name, _ in seen[0].remote] == ["openai/text-embedding-3-small"]
    assert seen[0].to_dict()["models"][0]["where"] is None, "hash/8 runs here"


def test_bench_on_given_documents() -> None:
    from vecshift.service.bench import BenchOptions, Documents, run_bench

    steps: list[str] = []
    result = run_bench(
        Documents(bench_documents()),
        BenchOptions(models=("hash/8",), num_queries=5, cache=False),
        on_step=steps.append,
    )
    assert result.documents == 30 and result.query_source == "proxy"
    assert steps == ["Embedding with hash/8…"]
    with pytest.raises(ServiceError) as caught:
        run_bench(Documents((("1", "a"), ("1", "b"))), BenchOptions())
    assert caught.value.code == "documents.invalid"
    with pytest.raises(ServiceError) as caught:
        run_bench(None, BenchOptions())
    assert caught.value.code == "options.invalid"
