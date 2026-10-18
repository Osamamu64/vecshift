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
