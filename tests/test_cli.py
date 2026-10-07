from typer.testing import CliRunner

from vecshift import __version__
from vecshift.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"vecshift {__version__}"


def test_fingerprint_prints_model_tag() -> None:
    result = runner.invoke(
        app,
        [
            "fingerprint",
            "--provider",
            "openai",
            "--model",
            "text-embedding-3-small",
            "--dimensions",
            "1536",
        ],
    )
    assert result.exit_code == 0
    assert result.output.startswith("openai/text-embedding-3-small@1536#")


def test_fingerprint_rejects_bad_dimensions() -> None:
    result = runner.invoke(
        app, ["fingerprint", "--provider", "p", "--model", "m", "--dimensions", "0"]
    )
    assert result.exit_code != 0
