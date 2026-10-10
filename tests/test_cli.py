import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vecshift import __version__
from vecshift.cli import app

runner = CliRunner()


def plain(text: str) -> str:
    """Output without colour codes, which typer adds when it thinks it's on a terminal (CI)."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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


def test_bare_command_shows_help_without_decoration() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage: vecshift" in plain(result.output) and "status" in plain(result.output)
    assert "●" not in result.output, "no banner off a terminal"


def test_env_file_supplies_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECSHIFT_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env = tmp_path / "settings.env"
    env.write_text("VECSHIFT_DSN='not a dsn but secret-value'\n")
    env.chmod(0o600)
    result = runner.invoke(app, ["--env-file", str(env), "doctor"])
    assert result.exit_code == 2
    assert "doesn't look like a valid PostgreSQL connection string" in result.output
    assert "secret-value" not in result.output

    monkeypatch.delenv("VECSHIFT_DSN", raising=False)
    ignored = runner.invoke(app, ["--env-file", str(env), "--no-env-file", "doctor"])
    assert ignored.exit_code == 2 and "Missing option" in plain(ignored.output)


def test_env_file_problems_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VECSHIFT_DSN", raising=False)
    env = tmp_path / ".env"
    env.write_text("VECSHIFT_DSN='unclosed secret-value\n")
    env.chmod(0o600)
    broken = runner.invoke(app, ["--env-file", str(env), "doctor"])
    assert broken.exit_code == 2 and "line 1 has an unclosed" in broken.output
    assert "secret-value" not in broken.output

    env.write_text("VECSHIFT_TEST_ONLY=1\n")
    env.chmod(0o644)
    monkeypatch.delenv("VECSHIFT_TEST_ONLY", raising=False)
    loose = runner.invoke(app, ["--env-file", str(env), "fingerprint", "--help"])
    assert loose.exit_code == 0 and f"chmod 600 {env}" in loose.output


def test_init_without_a_terminal_needs_flags(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "-o", str(tmp_path / "j.yaml")])
    assert result.exit_code == 2
    assert "needs --table and --model" in result.output
    assert not (tmp_path / "j.yaml").exists()
