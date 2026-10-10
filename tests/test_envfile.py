"""Reading and writing .env files."""

import os
import stat
from pathlib import Path

import pytest

from vecshift import envfile


def test_parses_common_forms() -> None:
    text = """
# a comment
PLAIN=value
export EXPORTED=yes
SPACED = padded  # trailing comment
SINGLE='it has # and $HOME'
DOUBLE="line\\nbreak and \\"quotes\\""
EMPTY=
URL=postgresql://u:p%40ss@host:5432/db?sslmode=require
"""
    assert envfile.parse(text) == {
        "PLAIN": "value",
        "EXPORTED": "yes",
        "SPACED": "padded",
        "SINGLE": "it has # and $HOME",
        "DOUBLE": 'line\nbreak and "quotes"',
        "EMPTY": "",
        "URL": "postgresql://u:p%40ss@host:5432/db?sslmode=require",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not a setting", "line 1 isn't NAME=value"),
        ("A=1\nB='secret-value", "line 2 has an unclosed '"),
        ('C="secret-value" extra', "line 1 has text after"),
    ],
)
def test_errors_name_the_line_never_the_value(text: str, message: str) -> None:
    with pytest.raises(envfile.EnvFileError, match=message) as caught:
        envfile.parse(text)
    assert "secret-value" not in str(caught.value)


def test_load_never_overrides_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VECSHIFT_DSN", "from-shell")
    monkeypatch.delenv("VECSHIFT_TEST_ONLY", raising=False)
    path = tmp_path / ".env"
    path.write_text("VECSHIFT_DSN=from-file\nVECSHIFT_TEST_ONLY=1\n")
    assert envfile.load(path) == ["VECSHIFT_TEST_ONLY"]
    assert os.environ["VECSHIFT_DSN"] == "from-shell"
    assert os.environ["VECSHIFT_TEST_ONLY"] == "1"
    assert envfile.load(tmp_path / "missing") == [], "the file is optional"


@pytest.mark.parametrize(
    "value",
    ["simple", "it's quoted", "back\\slash $dollar \"dq\" 'sq'", "two\nlines", "p@ss#word"],
)
def test_saved_values_read_back_exactly(tmp_path: Path, value: str) -> None:
    path = tmp_path / ".env"
    envfile.save(path, "VALUE", value)
    assert envfile.parse(path.read_text()) == {"VALUE": value}


def test_save_replaces_one_line_and_keeps_the_rest_private(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# keep me\nOTHER=1\nVECSHIFT_DSN=old\n")
    path.chmod(0o644)
    envfile.save(path, "VECSHIFT_DSN", "new")
    assert path.read_text() == "# keep me\nOTHER=1\nVECSHIFT_DSN='new'\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not envfile.readable_by_others(path)
    path.chmod(0o644)
    assert envfile.readable_by_others(path)
    with pytest.raises(ValueError, match="not a variable name"):
        envfile.save(path, "BAD NAME", "x")
