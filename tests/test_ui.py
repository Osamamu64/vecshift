"""The terminal UI's plain fallbacks, used off a terminal (pipes, CI, tests)."""

from typer.testing import CliRunner

from vecshift import ui

runner = CliRunner()


def test_off_a_terminal_there_is_no_decoration() -> None:
    with runner.isolation() as io:
        assert not ui.fancy()
        ui.banner("9.9.9", "tagline")
        ui.step(1, 3, "Database")
        ui.success("done")
        printed = io[0].getvalue().decode()
    assert "█" not in printed and "9.9.9" not in printed, "no banner"
    assert "Database" in printed and "✔ done" in printed


def test_select_falls_back_to_numbers() -> None:
    with runner.isolation(input="7\n2\n") as io:
        picked = ui.select("Pick one", [("first", "hint a"), ("second", "")])
        printed = io[0].getvalue().decode()
    assert picked == 1
    assert "1) first  hint a" in printed and "Pick a number from 1 to 2" in printed


def test_select_default() -> None:
    with runner.isolation(input="\n"):
        assert ui.select("Pick", [("a", ""), ("b", ""), ("c", "")], default=2) == 2


def test_text_asks_again_until_valid() -> None:
    def check(value: str) -> str | None:
        return None if value.isdigit() else "digits only"

    with runner.isolation(input="abc\n42\n") as io:
        assert ui.text("Number", validate=check) == "42"
        assert "digits only" in io[0].getvalue().decode()


def test_secret_is_not_echoed() -> None:
    with runner.isolation(input="s3cret-value\n") as io:
        assert ui.secret("Key") == "s3cret-value"
        assert "s3cret-value" not in io[0].getvalue().decode()
    with runner.isolation(input="\n"):
        assert ui.secret("Key", allow_empty=True) == ""


def test_confirm() -> None:
    with runner.isolation(input="\n"):
        assert ui.confirm("Go?", default=False) is False
    with runner.isolation(input="y\n"):
        assert ui.confirm("Go?", default=False) is True
