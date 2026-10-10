"""The terminal UI's plain fallbacks, used off a terminal (pipes, CI, tests)."""

import pytest
from rich.console import Console
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


# --- the styled renderers, which only run at a colour terminal


@pytest.fixture
def styled(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Pretend to be a colour terminal, recording what's printed."""
    console = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
    monkeypatch.setattr(ui, "fancy", lambda: True)
    monkeypatch.setattr(ui, "console", console)
    return console


def test_styled_plan_and_doctor(styled: Console) -> None:
    from tests.test_doctor_checks import make_profile
    from tests.test_planner import plan_for
    from vecshift.cli import _render as render_doctor
    from vecshift.cli_plan import _render as render_plan
    from vecshift.doctor import run_checks

    render_plan(plan_for())
    render_doctor(run_checks(make_profile()), "PostgreSQL")
    text = styled.export_text()
    assert "plan" in text and "Changes" in text and "ALTER TABLE" in text
    assert "Ready to apply." in text and "Next" in text
    assert "Findings" in text and "errors" in text


def test_styled_eval_bench_and_apply(styled: Console) -> None:
    import asyncio

    from tests.test_bench import make_docs
    from tests.test_eval import run_fake
    from vecshift.bench import Benchmark, corpus, run
    from vecshift.cli_apply import _summary
    from vecshift.cli_bench import _render as render_bench
    from vecshift.cli_eval import _render as render_eval
    from vecshift.embeddings import parse_spec
    from vecshift.migrate import ApplyResult

    render_eval(run_fake(), "docs")
    edited, queries, notes = corpus.proxy_queries(make_docs(), 10, seed=2)
    result = asyncio.run(
        run(Benchmark(edited, queries, "proxy", notes), [parse_spec("hash/32")], None, "t")
    )
    render_bench(result)
    _summary(ApplyResult("complete", rows_written=10, tokens=40, seconds=1.0))
    _summary(ApplyResult("stopped", rows_written=5, message="Stopped by you."))
    text = styled.export_text()
    assert "Search quality" in text and "Verdict: GO" in text and "ef_search=40" in text
    assert "Recall@10" in text and "hash/32" in text
    assert "Apply complete." in text and "Apply stopped." in text and "Rows written" in text


def test_styled_status(styled: Console) -> None:
    from vecshift.cli_status import Status, _render_rich

    for stage, missing in (("filling", 30), ("ready", 0), ("cut_over", 0), ("conflict", 0)):
        _render_rich(
            Status(
                stage=stage,
                table="public.docs",
                live="embedding",
                new="embedding_v2",
                previous="embedding_old",
                rows=100,
                missing=missing,
                index="hnsw",
                trigger=True,
                runs=1,
                rows_written=70,
                spent_usd=0.5,
                failed=0,
                last_event=None,
                next="do the next thing",
            ),
            "docs",
        )
    text = styled.export_text()
    assert "70.0%" in text and "Ready to cut over" in text and "Needs attention" in text
    assert "do the next thing" in text
