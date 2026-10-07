import re
from collections import Counter
from datetime import UTC, datetime

import pytest

from tests.test_doctor_checks import make_profile
from vecshift.doctor import run_checks
from vecshift.doctor.html import MAX_BARS, compact, percent, render_html

WHEN = datetime(2026, 10, 7, 22, 40, tzinfo=UTC)


def page_for(**overrides: object) -> str:
    report = run_checks(make_profile(**overrides))
    return render_html(
        report,
        connection="postgresql://postgres@db.example.supabase.co:5432/postgres",
        connection_kind="Supabase direct",
        version="9.9.9",
        generated_at=WHEN,
    )


def test_is_a_complete_self_contained_document() -> None:
    page = page_for()
    assert page.startswith("<!doctype html>")
    assert "<style>" in page and "<script>" in page
    # The only URL-bearing attribute is the inline favicon; nothing points off the page.
    assert re.findall(r'\b(?:src|href)="([a-z]+):', page) == ["data"]
    assert not re.search(r"<(img|iframe|script src)\b", page)
    assert "https://" not in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")
    assert "7 Oct 2026, 22:40 UTC" in page
    assert "vecshift 9.9.9" in page


@pytest.mark.parametrize(
    ("overrides", "title"),
    [
        ({}, "Ready to migrate"),
        ({"updated_at_field": None}, "Ready to migrate, with caveats"),
        ({"text_field": None, "texts_present": None}, "Not ready to migrate"),
        (
            {
                "rows": 0,
                "dimensions": Counter(),
                "norms": [],
                "texts_present": 0,
                "models": Counter(),
            },
            "Nothing to inspect",
        ),
    ],
)
def test_verdict(overrides: dict[str, object], title: str) -> None:
    assert f'id="verdict-title">{title}</h2>' in page_for(**overrides)


def test_untrusted_names_are_escaped() -> None:
    evil = "<script>alert(1)</script>"
    page = page_for(
        target=f"public.{evil}.embedding",
        models=Counter({evil: 60, "b": 40}),
        text_field=evil,
    )
    assert evil not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_backticks_become_code() -> None:
    page = page_for(texts_present=90)
    assert "<code>content</code>" in page
    assert "`content`" not in page


def test_histogram_needs_spread() -> None:
    flat = page_for()
    assert 'class="bin"' not in flat
    assert "All 100 vectors have length 1.00" in flat

    spread = page_for(norms=[0.5 + i / 50 for i in range(100)])
    assert spread.count('class="bin"') == 20
    assert "1.0 = unit length" in spread
    assert "View as table" in spread


def test_breakdown_bars_and_other_bucket() -> None:
    many = Counter({f"model-{i}": 100 - i for i in range(10)})
    page = page_for(models=many)
    assert page.count('class="bar-row"') == MAX_BARS
    assert "Other (4)" in page

    single = page_for()
    assert 'class="bar-row"' not in single
    assert "The only model named by the 100 sampled vectors" in single


def test_tiles_reflect_problems() -> None:
    page = page_for(declared_dimensions=None, dimensions=Counter({384: 90, 768: 10}))
    assert '<div class="tile-value">Mixed</div>' in page
    assert '<div class="tile-value">None</div>' in page_for(text_field=None, texts_present=None)


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0"),
        (999, "999"),
        (9_999, "9,999"),
        (12_000, "12K"),
        (12_900, "12.9K"),
        (4_200_000, "4.2M"),
        (250_000, "250K"),
    ],
)
def test_compact(n: int, text: str) -> None:
    assert compact(n) == text


@pytest.mark.parametrize(
    ("part", "whole", "text"),
    [(0, 10, "0%"), (1, 1000, "<1%"), (999, 1000, ">99%"), (10, 10, "100%"), (1, 0, "—")],
)
def test_percent(part: int, whole: int, text: str) -> None:
    assert percent(part, whole) == text


def test_json_report_includes_facts() -> None:
    data = run_checks(make_profile(norms=[0.5 + i / 50 for i in range(100)])).to_dict()
    facts = data["facts"]
    assert facts["text_field"] == "content"
    assert facts["dimensions"] == {"3": 100}
    assert len(facts["norms"]["histogram"]) == 20
    assert sum(b["count"] for b in facts["norms"]["histogram"]) == 100
