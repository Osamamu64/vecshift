"""Writing job files: new ones from init's templates, and edits that keep comments."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from vecshift.jobs import JobSpec, load_job
from vecshift.jobs.spec import qdrant_template, template
from vecshift.service.editing import InvalidJob, render, update, validate, write
from vecshift.service.errors import ServiceError

SPECS: list[dict[str, Any]] = [
    {"source": {"table": "docs"}, "model": "hash/8"},
    {
        "name": "full",
        "source": {
            "dsn_env": "REPLICA_DSN",
            "table": "public.docs",
            "vector_column": "embedding",
            "text_column": "body",
            "model": "openai/text-embedding-3-small",
        },
        "target": {
            "column": "emb_v2",
            "vector_type": "halfvec",
            "index": "ivfflat",
            "metric": "l2",
        },
        "model": "openai/text-embedding-3-large,dims=1024",
        "limits": {"budget_usd": 12.5, "tokens_per_minute": 1000, "requests_per_minute": 10},
    },
    {
        "source": {"type": "qdrant", "collection": "live", "vector": "dense", "text_field": "a.b"},
        "target": {"type": "qdrant", "alias": "live", "collection": "live_v3", "index": "none"},
        "model": "ollama/nomic-embed-text",
    },
    {
        "source": {"type": "qdrant", "url_env": "Q_URL", "key_env": "Q_KEY", "collection": "c"},
        "target": {"type": "qdrant", "url_env": "OTHER_URL", "metric": "inner_product"},
        "model": "hash/16",
        "limits": {"budget_usd": 3},
    },
    {
        "source": {"table": "docs", "text_column": "body"},
        "target": {"type": "qdrant", "alias": "docs_live", "key_env": "K"},
        "model": "hash/8",
    },
    {"name": 'odd "name" # not a comment', "source": {"table": "t"}, "model": "hash/8"},
]


def spec(data: dict[str, Any]) -> JobSpec:
    return validate(data)


def reads_as(text: str, expected: JobSpec) -> None:
    assert JobSpec.model_validate(yaml.safe_load(text)) == expected


@pytest.mark.parametrize("data", SPECS)
def test_a_new_file_is_inits_template_and_reads_back_as_the_spec(data: dict[str, Any]) -> None:
    job = spec(data)
    written = render(job)
    assert written.comments_kept
    assert written.text.startswith("# vecshift job: re-embed")
    assert "# Check it with `vecshift plan`" in written.text
    reads_as(written.text, job)


@pytest.mark.parametrize("data", SPECS)
def test_an_unchanged_spec_leaves_the_file_as_it_was(data: dict[str, Any]) -> None:
    text = render(spec(data)).text
    assert update(text, spec(data)).text == text


def test_inits_own_files_are_left_alone_when_nothing_changes() -> None:
    for text in (
        template("docs", "hash/8"),
        template("s.docs", "hash/8", vector_column="e", budget_usd=5),
        qdrant_template("hash/8", collection="live", alias="live"),
        qdrant_template("hash/8", table="docs", new_collection="n"),
    ):
        job = JobSpec.model_validate(yaml.safe_load(text))
        assert update(text, job) == type(update(text, job))(text, True)


def test_a_change_touches_only_its_line() -> None:
    before = template("docs", "hash/8", budget_usd=5)
    job = JobSpec.model_validate(yaml.safe_load(before))
    after = update(before, job.model_copy(update={"model": "hash/16"})).text
    changed = [
        (a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b
    ]
    assert changed == [('model: "hash/8"', "model: hash/16")]


def test_unset_settings_are_commented_out_and_examples_are_used() -> None:
    before = template("docs", "hash/8", budget_usd=5)
    job = validate(
        {
            "name": "docs-reembed",
            "source": {"table": "docs", "text_column": "body"},
            "model": "hash/8",
            "limits": {"tokens_per_minute": 2000},
        }
    )
    after = update(before, job)
    assert after.comments_kept
    lines = after.text.splitlines()
    budget = next(line for line in lines if "budget_usd" in line)
    old = next(line for line in before.splitlines() if "budget_usd" in line)
    assert budget.startswith("  # budget_usd: 5")
    assert budget.index("# plan fails") == old.index("# plan fails"), "the note stays put"
    text = next(line for line in lines if "text_column" in line)
    assert text.startswith("  text_column: body ")
    assert text.index("# detected") == budget.index("# plan fails"), "the example's note stays"
    assert any(line.startswith("  tokens_per_minute: 2000 ") for line in lines)
    assert after.text.count("\n") == before.count("\n"), "examples are filled in, not added"
    reads_as(after.text, job)


def test_new_settings_go_in_their_section() -> None:
    before = "name: docs\nsource:\n  table: docs  # the table\nmodel: hash/8\n"
    job = validate(
        {
            "name": "docs",
            "source": {"table": "docs", "dsn_env": "OTHER"},
            "target": {"column": "e3"},
            "model": "hash/8",
            "limits": {"budget_usd": 2},
        }
    )
    after = update(before, job)
    assert after.comments_kept
    assert after.text.startswith(
        "name: docs\nsource:\n  table: docs  # the table\n  dsn_env: OTHER\nmodel: hash/8\n"
    )
    assert "\ntarget:\n  column: e3\n" in after.text
    assert "\nlimits:\n  budget_usd: 2\n" in after.text
    reads_as(after.text, job)


def test_quotes_and_hashes_in_values() -> None:
    before = "name: \"a # b\"   # the name\nsource:\n  table: docs\nmodel: 'hash/8'  # model\n"
    job = validate({"name": "c # d", "source": {"table": "docs"}, "model": "hash/8"})
    after = update(before, job).text
    assert after.splitlines()[0] == 'name: "c # d"   # the name'
    assert after.splitlines()[3] == "model: 'hash/8'  # model", "an equal value isn't rewritten"
    reads_as(after, job)


@pytest.mark.parametrize(
    "text",
    [
        "name: docs\nsource: {table: docs}\nmodel: hash/8\n",
        "name: docs\nsource:\n  table: &t docs\nmodel: hash/8\n",
        "name: >\n  docs\nsource:\n  table: docs\nmodel: hash/8\n",
        "name: docs\nsource:\n    table: docs\n  dsn_env: X\nmodel: hash/8\n",
        "name: docs\ntarget: null\nsource:\n  table: docs\nmodel: hash/8\n",
        "name: docs\nsource:\n\ttable: docs\nmodel: hash/8\n",
    ],
)
def test_layouts_it_cant_follow_are_written_afresh(text: str) -> None:
    job = validate({"name": "docs", "source": {"table": "docs2"}, "model": "hash/8"})
    after = update(text, job)
    assert not after.comments_kept
    reads_as(after.text, job)


def test_problems_use_the_job_files_wording() -> None:
    with pytest.raises(InvalidJob) as caught:
        validate({"source": {"type": "qdrant"}, "model": "nope", "extra": 1})
    error = caught.value
    assert error.code == "job.invalid"
    assert error.message.startswith("The job has problems:\n")
    assert {"loc": "source.collection", "message": "required"} in error.errors
    assert {"loc": "extra", "message": "unknown setting"} in error.errors
    assert any(e["loc"] == "model" for e in error.errors)


def test_writing(tmp_path: Path) -> None:
    path = tmp_path / "docs.yaml"
    text = render(spec(SPECS[0])).text
    write(path, text, new=True)
    assert load_job(path) == spec(SPECS[0])
    with pytest.raises(ServiceError) as caught:
        write(path, text, new=True)
    assert caught.value.code == "job.exists"
    write(path, "name: x\nsource:\n  table: t\nmodel: hash/8\n", new=False)
    assert load_job(path).name == "x"
    assert [p.name for p in tmp_path.iterdir()] == ["docs.yaml"], "no temp file left"
