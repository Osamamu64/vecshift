from __future__ import annotations

import json

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from vecshift.cli import app
from vecshift.connectors.pgvector import (
    TargetSelectionError,
    connect,
    inspect,
    prepare,
)
from vecshift.doctor import IndexProfile, Severity, run_checks

TAG = "openai/text-embedding-3-small@3#abc"


def profile_of(dsn: str, **kwargs: object) -> IndexProfile:
    conn = connect(prepare(dsn))
    try:
        return inspect(conn, **kwargs)  # type: ignore[arg-type]
    finally:
        conn.rollback()
        conn.close()


def finding_ids(profile: IndexProfile) -> dict[str, Severity]:
    return {f.id: f.severity for f in run_checks(profile).findings}


def create_healthy(conn: psycopg.Connection, name: str = "documents", rows: int = 200) -> None:
    conn.execute(
        f"""
        CREATE TABLE public.{name} (
            id bigserial PRIMARY KEY,
            content text NOT NULL,
            metadata jsonb NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now(),
            embedding extensions.vector(3)
        )
        """
    )
    conn.execute(
        f"""
        INSERT INTO public.{name} (content, metadata, embedding)
        SELECT 'chunk ' || g,
               jsonb_build_object('model_tag', %s::text),
               ('[' || cos(g) || ',' || sin(g) || ',0]')::extensions.vector
        FROM generate_series(1, %s) g
        """,
        (TAG, rows),
    )
    conn.execute(
        f"CREATE INDEX ON public.{name} USING hnsw (embedding extensions.vector_cosine_ops)"
    )


def test_healthy_supabase_style_table(db_dsn: str, setup: psycopg.Connection) -> None:
    create_healthy(setup)
    profile = profile_of(db_dsn)

    assert profile.target == "public.documents.embedding"
    assert profile.declared_dimensions == 3
    assert profile.text_field == "content"
    assert profile.model_field == "metadata->>'model_tag'"
    assert profile.updated_at_field == "updated_at"
    assert profile.sample.rows == 200
    assert profile.sample.method == "full table"
    assert [(i.method, i.metric) for i in profile.ann_indexes] == [("hnsw", "cosine")]
    assert run_checks(profile).worst is Severity.OK


def test_messy_table(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.chunks (content text, metadata jsonb, embedding extensions.vector)"
    )
    setup.execute(
        """
        INSERT INTO public.chunks (content, metadata, embedding)
        SELECT 'doc ' || g, '{"model": "ada-002"}',
               ('[' || cos(g) || ',' || sin(g) || ',0]')::extensions.vector
        FROM generate_series(1, 90) g
        """
    )
    setup.execute(
        """
        INSERT INTO public.chunks (content, metadata, embedding)
        SELECT CASE WHEN g = 1 THEN NULL ELSE 'same text' END, '{"model": "3-small"}',
               ('[' || g || ',1,2,3]')::extensions.vector
        FROM generate_series(1, 10) g
        """
    )
    setup.execute("INSERT INTO public.chunks (content, embedding) VALUES ('zero', '[0,0,0]')")
    setup.execute("INSERT INTO public.chunks (content) VALUES ('not embedded yet')")

    profile = profile_of(db_dsn)
    found = finding_ids(profile)

    assert profile.model_field == "metadata->>'model'"
    for expected in (
        "dims.mixed",
        "dims.undeclared",
        "norms.zero",
        "norms.mixed",
        "models.mixed",
        "models.partial",
        "text.partial",
        "vectors.null",
        "duplicates.text",
        "index.missing",
        "sync.none",
        "sync.no_primary_key",
    ):
        assert expected in found, expected


def test_table_without_text(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute("CREATE TABLE public.vecs (id text PRIMARY KEY, embedding extensions.vector(2))")
    setup.execute("INSERT INTO public.vecs VALUES ('a', '[1,0]'), ('b', '[0,1]')")
    found = finding_ids(profile_of(db_dsn))
    assert found["text.missing"] is Severity.ERROR
    assert found["models.untracked"] is Severity.INFO


def test_explicit_text_column(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.t (id int PRIMARY KEY, passage text, embedding extensions.vector(2))"
    )
    setup.execute("INSERT INTO public.t VALUES (1, 'hello', '[1,0]')")
    assert profile_of(db_dsn).text_field is None
    assert profile_of(db_dsn, text_column="passage").text_field == "passage"
    with pytest.raises(TargetSelectionError):
        profile_of(db_dsn, text_column="nope")


def test_halfvec_column(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.h (id int PRIMARY KEY, content text, embedding extensions.halfvec(2))"
    )
    setup.execute("INSERT INTO public.h VALUES (1, 'a', '[0.6,0.8]'), (2, 'b', '[1,0]')")
    profile = profile_of(db_dsn)
    assert profile.vector_type == "halfvec"
    assert profile.max_indexable_dimensions == 4000
    assert finding_ids(profile)["norms.ok"] is Severity.OK


def test_large_vectors_and_halfvec_expression_index(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.big (id int PRIMARY KEY, content text, "
        "embedding extensions.vector(3000))"
    )
    setup.execute(
        "INSERT INTO public.big SELECT g, 'x' || g, "
        "array_fill(0.01::real, ARRAY[3000])::extensions.vector FROM generate_series(1, 5) g"
    )
    assert "dims.too_large_to_index" in finding_ids(profile_of(db_dsn))

    setup.execute(
        "CREATE INDEX big_hnsw ON public.big USING hnsw "
        "((embedding::extensions.halfvec(3000)) extensions.halfvec_cosine_ops)"
    )
    profile = profile_of(db_dsn)
    assert [(i.name, i.metric) for i in profile.ann_indexes] == [("big_hnsw", "cosine")]
    assert "dims.too_large_to_index" not in finding_ids(profile)


def test_target_selection(db_dsn: str, setup: psycopg.Connection) -> None:
    create_healthy(setup, "documents")
    setup.execute("CREATE SCHEMA rag")
    setup.execute(
        "CREATE TABLE rag.documents (id int PRIMARY KEY, a extensions.vector(2), "
        "b extensions.vector(2))"
    )

    with pytest.raises(TargetSelectionError) as exc:
        profile_of(db_dsn)
    assert len(exc.value.candidates) == 3

    assert profile_of(db_dsn, table="public.documents").target == "public.documents.embedding"
    assert profile_of(db_dsn, table="rag.documents", column="b").target == "rag.documents.b"
    with pytest.raises(TargetSelectionError):
        profile_of(db_dsn, table="documents")  # ambiguous across schemas
    with pytest.raises(TargetSelectionError):
        profile_of(db_dsn, table="missing")


def test_no_vector_columns(db_dsn: str) -> None:
    with pytest.raises(TargetSelectionError, match="No pgvector columns"):
        profile_of(db_dsn)


def test_large_table_is_sampled(db_dsn: str, setup: psycopg.Connection) -> None:
    create_healthy(setup, rows=20_000)
    setup.execute("ANALYZE public.documents")
    profile = profile_of(db_dsn, sample_size=500)
    assert profile.sample.method == "table sample"
    assert 250 <= profile.sample.rows <= 500
    assert profile.estimated_rows == 20_000


def test_row_level_security(reader: str, setup: psycopg.Connection) -> None:
    create_healthy(setup)
    role = conninfo_to_dict(reader)["user"]
    setup.execute(f"GRANT SELECT ON public.documents TO {role}")
    setup.execute("ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY")

    profile = profile_of(reader)
    found = finding_ids(profile)
    assert profile.rows_hidden_by_access_rules
    assert profile.sample.rows == 0
    assert found["access.row_security"] is Severity.WARNING
    assert found["sample.empty"] is Severity.INFO


def test_connection_is_read_only_and_pooler_safe(db_dsn: str) -> None:
    conn = connect(prepare(db_dsn), statement_timeout_s=5)
    try:
        assert conn.prepare_threshold is None
        assert conn.execute("SHOW statement_timeout").fetchone() == ("5s",)
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("CREATE TABLE should_fail (id int)")
    finally:
        conn.rollback()
        conn.close()


def test_cli_json(db_dsn: str, setup: psycopg.Connection) -> None:
    create_healthy(setup)
    result = CliRunner().invoke(app, ["doctor", "--json"], env={"VECSHIFT_DSN": db_dsn})
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target"] == "public.documents.embedding"
    assert data["summary"]["error"] == 0
    params = conninfo_to_dict(db_dsn)
    assert data["connection"] == (
        f"postgresql://{params['user']}@{params.get('host', 'localhost')}:"
        f"{params.get('port', 5432)}/{params['dbname']}"
    )
    if params.get("password"):
        assert f":{params['password']}@" not in result.output


def test_cli_fail_on_and_candidates(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute("CREATE TABLE public.a (embedding extensions.vector(2))")
    setup.execute("CREATE TABLE public.b (embedding extensions.vector(2))")
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"], env={"VECSHIFT_DSN": db_dsn})
    assert result.exit_code == 2
    assert "public.a.embedding" in result.output and "public.b.embedding" in result.output

    result = runner.invoke(
        app, ["doctor", "--table", "a", "--fail-on", "error"], env={"VECSHIFT_DSN": db_dsn}
    )
    assert result.exit_code == 1  # no primary key


def test_cli_connection_failure_is_clean() -> None:
    result = CliRunner().invoke(app, ["doctor", "--dsn", "postgresql://u:hunter2@127.0.0.1:1/db"])
    assert result.exit_code == 2
    assert "Couldn't connect" in result.output
    assert "hunter2" not in result.output


def test_model_recorded_in_its_own_column(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.m (id int PRIMARY KEY, content text, embedding_model text, "
        "embedding extensions.vector(2))"
    )
    setup.execute(
        "INSERT INTO public.m VALUES (1, 'a', 'ada-002', '[1,0]'), (2, 'b', '3-small', '[0,1]')"
    )
    profile = profile_of(db_dsn)
    assert profile.model_field == "embedding_model"
    assert dict(profile.sample.models) == {"ada-002": 1, "3-small": 1}
    assert finding_ids(profile)["models.mixed"] is Severity.ERROR


def test_read_all_policy_is_not_flagged(reader: str, setup: psycopg.Connection) -> None:
    """The read-only role recipe in docs/connectors/pgvector.md sees every row."""
    create_healthy(setup)
    role = conninfo_to_dict(reader)["user"]
    setup.execute(f"GRANT SELECT ON public.documents TO {role}")
    setup.execute("ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY")
    setup.execute(f"CREATE POLICY read_all ON public.documents FOR SELECT TO {role} USING (true)")

    profile = profile_of(reader)
    assert not profile.rows_hidden_by_access_rules
    assert profile.sample.rows == 200

    setup.execute(
        f"CREATE POLICY narrow ON public.documents AS RESTRICTIVE FOR SELECT TO {role} "
        "USING (id < 100)"
    )
    assert profile_of(reader).rows_hidden_by_access_rules


@pytest.mark.parametrize(
    ("rows", "sample_size", "method"),
    [
        (20_000, 500, "table sample"),
        (3_000, 2_000, "spread across the table"),
    ],
)
def test_sample_spans_the_whole_table(
    db_dsn: str, setup: psycopg.Connection, rows: int, sample_size: int, method: str
) -> None:
    """Rows written later (here, by a newer model) must show up in the sample."""
    setup.execute(
        "CREATE TABLE public.docs (id int PRIMARY KEY, content text, "
        "embedding_model text, embedding extensions.vector(2))"
    )
    setup.execute(
        "INSERT INTO public.docs SELECT g, 'c' || g, "
        "CASE WHEN g <= %s THEN 'old' ELSE 'new' END, '[1,0]' FROM generate_series(1, %s) g",
        (rows // 2, rows),
    )
    setup.execute("ANALYZE public.docs")
    profile = profile_of(db_dsn, sample_size=sample_size)
    assert profile.sample.method == method
    assert profile.sample.rows == sample_size
    old, new = profile.sample.models["old"], profile.sample.models["new"]
    assert min(old, new) > sample_size * 0.3, (old, new)


def test_cli_html_report(
    db_dsn: str, setup: psycopg.Connection, tmp_path: pytest.TempPathFactory
) -> None:
    create_healthy(setup)
    out = tmp_path / "report.html"  # type: ignore[operator]
    result = CliRunner().invoke(
        app, ["doctor", "--json", "--html", str(out)], env={"VECSHIFT_DSN": db_dsn}
    )
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)  # the notice goes to stderr, so stdout stays valid JSON
    assert "HTML report written" in result.stderr
    page = out.read_text(encoding="utf-8")
    assert "Ready to migrate" in page
    assert "public.documents.embedding" in page
