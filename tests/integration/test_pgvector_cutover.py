"""``vecshift cutover`` and ``rollback`` against a real PostgreSQL."""

import json
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

from tests.integration.test_pgvector_apply import (
    DIMS,
    apply_job,
    json_lines,
    layout,
    mismatched,
    model_url,  # noqa: F401  (a fixture)
    run_apply,
)
from tests.integration.test_pgvector_doctor import create_healthy
from vecshift.cli import app
from vecshift.connectors.pgvector.switch import PgSwitch
from vecshift.connectors.pgvector.writer import Busy, PgWriter

runner = CliRunner()


def run(command: str, path: Path, dsn: str, *args: str) -> Any:
    return runner.invoke(app, [command, str(path), *args], env={"VECSHIFT_DSN": dsn})


def columns(conn: psycopg.Connection) -> dict[str, int]:
    """Vector columns of public.documents and their sizes."""
    rows = conn.execute(
        """
        SELECT a.attname, a.atttypmod FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
        WHERE a.attrelid = 'public.documents'::regclass AND t.typname = 'vector'
          AND NOT a.attisdropped
        """
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def indexes(conn: psycopg.Connection) -> dict[str, str]:
    """Index name -> the column it covers."""
    rows = conn.execute(
        """
        SELECT c.relname, a.attname FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
        WHERE i.indrelid = 'public.documents'::regclass AND NOT i.indisprimary
        """
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def triggers(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT tgname FROM pg_trigger WHERE tgrelid = 'public.documents'::regclass "
        "AND NOT tgisinternal"
    ).fetchall()
    return {r[0] for r in rows}


@pytest.fixture
def applied(db_dsn: str, setup: psycopg.Connection, tmp_path: Path, model_url: str) -> Path:  # noqa: F811
    create_healthy(setup, rows=120)
    path = apply_job(tmp_path, model_url)
    result = run_apply(path, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    return path


def test_cutover_and_rollback(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    check = run("cutover", applied, db_dsn, "--check", "--json")
    assert check.exit_code == 0, check.output
    report = json.loads(check.stdout)
    assert report["status"] == "ready"
    assert "cutover.size_change" in {f["id"] for f in report["findings"]}
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}, "--check changes nothing"

    result = run("cutover", applied, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "cut_over"
    assert columns(setup) == {"embedding": DIMS, "embedding_old": 3}
    assert mismatched(setup, "embedding") == 0, "the app's column now holds the new vectors"
    names = indexes(setup)
    assert names["documents_embedding_hnsw_idx"] == "embedding"
    assert names["documents_embedding_idx"] == "embedding_old"
    assert not any("pending" in n for n in names), "the temporary index is gone"
    assert triggers(setup) == {"vecshift_sync_embedding_old"}

    # The application's search SQL is unchanged and uses the new vectors and index.
    setup.execute("SET enable_seqscan = off")  # the table is tiny
    plan = setup.execute(
        "EXPLAIN SELECT id FROM public.documents ORDER BY embedding "
        "OPERATOR(extensions.<=>) %s::extensions.vector LIMIT 5",
        ("[" + ",".join(["0.1"] * DIMS) + "]",),
    ).fetchall()
    assert "documents_embedding_hnsw_idx" in " ".join(r[0] for r in plan)
    setup.execute("RESET enable_seqscan")

    # An edit after cutover clears the old vector, so rollback can report it.
    setup.execute("UPDATE public.documents SET content = 'edited' WHERE id = 5")
    assert setup.execute(
        "SELECT embedding_old IS NULL, embedding IS NULL FROM public.documents WHERE id = 5"
    ).fetchone() == (True, False)

    blocked = run_apply(applied, db_dsn, "--yes")
    assert blocked.exit_code == 1 and "already cut over" in blocked.output
    again = run("cutover", applied, db_dsn, "--check", "--json")
    assert again.exit_code == 1
    assert "cutover.done" in {f["id"] for f in json.loads(again.stdout)["findings"]}

    back = run("rollback", applied, db_dsn, "--yes", "--json")
    assert back.exit_code == 0, back.output
    assert json.loads(back.stdout) == {
        "status": "rolled_back",
        "live": "embedding",
        "new": "embedding_v2",
        "missing": 1,
    }
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}
    names = indexes(setup)
    assert names["documents_embedding_idx"] == "embedding"
    assert names["documents_embedding_v2_hnsw_idx"] == "embedding_v2"
    assert triggers(setup) == {"vecshift_sync_embedding_v2"}

    history = json.loads(
        (applied.parent / ".vecshift" / "documents-reembed.state.json").read_text()
    )
    assert [h["event"] for h in history["history"]] == ["cutover", "rollback"]

    nothing = run("rollback", applied, db_dsn, "--yes")
    assert nothing.exit_code == 2 and "no cutover to roll back" in nothing.output


def test_cutover_catches_up_first(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    setup.execute("UPDATE public.documents SET content = content || ' late' WHERE id <= 7")
    setup.execute("INSERT INTO public.documents (content, metadata) VALUES ('newest', '{}')")
    check = json.loads(run("cutover", applied, db_dsn, "--check", "--json").stdout)
    assert check["pending"] == 8 and check["status"] == "blocked"

    result = run("cutover", applied, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["caught_up"] == 8
    assert mismatched(setup, "embedding") == 0


def test_bound_views_block_cutover(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    setup.execute("CREATE VIEW public.doc_vectors AS SELECT id, embedding FROM public.documents")
    result = run("cutover", applied, db_dsn, "--yes", "--json")
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "blocked"
    finding = next(f for f in report["findings"] if f["id"] == "cutover.dependents")
    assert "doc_vectors" in finding["detail"]
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}


def test_nothing_to_cut_over(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    model_url: str,  # noqa: F811
) -> None:
    create_healthy(setup, rows=10)
    path = apply_job(tmp_path, model_url)
    result = run("cutover", path, db_dsn, "--check", "--json")
    assert result.exit_code == 1
    assert "cutover.not_applied" in {f["id"] for f in json.loads(result.stdout)["findings"]}


def test_cutover_needs_confirmation(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    result = run("cutover", applied, db_dsn)
    assert result.exit_code == 2 and "--yes" in result.output
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}


def test_cutover_backs_off_when_busy(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    sleeps: list[float] = []
    lay = layout()
    lay = type(lay)(**{**{f: getattr(lay, f) for f in lay.__slots__}, "live": "embedding"})
    with psycopg.connect(db_dsn) as blocker, psycopg.connect(db_dsn, autocommit=True) as conn:
        blocker.execute("LOCK TABLE public.documents IN ACCESS SHARE MODE")
        writer = PgWriter(conn, lay, lock_retries=2, lock_timeout_ms=100, sleep=sleeps.append)
        with pytest.raises(Busy, match="cut over"):
            PgSwitch(writer).cutover()
        assert sleeps == [1.0]
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}


# --- Apply features that cutover relies on


def test_apply_until_stops_part_way(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    model_url: str,  # noqa: F811
) -> None:
    create_healthy(setup, rows=200)
    path = apply_job(tmp_path, model_url, batch=5)
    half = run_apply(path, db_dsn, "--yes", "--json", "--until", "50")
    assert half.exit_code == 3, half.output
    result = json_lines(half.stdout)[-1]
    assert result["rows_written"] == 100 and result["remaining"] == 100
    assert result["index"] == "pending"
    rest = json_lines(run_apply(path, db_dsn, "--yes", "--json").stdout)[-1]
    assert rest["status"] == "complete" and rest["rows_written"] == 100


def test_trigger_keeps_vectors_the_app_writes(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    """An app already writing the new model's vectors isn't undone by the sync trigger."""
    vector = "[" + ",".join(["0.5"] * DIMS) + "]"
    setup.execute(
        "UPDATE public.documents SET content = 'by app', embedding_v2 = %s::extensions.vector "
        "WHERE id = 1",
        (vector,),
    )
    setup.execute("UPDATE public.documents SET content = 'text only' WHERE id = 2")
    rows = setup.execute(
        "SELECT id, embedding_v2 IS NULL FROM public.documents WHERE id IN (1, 2) ORDER BY id"
    ).fetchall()
    assert rows == [(1, False), (2, True)]


def test_rejected_rows_block_cutover_unless_allowed(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    model_url: str,  # noqa: F811
) -> None:
    create_healthy(setup, rows=30)
    setup.execute("UPDATE public.documents SET content = 'REJECT-ME' WHERE id = 9")
    path = apply_job(tmp_path, model_url)
    assert run_apply(path, db_dsn, "--yes").exit_code == 0

    refused = run("cutover", path, db_dsn, "--yes")
    assert refused.exit_code == 2 and "--allow-missing" in refused.output
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}

    allowed = run("cutover", path, db_dsn, "--yes", "--allow-missing")
    assert allowed.exit_code == 0, allowed.output
    assert setup.execute("SELECT embedding FROM public.documents WHERE id = 9").fetchone() == (
        None,
    )


def test_cleanup_drops_the_old_vectors(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    assert run("cleanup", applied, db_dsn, "--yes").exit_code == 2, "nothing to clean up yet"
    assert run("cutover", applied, db_dsn, "--yes").exit_code == 0

    # Dropping the column by hand is refused rather than leaving a broken trigger behind.
    with pytest.raises(psycopg.errors.DependentObjectsStillExist):
        setup.execute("ALTER TABLE public.documents DROP COLUMN embedding_old")

    unconfirmed = run("cleanup", applied, db_dsn)
    assert unconfirmed.exit_code == 2 and "--yes" in unconfirmed.output
    result = run("cleanup", applied, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "cleaned_up", "dropped": "embedding_old"}
    assert columns(setup) == {"embedding": DIMS}
    assert triggers(setup) == set()
    assert set(indexes(setup)) == {"documents_embedding_hnsw_idx"}
    setup.execute("UPDATE public.documents SET content = 'still works' WHERE id = 1")

    # A later migration can start from here.
    plan = runner.invoke(app, ["plan", str(applied), "--json"], env={"VECSHIFT_DSN": db_dsn})
    assert "plan.cut_over" not in plan.stdout


# --- Edge cases


@pytest.mark.parametrize("pk_type", ["uuid", "text"])
def test_round_trip_with_non_numeric_keys(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    model_url: str,  # noqa: F811
    pk_type: str,
) -> None:
    """Supabase tables often use UUID keys; keyset paging and rejected-row IDs must work."""
    key = "gen_random_uuid()" if pk_type == "uuid" else "'doc-' || g"
    setup.execute(
        f"CREATE TABLE public.documents (id {pk_type} PRIMARY KEY, content text, "
        "metadata jsonb, embedding extensions.vector(3))"
    )
    setup.execute(
        f"INSERT INTO public.documents SELECT {key}, 'chunk ' || g, '{{}}', '[1,0,0]' "
        "FROM generate_series(1, 150) g"
    )
    setup.execute("UPDATE public.documents SET content = 'REJECT-ME' WHERE content = 'chunk 77'")
    path = apply_job(tmp_path, model_url, batch=7)
    applied = run_apply(path, db_dsn, "--yes", "--json")
    assert applied.exit_code == 0, applied.output
    final = json_lines(applied.stdout)[-1]
    assert final["rows_written"] == 149 and final["rows_failed"] == 1
    again = json_lines(run_apply(path, db_dsn, "--yes", "--json").stdout)[-1]
    assert again["rows_written"] == 0, "the rejected row is skipped, not retried in a loop"

    assert run("cutover", path, db_dsn, "--yes", "--allow-missing").exit_code == 0
    assert columns(setup) == {"embedding": DIMS, "embedding_old": 3}
    assert run("rollback", path, db_dsn, "--yes").exit_code == 0
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}


def test_rows_arriving_during_the_switch_are_caught_up(
    db_dsn: str, setup: psycopg.Connection, applied: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row written after the final catch-up makes the switch retry, not lose it."""
    import vecshift.cli_cutover as cli_cutover
    from vecshift.connectors.pgvector import switch as switch_module

    real = switch_module.PgSwitch.prepare
    calls = 0

    def prepare_then_insert(self: PgSwitch) -> None:
        nonlocal calls
        real(self)
        calls += 1
        if calls == 1:
            setup.execute("INSERT INTO public.documents (content, metadata) VALUES ('gap', '{}')")

    monkeypatch.setattr(switch_module.PgSwitch, "prepare", prepare_then_insert)
    assert cli_cutover.CATCH_UP_TRIES >= 2
    result = run("cutover", applied, db_dsn, "--yes", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["caught_up"] == 1
    assert mismatched(setup, "embedding") == 0


def test_a_failed_switch_changes_nothing(
    db_dsn: str, setup: psycopg.Connection, applied: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If any statement in the switch fails, the renames roll back with it."""
    from psycopg import sql

    monkeypatch.setattr(
        PgWriter, "sync_trigger_sql", lambda self, column=None: sql.SQL("SELECT 1/0")
    )
    result = run("cutover", applied, db_dsn, "--yes")
    assert result.exit_code == 2 and "division by zero" in result.output
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}
    assert triggers(setup) == {"vecshift_sync_embedding_v2"}
    assert not any("pending" in n for n in indexes(setup)), "the temporary index is removed"


def test_index_names_that_are_taken(db_dsn: str, setup: psycopg.Connection, applied: Path) -> None:
    """An old index already using the name the new one wants is renamed out of the way."""
    setup.execute(
        "ALTER INDEX public.documents_embedding_idx RENAME TO documents_embedding_hnsw_idx"
    )
    assert run("cutover", applied, db_dsn, "--yes").exit_code == 0
    assert indexes(setup) == {
        "documents_embedding_hnsw_idx": "embedding",
        "documents_embedding_old_hnsw_idx": "embedding_old",
    }
    assert run("rollback", applied, db_dsn, "--yes").exit_code == 0
    assert indexes(setup) == {
        "documents_embedding_hnsw_idx": "embedding",
        "documents_embedding_v2_hnsw_idx": "embedding_v2",
    }


def test_cutover_waits_for_a_running_apply(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    lay = layout()
    with psycopg.connect(db_dsn, autocommit=True) as other:
        PgWriter(other, lay).acquire()  # an apply in progress elsewhere
        result = run("cutover", applied, db_dsn, "--yes")
    assert result.exit_code == 2 and "already working" in result.output
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}


def test_leftover_temporary_index_is_replaced(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    """A cutover killed part way leaves its temporary index; the next one copes."""
    setup.execute(
        "CREATE INDEX documents_embedding_v2_pending_idx ON public.documents (id) "
        "WHERE embedding_v2 IS NULL"
    )
    assert run("cutover", applied, db_dsn, "--yes").exit_code == 0
    assert not any("pending" in n for n in indexes(setup))


def test_views_block_rollback_and_cleanup(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    assert run("cutover", applied, db_dsn, "--yes").exit_code == 0
    setup.execute("CREATE VIEW public.v_old AS SELECT id, embedding_old FROM public.documents")
    for command in ("rollback", "cleanup"):
        result = run(command, applied, db_dsn, "--yes")
        assert result.exit_code == 2 and "v_old" in result.output, command
    assert columns(setup) == {"embedding": DIMS, "embedding_old": 3}


def test_rollback_counts_rows_inserted_after_cutover(
    db_dsn: str, setup: psycopg.Connection, applied: Path
) -> None:
    assert run("cutover", applied, db_dsn, "--yes").exit_code == 0
    vector = "[" + ",".join(["0.2"] * DIMS) + "]"
    setup.execute(
        "INSERT INTO public.documents (content, metadata, embedding) "
        "SELECT 'post ' || g, '{}', %s::extensions.vector FROM generate_series(1, 3) g",
        (vector,),
    )
    back = run("rollback", applied, db_dsn, "--yes", "--json")
    assert json.loads(back.stdout)["missing"] == 3


def test_budget_during_the_final_catch_up(
    db_dsn: str,
    setup: psycopg.Connection,
    tmp_path: Path,
    model_url: str,  # noqa: F811
) -> None:
    create_healthy(setup, rows=40)
    path = apply_job(tmp_path, model_url, "budget_usd: 0.001")
    assert run_apply(path, db_dsn, "--yes").exit_code == 0
    setup.execute("UPDATE public.documents SET content = repeat('long text ', 3000)")
    result = run("cutover", path, db_dsn, "--yes")
    assert result.exit_code == 2 and "Nothing was switched" in result.output
    assert columns(setup) == {"embedding": 3, "embedding_v2": DIMS}
