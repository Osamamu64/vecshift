from collections.abc import Mapping

import pytest
from psycopg.conninfo import conninfo_to_dict

from vecshift.connectors.pgvector import ConnectError, SupabaseMode, prepare
from vecshift.connectors.pgvector.connection import _hint_for

SECRET = "s3cr3t-p4ss"


def params(dsn: str) -> Mapping[str, object]:
    return conninfo_to_dict(prepare(dsn).conninfo)


def test_plain_postgres_is_left_alone() -> None:
    settings = prepare(f"postgresql://app:{SECRET}@localhost:5432/rag")
    assert settings.supabase is None
    assert "sslmode" not in params(f"postgresql://app:{SECRET}@localhost:5432/rag")
    assert settings.description == "PostgreSQL"


@pytest.mark.parametrize(
    ("host", "port", "mode"),
    [
        ("db.abcdefgh.supabase.co", 5432, SupabaseMode.DIRECT),
        ("db.abcdefgh.supabase.co", 6543, SupabaseMode.TRANSACTION_POOLER),
        ("aws-0-eu-central-1.pooler.supabase.com", 5432, SupabaseMode.SESSION_POOLER),
        ("aws-1-us-east-1.pooler.supabase.com", 6543, SupabaseMode.TRANSACTION_POOLER),
    ],
)
def test_supabase_modes_and_tls(host: str, port: int, mode: SupabaseMode) -> None:
    dsn = f"postgresql://postgres.abcdefgh:{SECRET}@{host}:{port}/postgres"
    settings = prepare(dsn)
    assert settings.supabase is mode
    assert params(dsn)["sslmode"] == "require"
    assert settings.description == f"Supabase {mode}"


def test_explicit_sslmode_is_kept() -> None:
    dsn = f"postgresql://postgres:{SECRET}@db.x.supabase.co:5432/postgres?sslmode=verify-full"
    assert params(dsn)["sslmode"] == "verify-full"


def test_key_value_format() -> None:
    settings = prepare(f"host=db.x.supabase.co port=5432 user=postgres password={SECRET}")
    assert settings.supabase is SupabaseMode.DIRECT


def test_display_never_contains_the_password() -> None:
    settings = prepare(f"postgresql://postgres:{SECRET}@db.x.supabase.co:5432/postgres")
    assert SECRET not in settings.display
    assert settings.display == "postgresql://postgres@db.x.supabase.co:5432/postgres"


def test_application_name_is_set() -> None:
    assert params("postgresql://localhost/db")["application_name"] == "vecshift"


def test_invalid_connection_string() -> None:
    with pytest.raises(ConnectError):
        prepare("this is not = a = dsn")


def test_ipv6_hint_for_supabase_direct() -> None:
    settings = prepare("postgresql://postgres@db.x.supabase.co:5432/postgres")
    hint = _hint_for(settings, "connection failed: Network is unreachable")
    assert hint is not None and "session pooler" in hint


def test_pooler_user_hint() -> None:
    settings = prepare("postgresql://postgres@aws-0-x.pooler.supabase.com:6543/postgres")
    hint = _hint_for(settings, 'FATAL: password authentication failed for user "postgres"')
    assert hint is not None and "postgres.<project-ref>" in hint
