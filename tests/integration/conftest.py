"""Integration tests against a real PostgreSQL with pgvector.

Set VECSHIFT_TEST_PG_DSN to a superuser connection string to run them, for example:

    docker run -d -p 55432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
    VECSHIFT_TEST_PG_DSN=postgresql://postgres:postgres@localhost:55432/postgres uv run pytest

Each test gets a fresh database laid out like Supabase: pgvector is installed in an
``extensions`` schema that isn't on the search path.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ADMIN_DSN = os.environ.get("VECSHIFT_TEST_PG_DSN")

pytestmark = pytest.mark.integration


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "integration" in str(item.path):
            item.add_marker(pytest.mark.integration)
            if not ADMIN_DSN:
                item.add_marker(pytest.mark.skip(reason="VECSHIFT_TEST_PG_DSN is not set"))


@pytest.fixture
def db_dsn() -> Iterator[str]:
    assert ADMIN_DSN
    name = f"vecshift_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    dsn = make_conninfo("", **{**conninfo_to_dict(ADMIN_DSN), "dbname": name})
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE SCHEMA extensions")
        conn.execute("CREATE EXTENSION vector WITH SCHEMA extensions")
    try:
        yield dsn
    finally:
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture
def setup(db_dsn: str) -> Iterator[psycopg.Connection]:
    """An autocommit superuser connection for arranging test data."""
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
def reader(db_dsn: str, setup: psycopg.Connection) -> Iterator[str]:
    """A login role without BYPASSRLS, like a Supabase app role."""
    role = f"reader_{uuid.uuid4().hex[:8]}"
    setup.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'reader'")
    setup.execute(f"GRANT USAGE ON SCHEMA extensions TO {role}")
    try:
        yield make_conninfo("", **{**conninfo_to_dict(db_dsn), "user": role, "password": "reader"})
    finally:
        setup.execute(f"DROP OWNED BY {role}")
        setup.execute(f"DROP ROLE {role}")
