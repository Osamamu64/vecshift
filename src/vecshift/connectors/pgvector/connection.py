"""Connecting to PostgreSQL safely, with Supabase's connection modes in mind.

Supabase offers a direct connection and Supavisor poolers. The ones that matter here:

- ``db.<ref>.supabase.co:5432``: direct. IPv6 only unless the project has the IPv4 add-on.
- ``*.pooler.supabase.com:5432``: session pooler. Works over IPv4.
- ``*.pooler.supabase.com:6543`` and ``db.<ref>.supabase.co:6543``: transaction poolers,
  which don't support prepared statements.

VecShift always disables prepared statements and runs inside one read-only transaction,
so every mode works.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

APPLICATION_NAME = "vecshift"
_TRANSACTION_POOLER_PORT = "6543"


class ConnectError(Exception):
    """A connection failed. The message is safe to show: it never includes a password."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class SupabaseMode(StrEnum):
    DIRECT = "direct"
    SESSION_POOLER = "session pooler"
    TRANSACTION_POOLER = "transaction pooler"


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    """A validated connection string plus what we learned about where it points."""

    conninfo: str
    display: str
    """The connection string with the password removed, safe to print."""
    supabase: SupabaseMode | None

    @property
    def description(self) -> str:
        return f"Supabase {self.supabase}" if self.supabase else "PostgreSQL"


def _supabase_mode(host: str, port: str) -> SupabaseMode | None:
    host = host.lower().rstrip(".")
    if host.endswith(".pooler.supabase.com"):
        if port == _TRANSACTION_POOLER_PORT:
            return SupabaseMode.TRANSACTION_POOLER
        return SupabaseMode.SESSION_POOLER
    if host.endswith(".supabase.co"):
        if port == _TRANSACTION_POOLER_PORT:
            return SupabaseMode.TRANSACTION_POOLER
        return SupabaseMode.DIRECT
    return None


def prepare(dsn: str) -> ConnectionSettings:
    """Parse and harden a connection string (a ``postgresql://`` URL or ``key=value`` pairs).

    For Supabase hosts, TLS is required unless the string already sets ``sslmode``.
    """
    try:
        params = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError as exc:
        raise ConnectError(
            "That doesn't look like a valid PostgreSQL connection string.",
            hint="Expected something like postgresql://user:password@host:5432/dbname",
        ) from exc

    host = str(params.get("host") or "")
    port = str(params.get("port") or "5432")
    mode = _supabase_mode(host, port)
    if mode is not None:
        params.setdefault("sslmode", "require")
    params.setdefault("application_name", APPLICATION_NAME)
    params.setdefault("connect_timeout", "10")

    shown = {k: v for k, v in params.items() if k == "user" or k == "dbname"}
    user = f"{shown['user']}@" if shown.get("user") else ""
    display = f"postgresql://{user}{host or 'localhost'}:{port}/{shown.get('dbname', '')}"

    return ConnectionSettings(conninfo=make_conninfo("", **params), display=display, supabase=mode)


def _hint_for(settings: ConnectionSettings, error: str) -> str | None:
    lowered = error.lower()
    if settings.supabase is SupabaseMode.DIRECT and any(
        s in lowered
        for s in ("network is unreachable", "cannot assign requested address", "translate host")
    ):
        return (
            "Supabase's direct connection is IPv6 only unless your project has the IPv4 "
            "add-on. Use the session pooler string from the dashboard's Connect button "
            "instead (host ending in .pooler.supabase.com, port 5432)."
        )
    if "password authentication failed" in lowered and settings.supabase in (
        SupabaseMode.SESSION_POOLER,
        SupabaseMode.TRANSACTION_POOLER,
    ):
        return (
            "Supabase poolers expect the user as postgres.<project-ref>, not just postgres. "
            "Copy the exact string from the dashboard's Connect button."
        )
    if "timeout" in lowered:
        return "Check the host and port, and that your network allows outbound connections."
    return None


def connect(settings: ConnectionSettings, statement_timeout_s: int = 60) -> psycopg.Connection:
    """Open a read-only connection with a statement timeout.

    Prepared statements are disabled so transaction poolers work. The caller should
    roll back and close the connection when done.
    """
    try:
        conn = psycopg.connect(settings.conninfo, prepare_threshold=None)
    except psycopg.OperationalError as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else "Connection failed."
        raise ConnectError(message, hint=_hint_for(settings, str(exc))) from exc

    conn.read_only = True
    # SET LOCAL lasts for the current transaction only, which also suits transaction poolers.
    conn.execute(
        sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(statement_timeout_s * 1000))
    )
    return conn
