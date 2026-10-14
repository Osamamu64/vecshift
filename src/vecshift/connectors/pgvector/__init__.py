"""PostgreSQL with the pgvector extension, including Supabase."""

from vecshift.connectors.pgvector.connection import (
    ConnectError,
    ConnectionSettings,
    SupabaseMode,
    connect,
    connect_reader,
    connect_writer,
    prepare,
)
from vecshift.connectors.pgvector.inspect import (
    TargetSelectionError,
    VectorColumn,
    find_vector_columns,
    inspect,
)

__all__ = [
    "ConnectError",
    "ConnectionSettings",
    "SupabaseMode",
    "TargetSelectionError",
    "VectorColumn",
    "connect",
    "connect_reader",
    "connect_writer",
    "find_vector_columns",
    "inspect",
    "prepare",
]
