"""PostgreSQL with the pgvector extension, including Supabase."""

from vecshift.connectors.pgvector.connection import (
    ConnectError,
    ConnectionSettings,
    SupabaseMode,
    connect,
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
    "find_vector_columns",
    "inspect",
    "prepare",
]
