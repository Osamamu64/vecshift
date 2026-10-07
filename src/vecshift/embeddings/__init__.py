"""Embedding providers: turn text into vectors."""

from vecshift.embeddings.cache import CachedEmbedder, EmbeddingCache
from vecshift.embeddings.providers import (
    EmbeddingError,
    HashingEmbedder,
    OpenAICompatEmbedder,
    create_embedder,
)
from vecshift.embeddings.spec import ModelSpec, SpecError, parse_spec

__all__ = [
    "CachedEmbedder",
    "EmbeddingCache",
    "EmbeddingError",
    "HashingEmbedder",
    "ModelSpec",
    "OpenAICompatEmbedder",
    "SpecError",
    "create_embedder",
    "parse_spec",
]
