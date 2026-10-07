"""Model specs: one string that says which model to call and how.

    provider/model[,key=value,...]

Examples:

    openai/text-embedding-3-small
    openai/text-embedding-3-large,dims=256
    ollama/nomic-embed-text
    compat/BAAI/bge-m3,url=http://localhost:8080/v1,price=0
    hash/512

Options: ``url``, ``dims``, ``price`` (USD per million tokens), ``key_env`` (the
environment variable holding the API key), ``query_prefix``, ``doc_prefix``, and
``batch`` (texts per request).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


class SpecError(ValueError):
    """A model spec couldn't be understood."""


@dataclass(frozen=True, slots=True)
class Preset:
    url: str | None
    key_env: str | None
    note: str


PRESETS = {
    "openai": Preset("https://api.openai.com/v1", "OPENAI_API_KEY", "OpenAI"),
    "ollama": Preset("http://localhost:11434/v1", None, "Ollama on this machine"),
    "compat": Preset(None, "VECSHIFT_API_KEY", "any OpenAI-compatible server"),
    "hash": Preset(None, None, "built-in hashing baseline, free and offline"),
}

# USD per million input tokens for OpenAI's embedding models. Prices change; pass
# price= in the spec to override.
KNOWN_PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# Native output size of well-known models, so plans know the target column size without
# calling the API. Matched on the model's base name, without any path prefix or tag.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "all-minilm": 384,
}

# Models known to accept a smaller ``dimensions`` request (Matryoshka truncation).
SUPPORTS_DIMENSIONS = {"text-embedding-3-small", "text-embedding-3-large"}

_SEARCH_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# (pattern, query prefix, document prefix) for models trained with prompts. Using the
# wrong prefix quietly costs these models a lot of quality.
KNOWN_PREFIXES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"nomic-embed-text"), "search_query: ", "search_document: "),
    (re.compile(r"(^|[/_-])e5-"), "query: ", "passage: "),
    (re.compile(r"bge-(small|base|large)-(en|zh)"), _SEARCH_INSTRUCTION, ""),
    (re.compile(r"mxbai-embed-large"), _SEARCH_INSTRUCTION, ""),
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    raw: str
    provider: str
    model: str
    url: str | None = None
    dimensions: int | None = None
    price: float | None = None
    """USD per million tokens, or ``None`` if unknown."""
    key_env: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    batch_size: int = 64

    @property
    def name(self) -> str:
        """Short display name, such as ``openai/text-embedding-3-large (256d)``."""
        base = f"{self.provider}/{self.model}"
        explicit = self.dimensions and self.provider != "hash"
        return f"{base} ({self.dimensions}d)" if explicit else base

    @property
    def is_local(self) -> bool:
        """Whether text stays on this machine."""
        if self.provider == "hash":
            return True
        host = urlparse(self.url or "").hostname or ""
        return host in LOCAL_HOSTS

    @property
    def base_model(self) -> str:
        """``BAAI/bge-m3`` → ``bge-m3``; ``nomic-embed-text:latest`` → ``nomic-embed-text``."""
        return self.model.rsplit("/", 1)[-1].split(":", 1)[0].lower()

    @property
    def native_dimensions(self) -> int | None:
        return KNOWN_DIMENSIONS.get(self.base_model)

    @property
    def output_dimensions(self) -> int | None:
        """What this spec will produce, if known without calling the model."""
        return self.dimensions or self.native_dimensions

    @property
    def identity(self) -> str:
        """Everything that changes the vectors this spec produces, for caching."""
        return "|".join(
            str(x)
            for x in (
                self.provider,
                self.model,
                self.url,
                self.dimensions,
                self.query_prefix,
                self.doc_prefix,
            )
        )


def _number(key: str, value: str, kind: type[int] | type[float]) -> float:
    try:
        number = kind(value)
    except ValueError:
        raise SpecError(f"{key}= must be a number, got {value!r}") from None
    if number < 0 or (kind is int and number == 0):
        raise SpecError(f"{key}= must be positive, got {value!r}")
    return number


def parse_spec(raw: str) -> ModelSpec:
    """Parse ``provider/model[,key=value,...]``."""
    head, *options = [part.strip() for part in raw.split(",")]
    provider, sep, model = head.partition("/")
    provider = provider.lower()
    if not sep or not model:
        raise SpecError(
            f"Expected provider/model, got {raw!r}. For example: openai/text-embedding-3-small"
        )
    if provider not in PRESETS:
        raise SpecError(f"Unknown provider {provider!r}. Use one of: {', '.join(sorted(PRESETS))}.")
    preset = PRESETS[provider]
    values: dict[str, str] = {}
    for option in options:
        key, eq, value = option.partition("=")
        key = key.strip().lower()
        if not eq or key not in {
            "url",
            "dims",
            "price",
            "key_env",
            "query_prefix",
            "doc_prefix",
            "batch",
        }:
            raise SpecError(f"Unknown option {option!r} in {raw!r}")
        values[key] = value

    url = values.get("url", preset.url)
    if provider == "compat" and not url:
        raise SpecError(
            f"compat models need a url=, for example {raw},url=http://localhost:8080/v1"
        )
    if url:
        url = url.rstrip("/")
        if urlparse(url).scheme not in {"http", "https"}:
            raise SpecError(f"url= must start with http:// or https://, got {url!r}")

    dimensions = int(_number("dims", values["dims"], int)) if "dims" in values else None
    if provider == "hash":
        dimensions = int(_number("hash dimensions", model, int))

    price: float | None
    if "price" in values:
        price = float(_number("price", values["price"], float))
    elif provider == "openai":
        price = KNOWN_PRICES.get(model)
    else:
        price = 0.0 if provider in {"hash", "ollama"} else None

    query_prefix, doc_prefix = "", ""
    for pattern, qp, dp in KNOWN_PREFIXES:
        if pattern.search(model.lower()):
            query_prefix, doc_prefix = qp, dp
            break
    query_prefix = values.get("query_prefix", query_prefix)
    doc_prefix = values.get("doc_prefix", doc_prefix)

    batch = int(_number("batch", values["batch"], int)) if "batch" in values else 64
    return ModelSpec(
        raw=raw,
        provider=provider,
        model=model,
        url=url,
        dimensions=dimensions,
        price=price,
        key_env=values.get("key_env", preset.key_env),
        query_prefix=query_prefix,
        doc_prefix=doc_prefix,
        batch_size=batch,
    )
