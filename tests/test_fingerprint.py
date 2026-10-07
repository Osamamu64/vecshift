from dataclasses import replace

import pytest

from vecshift import EmbeddingFingerprint

BASE = EmbeddingFingerprint(provider="openai", model="text-embedding-3-small", dimensions=1536)


def test_model_tag_is_readable_and_stable() -> None:
    tag = BASE.model_tag
    assert tag.startswith("openai/text-embedding-3-small@1536#")
    assert (
        tag
        == EmbeddingFingerprint(
            provider="openai", model="text-embedding-3-small", dimensions=1536
        ).model_tag
    )


def test_provider_and_model_casing_and_whitespace_do_not_matter() -> None:
    other = EmbeddingFingerprint(
        provider=" OpenAI ", model="Text-Embedding-3-Small", dimensions=1536
    )
    assert other.model_tag == BASE.model_tag
    assert other.is_compatible_with(BASE)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "text-embedding-3-large"},
        {"dimensions": 512},
        {"version": "2"},
        {"task": "retrieval_query"},
        {"prefix": "passage: "},
        {"normalized": False},
    ],
)
def test_any_field_change_is_a_new_vector_space(change: dict[str, object]) -> None:
    other = replace(BASE, **change)  # type: ignore[arg-type]
    assert other.digest != BASE.digest
    assert not other.is_compatible_with(BASE)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"provider": " ", "model": "m", "dimensions": 8}, "provider"),
        ({"provider": "p", "model": "", "dimensions": 8}, "model"),
        ({"provider": "p", "model": "m", "dimensions": 0}, "dimensions"),
    ],
)
def test_invalid_fingerprints_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingFingerprint(**kwargs)  # type: ignore[arg-type]
