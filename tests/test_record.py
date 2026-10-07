from datetime import UTC, datetime

import pytest

from vecshift import Record, SparseVector

EARLY = datetime(2026, 1, 1, tzinfo=UTC)
LATE = datetime(2026, 6, 1, tzinfo=UTC)


def test_defaults_are_independent_per_record() -> None:
    a, b = Record(id="a"), Record(id="b")
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_can_reembed_requires_text() -> None:
    assert Record(id="1", text="hello").can_reembed
    assert not Record(id="1").can_reembed
    assert not Record(id="1", text="").can_reembed


def test_tombstone() -> None:
    t = Record.tombstone("1", updated_at=LATE)
    assert t.deleted and t.id == "1" and t.updated_at == LATE


@pytest.mark.parametrize(
    ("mine", "theirs", "expected"),
    [
        (LATE, EARLY, True),
        (EARLY, LATE, False),
        (EARLY, EARLY, False),
        (EARLY, None, True),
        (None, EARLY, False),
        (None, None, False),
    ],
)
def test_is_newer_than(mine: datetime | None, theirs: datetime | None, expected: bool) -> None:
    assert (
        Record(id="1", updated_at=mine).is_newer_than(Record(id="1", updated_at=theirs)) is expected
    )


def test_sparse_vector_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="equal length"):
        SparseVector(indices=(1, 2), values=(0.5,))
