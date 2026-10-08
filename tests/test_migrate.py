"""The apply loop and its state file, with an in-memory table and a fake model."""

import asyncio
import json
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vecshift.embeddings import EmbeddingError
from vecshift.migrate import ApplyResult, Event, JobState, apply
from vecshift.migrate.engine import MAX_CHARS


@dataclass(frozen=True)
class FakeRow:
    key: int
    text: str

    @property
    def id(self) -> str:
        return str(self.key)


class FakeTable:
    """Acts like PgWriter over a dict of id -> [text, vector]."""

    def __init__(self, rows: int) -> None:
        self.rows: dict[int, list[Any]] = {i: [f"text {i}", None] for i in range(1, rows + 1)}
        self.locked = False
        self.column = False
        self.trigger = False
        self.index: str | None = None
        self.on_fetch: Any = None

    def acquire(self) -> None:
        assert not self.locked
        self.locked = True

    def release(self) -> None:
        self.locked = False

    def ensure_column(self) -> bool:
        added, self.column = not self.column, True
        return added

    def ensure_trigger(self) -> bool:
        added, self.trigger = not self.trigger, True
        return added

    def edit(self, key: int, text: str) -> None:
        """An application update; the trigger clears the vector."""
        self.rows[key] = [text, None]

    def _pending(self, exclude: Sequence[str]) -> list[int]:
        return [
            k
            for k, (text, vec) in sorted(self.rows.items())
            if vec is None and text.strip() and str(k) not in exclude
        ]

    def pending_count(self, exclude: Sequence[str] = ()) -> int:
        return len(self._pending(exclude))

    def fetch(self, after: Any, limit: int, exclude: Sequence[str] = ()) -> list[FakeRow]:
        keys = [k for k in self._pending(exclude) if after is None or k > after][:limit]
        rows = [FakeRow(k, self.rows[k][0]) for k in keys]
        if self.on_fetch:
            self.on_fetch(rows)
        return rows

    def write(self, items: Sequence[tuple[FakeRow, Sequence[float]]]) -> int:
        written = 0
        for row, vector in items:
            text, current = self.rows[row.key]
            if current is None and text == row.text:
                self.rows[row.key][1] = list(vector)
                written += 1
        return written

    def build_index(self, method: str, metric: str, rows: int) -> str:
        if self.index:
            return "exists"
        self.index = f"{method}/{metric}"
        return "built"

    def dimensions_in_use(self) -> set[int]:
        return {len(v) for _, v in self.rows.values() if v is not None}


class FakeModel:
    def __init__(self, dims: int = 4, reject: str | None = None) -> None:
        self.dims = dims
        self.reject = reject
        self.tokens = 0
        self.seen: list[str] = []

    async def embed(self, texts: Sequence[str], mode: str = "document") -> list[list[float]]:
        if self.reject and any(self.reject in t for t in texts):
            raise EmbeddingError("input rejected")
        self.seen.extend(texts)
        self.tokens += sum(len(t) for t in texts) // 4 or 1
        return [[float(len(t))] * self.dims for t in texts]


def run(
    table: FakeTable, model: FakeModel, state: JobState, **kwargs: Any
) -> tuple[ApplyResult, list[Event]]:
    events: list[Event] = []
    options: dict[str, Any] = {
        "dims": 4,
        "price_per_million": 1.0,
        "budget_usd": None,
        "chunk_rows": 10,
        "index": ("hnsw", "cosine"),
        "on_event": events.append,
    }
    options.update(kwargs)
    return asyncio.run(apply(table, model, state, **options)), events


@pytest.fixture
def state(tmp_path: Path) -> JobState:
    return JobState.for_job(tmp_path / "vecshift.yaml", "docs")


def test_fills_every_row_and_builds_the_index(state: JobState) -> None:
    table = FakeTable(35)
    table.rows[7][0] = "   "  # blank text gets no vector
    result, events = run(table, FakeModel(), state)
    assert result.status == "complete" and result.rows_written == 34
    assert all(v is not None for k, (_, v) in table.rows.items() if k != 7)
    assert table.index == "hnsw/cosine" and result.index == "built"
    assert not table.locked, "the lock is released"
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["column", "trigger", "start"] and kinds[-1] == "done"
    assert events[2].data["pending"] == 34


def test_running_again_does_nothing(state: JobState) -> None:
    table, model = FakeTable(20), FakeModel()
    run(table, model, state)
    before = len(model.seen)
    result, _ = run(table, model, state)
    assert result.status == "complete" and result.rows_written == 0
    assert len(model.seen) == before and result.index == "exists"
    assert state.runs == 2


def test_edits_during_the_run_are_caught_up(state: JobState) -> None:
    table = FakeTable(30)
    edited = False

    def edit_once(rows: list[FakeRow]) -> None:
        nonlocal edited
        if not edited and rows and rows[0].key == 11:
            # Rows already done and rows about to be written change under the run.
            table.edit(3, "changed 3")
            table.edit(12, "changed 12")
            table.rows[31] = ["new row", None]
            edited = True

    table.on_fetch = edit_once
    result, events = run(table, FakeModel(), state)
    assert result.status == "complete"
    for text, vector in table.rows.values():
        assert vector == [float(len(text))] * 4, "every vector matches its current text"
    assert sum(e.kind == "pass" for e in events) == 2


def test_stop_and_resume(state: JobState) -> None:
    table, model = FakeTable(50), FakeModel()
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    result, _ = run(table, model, state, should_stop=stop)
    assert result.status == "stopped" and result.rows_written == 20 and result.remaining == 30
    assert table.index is None
    resumed = JobState.for_job(state.path.parent.parent / "vecshift.yaml", "docs")
    assert resumed.rows_written == 20 and resumed.runs == 1
    result, _ = run(table, model, resumed)
    assert result.status == "complete" and result.rows_written == 30
    assert len(model.seen) == 50, "no row is embedded twice"


def test_budget_stops_before_overspending(state: JobState) -> None:
    table = FakeTable(100)
    for row in table.rows.values():
        row[0] = "x" * 400  # about 100 tokens a row
    # $1 per million tokens; a 10-row batch is ~1000 tokens = $0.001.
    result, _ = run(table, FakeModel(), state, budget_usd=0.0035)
    assert result.status == "budget" and result.rows_written == 30
    assert result.total_spent_usd is not None and result.total_spent_usd <= 0.0035
    result, _ = run(table, FakeModel(), state, budget_usd=0.0035)
    assert result.status == "budget" and result.rows_written == 0, "spend counts across runs"
    result, _ = run(table, FakeModel(), state, budget_usd=1.0)
    assert result.status == "complete" and result.rows_written == 70
    assert state.spent_usd == pytest.approx(0.01)


def test_rejected_rows_are_isolated_and_recorded(state: JobState) -> None:
    table = FakeTable(25)
    table.rows[13][0] = "poison"
    result, _ = run(table, FakeModel(reject="poison"), state)
    assert result.status == "complete" and result.rows_written == 24 and result.rows_failed == 1
    assert set(state.failed) == {"13"} and "rejected" in state.failed["13"]
    assert table.rows[13][1] is None
    assert "rejected by the provider" in (result.message or "")
    saved = json.loads(state.path.read_text())
    assert saved["failed"] == {"13": "input rejected"}


def test_wrong_dimensions_fail_before_writing(state: JobState) -> None:
    table = FakeTable(5)
    result, _ = run(table, FakeModel(dims=3), state)
    assert result.status == "failed" and "3 dimensions" in (result.message or "")
    assert all(v is None for _, v in table.rows.values())
    assert not table.locked


def test_long_texts_are_clipped(state: JobState) -> None:
    table = FakeTable(1)
    table.rows[1][0] = "y" * (MAX_CHARS + 500)
    model = FakeModel()
    run(table, model, state)
    assert len(model.seen[0]) == MAX_CHARS
    assert table.rows[1][1] is not None


def test_no_index_and_no_price(state: JobState) -> None:
    result, _ = run(FakeTable(3), FakeModel(), state, index=None, price_per_million=None)
    assert result.status == "complete" and result.index == "skipped"
    assert result.spent_usd is None and result.total_spent_usd is None


def test_state_file_is_private_and_atomic(state: JobState) -> None:
    state.failed["1"] = "bad"
    state.save()
    assert stat.S_IMODE(os.stat(state.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(state.path.parent).st_mode) == 0o700
    assert not state.path.with_suffix(".tmp").exists()
    assert state.path.name == "docs.state.json"


def test_state_file_names_are_safe(tmp_path: Path) -> None:
    state = JobState.for_job(tmp_path / "j.yaml", "../../etc/passwd")
    assert state.path.parent == tmp_path / ".vecshift"
    assert "/" not in state.path.name and ".." not in state.path.name.removesuffix(".state.json")
