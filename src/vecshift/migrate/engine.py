"""The apply loop: fill the new column, keep up with writes, then index and verify.

It talks to the database through a small writer interface and to the model through the
embedding contract, so it can be tested with fakes and reused for other stores.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from vecshift.embeddings.providers import EmbeddingError
from vecshift.migrate.state import JobState

CHARS_PER_TOKEN = 4
MAX_PASSES = 5
"""Catch-up passes after the first, for rows that changed or arrived during the run."""
MAX_CHARS = 24_000
"""Longer texts are cut to this length, inside common model input limits (about 6K tokens)."""

Status = Literal["complete", "stopped", "budget", "failed"]


class Row(Protocol):
    @property
    def key(self) -> Any: ...
    @property
    def text(self) -> str: ...
    @property
    def id(self) -> str: ...


class Writer(Protocol):
    def acquire(self) -> None: ...
    def release(self) -> None: ...
    def ensure_column(self) -> bool: ...
    def ensure_trigger(self) -> bool: ...
    def pending_count(self, exclude: Sequence[str] = ()) -> int: ...
    def fetch(self, after: Any, limit: int, exclude: Sequence[str] = ()) -> Sequence[Row]: ...
    def write(self, items: Sequence[tuple[Any, Sequence[float]]]) -> int: ...
    def build_index(self, method: str, metric: str, rows: int) -> str: ...
    def dimensions_in_use(self) -> set[int]: ...


class Embedder(Protocol):
    tokens: int

    async def embed(
        self, texts: Sequence[str], mode: Literal["document", "query"] = ...
    ) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    """``start``, ``column``, ``trigger``, ``batch``, ``pass``, ``index``, or ``done``."""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApplyResult:
    status: Status
    rows_written: int = 0
    rows_failed: int = 0
    remaining: int = 0
    tokens: int = 0
    spent_usd: float | None = 0.0
    """This run's spend, or ``None`` when the model has no price."""
    total_spent_usd: float | None = 0.0
    seconds: float = 0.0
    index: str = "pending"
    """``built``, ``rebuilt``, ``exists``, ``skipped``, or ``pending``."""
    message: str | None = None


class DimensionMismatch(Exception):
    pass


async def _embed_isolating(
    embedder: Embedder, rows: Sequence[Row], dims: int
) -> tuple[list[tuple[Row, list[float]]], dict[str, str]]:
    """Embed rows; if the provider rejects a request, split it to find the bad rows."""
    texts = [r.text[:MAX_CHARS] for r in rows]
    try:
        vectors = await embedder.embed(texts, "document")
    except EmbeddingError as exc:
        if len(rows) == 1:
            return [], {rows[0].id: str(exc)}
        mid = len(rows) // 2
        left, lf = await _embed_isolating(embedder, rows[:mid], dims)
        right, rf = await _embed_isolating(embedder, rows[mid:], dims)
        return left + right, {**lf, **rf}
    for vector in vectors:
        if len(vector) != dims:
            raise DimensionMismatch(
                f"The model returned {len(vector)} dimensions, but the column holds {dims}."
            )
    return list(zip(rows, vectors, strict=True)), {}


async def apply(
    writer: Writer,
    embedder: Embedder,
    state: JobState,
    *,
    dims: int,
    price_per_million: float | None,
    budget_usd: float | None,
    chunk_rows: int,
    index: tuple[str, str] | None,
    on_event: Callable[[Event], None] = lambda e: None,
    should_stop: Callable[[], bool] = lambda: False,
) -> ApplyResult:
    """Run (or resume) a migration. Safe to run again at any time to catch up."""
    started = time.monotonic()
    result = ApplyResult(status="complete")
    tokens_before = embedder.tokens
    spent_before = state.spent_usd
    state.runs += 1

    def spent() -> float:
        if price_per_million is None:
            return 0.0
        return (embedder.tokens - tokens_before) * price_per_million / 1_000_000

    def finish(status: Status, message: str | None = None) -> ApplyResult:
        result.status = status
        result.message = message
        result.tokens = embedder.tokens - tokens_before
        run_spend = spent()
        result.spent_usd = run_spend if price_per_million is not None else None
        state.spent_usd = spent_before + run_spend
        state.tokens += result.tokens
        state.save()
        result.total_spent_usd = state.spent_usd if price_per_million is not None else None
        result.rows_failed = len(state.failed)
        result.seconds = time.monotonic() - started
        on_event(Event("done", {"status": status, "message": message}))
        return result

    writer.acquire()
    try:
        # The column has to exist before rows can be counted against it.
        on_event(Event("column", {"added": writer.ensure_column()}))
        on_event(Event("trigger", {"added": writer.ensure_trigger()}))
        pending = writer.pending_count(tuple(state.failed))
        on_event(Event("start", {"pending": pending, "spent_before": spent_before}))

        for number in range(1, MAX_PASSES + 2):
            after: Any = None
            progress = 0
            while True:
                if should_stop():
                    result.remaining = writer.pending_count(tuple(state.failed))
                    return finish("stopped", "Stopped. Run apply again to continue.")
                rows = writer.fetch(after, chunk_rows, tuple(state.failed))
                if not rows:
                    break
                after = rows[-1].key
                if budget_usd is not None and price_per_million is not None:
                    estimate = sum(len(r.text[:MAX_CHARS]) for r in rows) / CHARS_PER_TOKEN
                    projected = spent_before + spent() + estimate * price_per_million / 1e6
                    if projected > budget_usd:
                        result.remaining = writer.pending_count(tuple(state.failed))
                        return finish(
                            "budget",
                            f"Stopped before the next batch would pass the ${budget_usd:,.2f} "
                            "budget. Raise limits.budget_usd and run apply again to continue.",
                        )
                try:
                    pairs, failed = await _embed_isolating(embedder, rows, dims)
                except DimensionMismatch as exc:
                    return finish("failed", str(exc))
                except EmbeddingError as exc:  # pragma: no cover - isolation catches these
                    return finish("failed", str(exc))
                state.failed.update(failed)
                written = writer.write(pairs)
                progress += written
                result.rows_written += written
                state.rows_written += written
                state.spent_usd = spent_before + spent()
                state.save()
                on_event(
                    Event(
                        "batch",
                        {
                            "pass": number,
                            "written": written,
                            "skipped": len(pairs) - written,
                            "failed": len(failed),
                            "rows_written": result.rows_written,
                            "spent_usd": state.spent_usd if price_per_million is not None else None,
                        },
                    )
                )
            remaining = writer.pending_count(tuple(state.failed))
            on_event(Event("pass", {"number": number, "remaining": remaining}))
            result.remaining = remaining
            if remaining == 0 or progress == 0:
                break

        if result.remaining:
            return finish(
                "stopped",
                f"{result.remaining:,} rows kept changing during the run. Run apply again "
                "to catch up.",
            )
        if index is None:
            result.index = "skipped"
        else:
            on_event(Event("index", {"state": "building", "method": index[0]}))
            result.index = writer.build_index(index[0], index[1], state.rows_written)
            on_event(Event("index", {"state": result.index}))
        sizes = writer.dimensions_in_use()
        if sizes and sizes != {dims}:
            return finish("failed", f"Found vectors of sizes {sorted(sizes)} in the new column.")
        failed_note = (
            f" {len(state.failed):,} rows were rejected by the provider; see the state file."
            if state.failed
            else ""
        )
        return finish("complete", f"Every row with text has a new vector.{failed_note}")
    finally:
        writer.release()
