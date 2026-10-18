"""``cutover``, ``rollback``, and ``cleanup`` for migrations into Qdrant.

The commands in ``cli_cutover`` hand Qdrant jobs to these. Cutover catches up with the
source, then points the application's alias at the new collection in one atomic request;
rollback points it back; cleanup deletes the old collection once you're sure.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from vecshift import ui
from vecshift.cli_style import fail as _fail
from vecshift.cli_style import finding_lines
from vecshift.service.errors import first_line as _first_line
from vecshift.service.qdrant import Located, history, locate
from vecshift.service.qdrant import record as _record

CATCH_UP_TRIES = 3


def _guarded(found: Located, run: Any) -> Any:
    """Turn Qdrant, database, and lock problems into clean messages; always close."""
    import psycopg

    from vecshift.connectors.qdrant import QdrantError
    from vecshift.connectors.qdrant.writer import AlreadyRunning
    from vecshift.embeddings import EmbeddingError

    try:
        return run()
    except AlreadyRunning as exc:
        raise _fail(str(exc), "Wait for it to finish, or stop it first.") from exc
    except (QdrantError, EmbeddingError) as exc:
        raise _fail(str(exc), getattr(exc, "hint", None) or "Nothing was switched.") from exc
    except psycopg.Error as exc:
        raise _fail(
            f"The database reported an error: {_first_line(exc)}", "Nothing was switched."
        ) from exc
    finally:
        found.close()


def _catch_up(found: Located) -> tuple[int, int]:
    """Embed what changed in the source since the last apply, and remove what was deleted.

    Returns (points written, points still missing a vector)."""
    from vecshift.cli_apply import _run
    from vecshift.embeddings import create_embedder
    from vecshift.embeddings.providers import CONCURRENCY

    spec = found.spec
    result = asyncio.run(
        _run(
            found.writer,
            create_embedder(spec),
            found.state,
            dims=found.writer.layout.dims,
            price_per_million=spec.price,
            budget_usd=found.job.limits.budget_usd,
            chunk_rows=spec.batch_size * CONCURRENCY,
            index=None,
        )
    )
    if result.status not in {"complete", "stopped"}:
        raise _fail(f"The final catch-up stopped: {result.message}", "Nothing was switched.")
    found.writer.remove_deleted()
    # Count again after deleting: the switch must never leave searches short of points.
    return result.rows_written, found.writer.pending_count(tuple(found.state.failed))


def _differences(found: Located, source: str, target: str) -> int:
    """Points of ``source`` missing from ``target`` or with different text there. Read-only."""
    from vecshift.connectors.qdrant.writer import Layout, QdrantSource, QdrantWriter

    lay = found.writer.layout
    reader = QdrantWriter(
        found.client,
        QdrantSource(found.client, source, found.text_field, None),
        Layout(target, lay.vector, lay.dims, lay.distance),
        read_only=True,
    )
    return reader.pending_count()


def cutover(
    job_file: Path, *, check: bool, yes: bool, allow_missing: bool, output_json: bool
) -> None:
    from vecshift.cli_cutover import _confirm

    found = locate(job_file)
    state = found.state

    def run() -> dict[str, Any]:
        found.writer.acquire()
        try:
            ready = found.switch.check()
            blocked = any(
                f.severity.rank == 3 and f.id != "cutover.pending" for f in ready.findings
            )
            if check or blocked:
                return {"status": "ready" if ready.ok else "blocked", "readiness": ready}
            if not output_json:
                ui.title("cutover", found.job.name)
                finding_lines([f for f in ready.findings if f.severity.rank >= 1])
            where = "your machine" if found.spec.is_local else found.spec.url
            sends = (
                f" First it embeds {ready.pending:,} points that changed since the last "
                f"apply, sending their text to {where}."
                if ready.pending
                else ""
            )
            now = f" (now {found.previous})" if found.previous else ""
            _confirm(
                f"\nCutover points the alias {found.alias}{now} at {found.new}, so searches "
                f"use {found.spec.name} vectors.{sends} Switch your application to "
                f"{found.spec.name} at the same time.",
                yes,
            )
            caught_up = 0
            for _ in range(CATCH_UP_TRIES):
                written, remaining = _catch_up(found)
                caught_up += written
                if not remaining:
                    break
            else:
                raise _fail(
                    f"{remaining:,} points still have no new vector.",
                    "Points keep changing; try again when writes are quieter.",
                )
            if state.failed and not allow_missing:
                raise _fail(
                    f"The provider rejected {len(state.failed):,} points, which would be "
                    "missing from the new collection.",
                    "Fix their text, or pass --allow-missing.",
                )
            found.switch.point(found.new)
            _record(found, "cutover")
            late = (
                _differences(found, found.previous, found.new)
                if found.from_qdrant and found.same_server and found.previous
                else 0
            )
            return {"status": "cut_over", "readiness": ready, "caught_up": caught_up, "late": late}
        finally:
            found.writer.release()

    outcome = _guarded(found, run)
    ready = outcome["readiness"]
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": outcome["status"],
                    "alias": found.alias,
                    "previous": found.previous,
                    "new": found.new,
                    "pending": ready.pending,
                    "caught_up": outcome.get("caught_up", 0),
                    "late": outcome.get("late", 0),
                    "findings": [f.to_dict() for f in ready.findings],
                }
            )
        )
    elif outcome["status"] in {"ready", "blocked"}:
        ui.title("cutover", found.job.name)
        finding_lines(ready.findings)
        if ready.ok:
            ui.verdict("Ready to cut over.")
            if ui.fancy():
                ui.next_step("vecshift cutover")
        else:
            ui.verdict("Not ready to cut over.", "error")
    else:
        ui.verdict("Cut over.")
        ui.detail(f"The alias {found.alias} now points at {found.new}.")
        if outcome.get("caught_up"):
            ui.detail(f"Embedded {outcome['caught_up']:,} late points first.")
        if outcome.get("late"):
            ui.detail(
                f"{outcome['late']:,} points of {found.previous} differ from {found.new}: "
                "edits made in the moments before the switch, or written to the old "
                "collection by name. Check them."
            )
        undo = (
            f"Undo with: vecshift rollback. When you're sure, delete {found.previous} with "
            "vecshift cleanup."
            if found.previous
            else "The source is unchanged."
        )
        ui.next_step(
            f"make sure your application embeds queries and new points with "
            f"{found.spec.name}.\n{undo}"
        )
    if outcome["status"] == "blocked":
        raise typer.Exit(1)


def rollback(job_file: Path, *, yes: bool, output_json: bool) -> None:
    from vecshift.cli_cutover import _confirm

    found = locate(job_file)

    def run() -> int:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise _fail(
                    "There's no cutover to roll back.",
                    f"The alias {found.alias or '(none)'} doesn't point at {found.new}.",
                )
            previous = found.previous
            recorded = [
                h
                for h in history(found.state, found.alias, found.new)
                if h.get("event") == "cutover"
            ]
            if not previous and not recorded:
                raise _fail(
                    "This folder has no record of the cutover, so vecshift doesn't know "
                    "which collection to point the alias back at.",
                    "Run rollback from the folder where you ran cutover (its record is in "
                    ".vecshift/), or move the alias yourself.",
                )
            if not previous:
                raise _fail(
                    "Before cutover the alias didn't exist, so there's nothing to point it "
                    "back at.",
                    "To stop using Qdrant, point your application back at its previous "
                    "store; the source was never changed.",
                )
            if not found.client.exists(previous):
                raise _fail(f"The old collection {previous} no longer exists.")
            _confirm(
                f"Rollback points the alias {found.alias} back at {previous}, so searches use "
                "the old vectors again. Switch your application back to the old model at "
                "the same time.",
                yes,
            )
            found.switch.point(previous)
            _record(found, "rollback")
            if found.from_qdrant and found.same_server:
                return _differences(found, found.new, previous)
            return 0
        finally:
            found.writer.release()

    changed = _guarded(found, run)
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "rolled_back",
                    "alias": found.alias,
                    "live": found.previous,
                    "new": found.new,
                    "changed": changed,
                }
            )
        )
        return
    if ui.fancy():
        ui.title("rollback", found.job.name)
    ui.verdict("Rolled back.")
    ui.detail(
        f"The alias {found.alias} points at {found.previous} again; {found.new} is kept for "
        "another cutover."
    )
    if changed:
        ui.detail(
            f"{changed:,} points were added or edited in {found.new} after cutover and are "
            f"missing or different in {found.previous}. Your application needs to write them "
            "again, embedded with the old model."
        )
    if ui.fancy():
        ui.console.print()


def cleanup(job_file: Path, *, yes: bool, output_json: bool) -> None:
    import sys

    found = locate(job_file)
    previous = found.previous

    def run() -> None:
        found.writer.acquire()
        try:
            if found.switch.stage() != "cut_over":
                raise _fail(
                    "There's no cutover to clean up after.",
                    "Cleanup deletes the old collection once the alias points at the new one.",
                )
            if not previous:
                raise _fail(
                    "vecshift doesn't know of an old collection to delete: either the alias "
                    "didn't exist before cutover, or this folder has no record of it.",
                    "Delete the old collection yourself when you're sure. A pgvector source "
                    "table is untouched; drop it yourself too.",
                )
            if previous == found.new or previous in dict(found.client.aliases()).values():
                raise _fail(f"{previous} is still in use, so it won't be deleted.")
            if not found.client.exists(previous):
                raise _fail(f"{previous} is already gone.")
            if not yes:
                warning = (
                    f"Cleanup deletes the collection {previous} and every point in it. The "
                    "old vectors are gone for good, and rollback is no longer possible."
                )
                if ui.fancy():
                    ui.warn(warning)
                    typed = ui.text(f"Type {previous} to confirm")
                else:
                    typer.echo(warning, err=True)
                    if not sys.stdin.isatty():
                        raise _fail("Not deleting anything without confirmation.", "Pass --yes.")
                    typed = typer.prompt(f"Type {previous} to confirm", err=True, default="")
                if typed != previous:
                    raise _fail("That didn't match, so nothing was deleted.")
            found.client.delete_collection(previous)
            _record(found, "cleanup")
        finally:
            found.writer.release()

    _guarded(found, run)
    if output_json:
        typer.echo(json.dumps({"status": "cleaned_up", "deleted": previous}))
    else:
        if ui.fancy():
            ui.title("cleanup", found.job.name)
        ui.verdict(f"Deleted {previous}. The migration is complete.")
        if ui.fancy():
            ui.console.print()
