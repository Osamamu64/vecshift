"""The ``vecshift apply`` command."""

from __future__ import annotations

import json
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import DEFAULT_JOB, _money, _size
from vecshift.cli_style import finding_lines, reports_errors
from vecshift.service import ServiceError

if TYPE_CHECKING:
    from vecshift.jobs import JobSpec
    from vecshift.migrate import ApplyResult, Event, JobState
    from vecshift.planning import Plan
    from vecshift.service.apply import ApplyNotice

EXIT_STOPPED = 3
"""Stopped on purpose (Ctrl-C or budget): run apply again to continue."""


class _Stop:
    """First Ctrl-C finishes the current batch and stops cleanly; a second one aborts."""

    def __init__(self) -> None:
        self.requested = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        typer.secho(
            "\nStopping after the current batch. Press Ctrl-C again to abort now.",
            err=True,
            fg=typer.colors.YELLOW,
        )


class _Progress:
    def __init__(self, live: bool, noun: str = "column") -> None:
        self.live = live
        self.noun = noun
        self.started = time.monotonic()
        self.pending = 0
        self.last_print = 0.0

    def __call__(self, event: Event) -> None:
        d = event.data
        if event.kind == "start":
            self.pending = d["pending"]
            typer.echo(f"{self.pending:,} rows need a new vector.")
        elif event.kind == "column" and d["added"]:
            typer.echo(f"✔ {_added(self.noun)}.")
        elif event.kind == "trigger" and d["added"]:
            typer.echo("✔ Added the sync trigger.")
        elif event.kind == "batch":
            self._batch(d)
        elif event.kind == "pass" and d["number"] > 1 and self.live:
            typer.echo()
        elif event.kind == "index":
            if self.live:
                typer.echo()
            if d["state"] == "building":
                typer.echo(_building(self.noun, d["method"]))
            else:
                typer.echo(f"✔ {_indexed(self.noun, d['state'])}.")

    def _batch(self, d: dict[str, Any]) -> None:
        done = d["rows_written"]
        elapsed = max(time.monotonic() - self.started, 1e-6)
        rate = done / elapsed
        left = max(self.pending - done, 0)
        eta = f"~{_eta(left / rate)} left" if rate > 0 and left else ""
        cost = f" · {_money(d['spent_usd'])}" if d["spent_usd"] is not None else ""
        line = f"  {done:,} / {self.pending:,} rows · {rate:,.0f} rows/s{cost} {eta}"
        if self.live:
            typer.echo("\r" + line.ljust(78), nl=False)
        elif time.monotonic() - self.last_print > 15 or not left:
            typer.echo(line)
            self.last_print = time.monotonic()


class _FancyProgress:
    """A live progress bar and spinners, for a person at a colour terminal."""

    def __init__(self, noun: str = "column") -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        from vecshift import ui

        self.ui = ui
        self.noun = noun
        self.started = time.monotonic()
        self.progress = Progress(
            SpinnerColumn(style=ui.ACCENT),
            TextColumn("{task.description}"),
            BarColumn(complete_style=ui.ACCENT, finished_style="green", bar_width=32),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[rate]}"),
            TextColumn("[dim]{task.fields[cost]}"),
            TimeRemainingColumn(compact=True),
            console=ui.console,
            transient=False,
        )
        self.task: Any = None
        self.status: Any = None
        self.done = 0

    def __call__(self, event: Event) -> None:
        ui, d = self.ui, event.data
        if event.kind == "column" and d["added"]:
            ui.success(_added(self.noun))
        elif event.kind == "trigger" and d["added"]:
            ui.success("Added the sync trigger")
        elif event.kind == "start":
            ui.note(f"  {d['pending']:,} rows need a new vector")
            self.progress.start()
            self.task = self.progress.add_task("Embedding", total=d["pending"], rate="", cost="")
        elif event.kind == "batch" and self.task is not None:
            self.done = d["rows_written"]
            rate = self.done / max(time.monotonic() - self.started, 1e-6)
            cost = _money(d["spent_usd"]) if d["spent_usd"] is not None else ""
            self.progress.update(
                self.task, completed=self.done, rate=f"{rate:,.0f} rows/s", cost=cost
            )
        elif event.kind == "pass" and d["number"] > 1 and self.task is not None:
            self.progress.update(
                self.task, description="Catching up", total=self.done + d["remaining"]
            )
        elif event.kind == "index":
            self.close()
            if d["state"] == "building":
                self.status = ui.console.status(
                    _building(self.noun, d["method"]),
                    spinner="dots",
                    spinner_style=ui.ACCENT,
                )
                self.status.start()
            else:
                ui.success(_indexed(self.noun, d["state"]))
        elif event.kind == "done":
            self.close()

    def close(self) -> None:
        if self.task is not None:
            self.progress.stop()
            self.task = None
        if self.status is not None:
            self.status.stop()
            self.status = None


def _added(noun: str) -> str:
    return "Created the new collection" if noun == "collection" else "Added the new column"


def _building(noun: str, method: str) -> str:
    if noun == "collection":
        return "Waiting for Qdrant to finish indexing the new collection…"
    return f"Building the {method} index concurrently. Writes continue…"


def _indexed(noun: str, state: str) -> str:
    if noun != "collection":
        return f"Index {state}"
    if state == "pending":
        return "Qdrant is still indexing; vecshift status shows when it's done"
    return "Qdrant has indexed the new collection"


def _eta(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


QDRANT_NEXT = (
    "check progress with vecshift status. Run apply again any time to copy what changed in "
    "the source since."
)


def _summary_fancy(result: ApplyResult, into_qdrant: bool = False) -> None:
    from vecshift import ui

    kind = {"complete": "ok", "failed": "error"}.get(result.status, "warn")
    ui.verdict(f"Apply {result.status}.", kind)
    if result.message:
        ui.detail(result.message)
    spent = _money(result.spent_usd) if result.spent_usd is not None else "unknown"
    total = _money(result.total_spent_usd) if result.total_spent_usd is not None else ""
    ui.console.print()
    ui.kv(
        [
            ("Rows written", f"{result.rows_written:,}", ""),
            ("Tokens", f"{result.tokens:,}", ""),
            ("Spent", spent, f"{total} across runs" if total else ""),
            ("Time", f"{result.seconds:,.0f} s", ""),
        ]
        + (
            [("Rejected", f"{result.rows_failed:,} rows", "retried on the next run")]
            if result.rows_failed
            else []
        )
    )
    if result.status == "complete" and into_qdrant:
        ui.next_step(QDRANT_NEXT)
    elif result.status == "complete":
        ui.next_step(
            "check it's safe to switch with vecshift cutover --check, then run vecshift "
            "cutover. It embeds rows added or edited since this run first."
        )
    elif result.status in {"stopped", "budget"}:
        ui.next_step("run vecshift apply again to continue where it stopped.")


def _summary(result: ApplyResult, into_qdrant: bool = False) -> None:
    from vecshift import ui

    if ui.fancy():
        _summary_fancy(result, into_qdrant)
        return
    color = {"complete": typer.colors.GREEN, "failed": typer.colors.RED}.get(
        result.status, typer.colors.YELLOW
    )
    typer.echo()
    typer.secho(f"Apply {result.status}.", fg=color, bold=True)
    if result.message:
        typer.echo(result.message)
    spent = _money(result.spent_usd) if result.spent_usd is not None else "unknown"
    total = (
        f" ({_money(result.total_spent_usd)} across runs)"
        if result.total_spent_usd is not None
        else ""
    )
    typer.echo(
        f"  Rows written {result.rows_written:,} · tokens {result.tokens:,} · spent {spent}"
        f"{total} · {result.seconds:,.0f} s"
    )
    if result.rows_failed:
        typer.echo(f"  Rejected by the provider: {result.rows_failed:,} rows (retried next run)")
    if result.status == "complete" and into_qdrant:
        typer.echo(f"\nNext: {QDRANT_NEXT}")
    elif result.status == "complete":
        typer.echo(
            "\nNext: check it's safe to switch with `vecshift cutover --check`, then run "
            "`vecshift cutover`.\nIt embeds any rows added or edited since this run first."
        )


@reports_errors
def apply(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before starting.")] = False,
    no_index: Annotated[
        bool, typer.Option("--no-index", help="Backfill only; build the index on a later run.")
    ] = False,
    until: Annotated[
        int | None,
        typer.Option(
            "--until",
            min=1,
            max=99,
            metavar="PERCENT",
            help="Stop once this share of rows has a new vector, to check before continuing.",
        ),
    ] = None,
    output_json: Annotated[
        bool, typer.Option("--json", help="Print progress as JSON lines, for scripts and UIs.")
    ] = False,
) -> None:
    """Run (or resume) a migration: add the column, embed every row, and build the index."""
    from vecshift.service.apply import ApplyOptions, PlanBlocked, run_apply

    view = _View(yes=yes, output_json=output_json)
    try:
        outcome = run_apply(
            job_file,
            ApplyOptions(no_index=no_index, until=until),
            confirm=view.confirm,
            on_start=view.start,
            on_event=view.event,
            should_stop=lambda: view.stop.requested,
        )
    except PlanBlocked as exc:
        typer.secho(exc.message, fg=typer.colors.RED, bold=True)
        finding_lines([f for f in exc.plan.findings if f.severity.rank >= 2])
        raise typer.Exit(1) from exc
    finally:
        view.close()

    result, into_qdrant = outcome.result, outcome.into_qdrant
    if output_json:
        typer.echo(json.dumps({"event": "result", **outcome.to_dict()}))
    else:
        removed, copied = outcome.removed, outcome.copied
        if removed or copied:
            parts = []
            if copied:
                parts.append(f"copied {copied:,} changed payloads")
            if removed:
                parts.append(f"removed {removed:,} points deleted from the source")
            message = "Also " + " and ".join(parts) + "."
            if view.fancy:
                from vecshift import ui

                ui.success(message)
            else:
                typer.echo(message)
        _summary(result, into_qdrant)
    if result.status == "failed":
        raise typer.Exit(1)
    if result.status in {"stopped", "budget"}:
        raise typer.Exit(EXIT_STOPPED)


class _View:
    """What a person sees of apply: the announcement, the question, and the progress."""

    def __init__(self, *, yes: bool, output_json: bool) -> None:
        from vecshift import ui

        self.yes = yes
        self.output_json = output_json
        self.fancy = ui.fancy() and not output_json
        self.stop = _Stop()
        self.noun = "column"
        self.progress: Callable[[Event], None] | None = None
        self.previous: Any = None
        self.started = False

    def confirm(self, notice: ApplyNotice) -> bool:
        from vecshift import ui

        self.noun = "collection" if notice.into_qdrant else "column"
        _announce(notice.plan, notice.job, notice.state, self.output_json, self.noun)
        if self.yes:
            return True
        if self.fancy:
            ui.console.print()
            ui.note(f"  {notice.message}")
            return ui.confirm("Apply?", default=False)
        typer.echo(f"\n{notice.message}", err=True)
        if not sys.stdin.isatty():
            raise ServiceError(
                "Not changing anything without confirmation.",
                "Pass --yes to proceed.",
                "confirm.required",
            )
        return typer.confirm("Apply?", err=True)

    def start(self) -> None:
        """Connected and about to change things: Ctrl-C now stops cleanly."""
        from vecshift import ui

        self.previous = signal.signal(signal.SIGINT, self.stop)
        self.started = True
        if self.output_json:
            self.progress = _json_lines
        elif self.fancy:
            ui.console.print()
            self.progress = _FancyProgress(self.noun)
        else:
            self.progress = _Progress(live=sys.stdout.isatty(), noun=self.noun)

    def event(self, event: Event) -> None:
        if self.progress is not None:
            self.progress(event)

    def close(self) -> None:
        if isinstance(self.progress, _FancyProgress):
            self.progress.close()
        if self.started:
            signal.signal(signal.SIGINT, self.previous)


def _json_lines(event: Event) -> None:
    typer.echo(json.dumps({"event": event.kind, **event.data}))


def _announce(plan: Plan, job: JobSpec, state: JobState, output_json: bool, noun: str) -> None:
    """What apply is about to do, before it asks."""
    from vecshift import ui

    est = plan.estimates
    warnings = [f for f in plan.findings if f.severity.rank == 2]
    if output_json:
        return
    if ui.fancy():
        from rich.text import Text

        from vecshift.cli_plan import CHANGE_MARKS

        ui.title("apply", job.name)
        for change in plan.changes:
            if change.kind != "cutover":
                mark, colour = CHANGE_MARKS.get(change.kind, ("·", "dim"))
                ui.console.print(Text.assemble((f"  {mark} ", f"bold {colour}"), change.summary))
        ui.console.print()
        rows: list[tuple[str, str, str]] = [
            ("Estimated cost", _money(est.cost_usd), ""),
            ("New vectors" if noun == "collection" else "New column", _size(est.new_bytes), ""),
        ]
        if job.limits.budget_usd:
            rows.append(("Budget", f"${job.limits.budget_usd:,.2f}", "apply stops there"))
        if state.runs:
            rows.append(
                (
                    "Resuming",
                    f"{state.rows_written:,} rows",
                    f"written in earlier runs, {_money(state.spent_usd)} spent",
                )
            )
        ui.kv(rows)
        if warnings:
            ui.console.print()
            ui.findings(warnings)
        return
    typer.secho(f"vecshift apply · {job.name}", bold=True)
    for change in plan.changes:
        if change.kind != "cutover":
            typer.echo(f"  · {change.summary}")
    cost = _money(est.cost_usd)
    budget = f", budget ${job.limits.budget_usd:,.2f}" if job.limits.budget_usd else ""
    size = "new vectors" if noun == "collection" else "new column"
    typer.echo(f"  Estimated cost {cost}{budget}; {size} {_size(est.new_bytes)}.")
    if state.runs:
        typer.echo(
            f"  Resuming: {state.rows_written:,} rows written in earlier runs, "
            f"{_money(state.spent_usd)} spent."
        )
    if warnings:
        finding_lines(warnings)
