"""The ``vecshift eval`` command: is the new model better on your data, and how fast is it?"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from vecshift.cli_plan import DEFAULT_JOB
from vecshift.cli_style import reports_errors
from vecshift.eval.html import SETTING_NAMES
from vecshift.service import EvalOptions, ServiceError, read_job, run_eval

if TYPE_CHECKING:
    from vecshift.eval import EvalReport, SideReport

EXIT_INCONCLUSIVE = 3


def _confirm(sends: list[tuple[str, str]], count: int, yes: bool) -> bool:
    """Show where query text goes, and ask unless ``yes``. Whether to go ahead."""
    from vecshift import ui

    remote = [(name, url) for name, url in sends if url]
    if not remote:
        return True
    if ui.fancy():
        ui.warn("eval sends text to:")
        ui.kv([(name, url, "") for name, url in remote], indent=4)
        ui.note(
            f"  About {count:,} short queries (sentences from your rows, or generated from them)."
        )
        return yes or ui.confirm("Continue?", default=False)
    out = sys.stderr
    print("eval sends text to:", file=out)
    for name, url in remote:
        print(f"  {name:<40} {url}", file=out)
    print(
        f"About {count:,} short queries (sentences from your rows, or generated from them).",
        file=out,
    )
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ServiceError(
            "Not sending data without confirmation.", "Pass --yes to proceed.", "consent.required"
        )
    return typer.confirm("Continue?", err=True)


@reports_errors
def eval_(
    job_file: Annotated[Path, typer.Argument(help="The job file.")] = DEFAULT_JOB,
    num_queries: Annotated[int, typer.Option(min=5, max=10_000, help="Queries to evaluate.")] = 200,
    queries: Annotated[
        Path | None,
        typer.Option(help='JSONL of labeled queries: {"query": ..., "relevant": [ids]} per line.'),
    ] = None,
    generate_queries: Annotated[
        str | None,
        typer.Option(
            metavar="CHAT_MODEL",
            help="Write realistic queries with this chat model, e.g. openai/gpt-4o-mini.",
        ),
    ] = None,
    cross_language: Annotated[
        bool,
        typer.Option(
            help="With --generate-queries: ask in the other language (Arabic for Latin-script "
            "rows, English for Arabic ones)."
        ),
    ] = False,
    old_model: Annotated[
        str | None,
        typer.Option(help="The model that made the old vectors (default: source.model)."),
    ] = None,
    group_by: Annotated[
        str | None,
        typer.Option(
            metavar="COLUMN[.KEY]",
            help="Which rows belong to one document, e.g. document_id or metadata.source.",
        ),
    ] = None,
    neighbours: Annotated[
        int, typer.Option(min=0, max=5_000, help="Rows whose nearest rows are compared.")
    ] = 100,
    tolerance: Annotated[
        float, typer.Option(min=0, max=1, help="How far recall@10 may fall and still pass.")
    ] = 0.02,
    min_slice: Annotated[
        int, typer.Option(min=1, help="Fewest queries for a language pair to decide the verdict.")
    ] = 30,
    max_p95_ms: Annotated[
        float | None, typer.Option(help="Fail if the new p95 (embed + search) is above this.")
    ] = None,
    max_slowdown: Annotated[
        float | None,
        typer.Option(metavar="PERCENT", help="Fail if the new p95 is this much slower than old."),
    ] = None,
    dsn_env: Annotated[
        str | None,
        typer.Option(help="Read the connection string from this variable, e.g. a replica's."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed for sampling.")] = 7,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Don't ask before sending data.")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Print results as JSON.")] = False,
    html: Annotated[
        Path | None,
        typer.Option("--html", dir_okay=False, help="Also write an HTML report to this file."),
    ] = None,
) -> None:
    """Compare the old and new vectors on your data: search quality and latency. Read-only."""
    options = EvalOptions(
        num_queries=num_queries,
        queries=queries,
        generate_queries=generate_queries,
        cross_language=cross_language,
        old_model=old_model,
        group_by=group_by,
        neighbours=neighbours,
        tolerance=tolerance,
        min_slice=min_slice,
        max_p95_ms=max_p95_ms,
        max_slowdown=max_slowdown,
        dsn_env=dsn_env,
        seed=seed,
    )
    progress = _Steps()
    try:
        report = run_eval(
            job_file,
            options,
            consent=lambda sending: _confirm(list(sending.to), sending.count, yes),
            on_step=None if output_json else progress.step,
        )
    finally:
        progress.stop()

    name = read_job(job_file).name
    if output_json:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _render(report, name)
    if html is not None:
        from vecshift import __version__
        from vecshift.eval.html import render_html

        html.write_text(render_html(report, job=name, version=__version__), encoding="utf-8")
        if not output_json:
            typer.echo(f"\nWrote {html}")
    if report.verdict.status == "no_go":
        raise typer.Exit(1)
    if report.verdict.status == "inconclusive":
        raise typer.Exit(EXIT_INCONCLUSIVE)


class _Steps:
    """eval's progress: a spinner at a colour terminal, dim lines elsewhere."""

    def __init__(self) -> None:
        from vecshift import ui

        self.ui = ui
        self.status: Any = None

    def step(self, message: str) -> None:
        if not self.ui.fancy():
            typer.secho(f"  {message}…", dim=True, err=True)
            return
        if self.status is None:
            self.status = self.ui.console.status(
                f"{message}…", spinner="dots", spinner_style=self.ui.ACCENT
            )
            self.status.start()
        else:
            self.status.update(f"{message}…")

    def stop(self) -> None:
        if self.status is not None:
            self.status.stop()
            self.status = None


# --- Rendering


def _ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return ""
    delta = after - before
    color = typer.colors.GREEN if delta > 0.005 else typer.colors.RED if delta < -0.005 else None
    return typer.style(f"{delta:+.3f}", fg=color)


def _delta(before: float | None, after: float | None) -> Any:
    from rich.text import Text

    if before is None or after is None:
        return Text("")
    delta = after - before
    colour = "green" if delta > 0.005 else "red" if delta < -0.005 else "dim"
    return Text(f"{delta:+.3f}", style=colour)


def _render_fancy(report: EvalReport, name: str) -> None:
    from rich.text import Text

    from vecshift import ui

    old, new = report.old, report.new
    ui.title("eval", name)
    kind = "partial migration" if report.partial else "complete migration"
    ui.kv(
        [
            ("Old", Text.assemble(old.column, (f"  {old.model or 'model unknown'}", "dim")), ""),
            ("New", Text.assemble(new.column, (f"  {new.model}", "dim")), ""),
            ("Queries", f"{report.queries:,} {report.query_source}", kind),
        ]
    )
    for side in (old, new):
        if side.note:
            ui.warn(f"  {side.note}")

    if new.scores:
        ui.section("Search quality")
        rows: list[list[str | Text]] = []
        for key, after in new.scores.items():
            before = old.scores.get(key)
            label = f"Recall@10 · {'all queries' if key == 'all' else key} ({after.queries})"
            b = before.recall_at_10 if before else None
            rows.append([label, _num(b), _num(after.recall_at_10), _delta(b, after.recall_at_10)])
        a_old, a_new = old.scores.get("all"), new.scores.get("all")
        for label, attr in (("Recall@1 · all", "recall_at_1"), ("MRR@10 · all", "mrr_at_10")):
            b = getattr(a_old, attr) if a_old else None
            a = getattr(a_new, attr) if a_new else None
            rows.append([label, _num(b), _num(a), _delta(b, a)])
        ui.table(["", "old", "new", "change"], rows, right=frozenset({1, 2, 3}))
        if report.result_overlap is not None:
            ui.detail(f"Top-10 results in common: {report.result_overlap:.0%}")

    if old.same_group_at_10 is not None or report.neighbour_overlap is not None:
        ui.section("Nearest rows (from stored vectors)")
        if old.same_group_at_10 is not None or new.same_group_at_10 is not None:
            ui.table(
                ["", "old", "new", "change"],
                [
                    [
                        "Same document in top 10",
                        _num(old.same_group_at_10),
                        _num(new.same_group_at_10),
                        _delta(old.same_group_at_10, new.same_group_at_10),
                    ]
                ],
                right=frozenset({1, 2, 3}),
            )
        if report.neighbour_overlap is not None:
            ui.detail(f"Nearest rows in common: {report.neighbour_overlap:.0%}")

    if old.sweep or new.sweep or old.embed_latency or new.embed_latency:
        ui.section("Latency vs accuracy")
        rows = []
        for side in (old, new):
            setting = SETTING_NAMES.get(side.method or "", "exact")
            lat = side.embed_latency
            embed = f"{_ms(lat.p50_ms)} / {_ms(lat.p95_ms)}" if lat else "—"
            for i, point in enumerate(side.sweep):
                lp = point.latency
                p50, p95, p99 = (lp.p50_ms, lp.p95_ms, lp.p99_ms) if lp else (None, None, None)
                value = point.setting if point.setting is not None else "—"
                where = Text(side.side if i == 0 else "", style="bold")
                knob = Text(f"{setting}={value}")
                if point.current:
                    knob.append(" ●", style=ui.ACCENT)
                rows.append(
                    [
                        where,
                        knob,
                        _num(point.index_recall),
                        _ms(p50),
                        _ms(p95),
                        _ms(p99),
                        embed if i == 0 else "",
                    ]
                )
        ui.table(
            ["", "setting", "index recall", "p50", "p95", "p99", "embed p50 / p95"],
            rows,
            right=frozenset({2, 3, 4, 5}),
        )
        samples = next((p.latency.samples for p in new.sweep if p.latency), 0)
        ui.note(
            f"  ● current setting · {samples} queries per setting · search times include the "
            "network round trip"
        )
        ui.detail(
            f"End-to-end p95 (embed + search): old {_ms(old.end_to_end_p95)} · "
            f"new {_ms(new.end_to_end_p95)}"
        )

    for note in report.notes:
        ui.console.print()
        ui.note(f"  {note}")
    verdict = report.verdict
    label, kind = {
        "go": ("GO", "ok"),
        "no_go": ("NO-GO", "error"),
        "inconclusive": ("INCONCLUSIVE", "warn"),
    }[verdict.status]
    ui.verdict(f"Verdict: {label}", kind)
    for reason in verdict.reasons:
        ui.detail(f"✖ {reason}" if verdict.status != "go" else reason)
    for warning in verdict.warnings:
        ui.warn(f"    {warning}")
    ui.console.print()


def _render(report: EvalReport, name: str) -> None:
    from vecshift import ui

    if ui.fancy():
        _render_fancy(report, name)
        return
    old, new = report.old, report.new
    typer.secho(f"vecshift eval · {name}", bold=True)
    typer.echo(f"  Old  {old.column} · {old.model or 'model unknown'}")
    typer.echo(f"  New  {new.column} · {new.model}")
    kind = "partial migration" if report.partial else "complete migration"
    typer.echo(f"  {report.queries:,} {report.query_source} queries · {kind}")
    for side in (old, new):
        if side.note:
            typer.secho(f"  ⚠ {side.note}", fg=typer.colors.YELLOW)

    if new.scores:
        typer.secho("\nSearch quality", bold=True)
        typer.echo(f"  {'':<30}{'old':>8}{'new':>8}  change")
        for key, after in new.scores.items():
            before = old.scores.get(key)
            label = f"{'all queries' if key == 'all' else key} ({after.queries})"
            typer.echo(
                f"  Recall@10 {label:<20}{_num(before.recall_at_10 if before else None):>8}"
                f"{_num(after.recall_at_10):>8}  "
                f"{_change(before.recall_at_10 if before else None, after.recall_at_10)}"
            )
        a_old, a_new = old.scores.get("all"), new.scores.get("all")
        typer.echo(
            f"  {'Recall@1 (all)':<30}{_num(a_old.recall_at_1 if a_old else None):>8}"
            f"{_num(a_new.recall_at_1 if a_new else None):>8}"
        )
        typer.echo(
            f"  {'MRR@10 (all)':<30}{_num(a_old.mrr_at_10 if a_old else None):>8}"
            f"{_num(a_new.mrr_at_10 if a_new else None):>8}"
        )
        if report.result_overlap is not None:
            typer.echo(f"  Top-10 results in common: {report.result_overlap:.0%}")

    if old.same_group_at_10 is not None or report.neighbour_overlap is not None:
        typer.secho("\nNearest rows (from stored vectors)", bold=True)
        if old.same_group_at_10 is not None or new.same_group_at_10 is not None:
            typer.echo(
                f"  {'Same document in top 10':<30}{_num(old.same_group_at_10):>8}"
                f"{_num(new.same_group_at_10):>8}  "
                f"{_change(old.same_group_at_10, new.same_group_at_10)}"
            )
        if report.neighbour_overlap is not None:
            typer.echo(f"  Nearest rows in common: {report.neighbour_overlap:.0%}")

    if old.sweep or new.sweep or old.embed_latency or new.embed_latency:
        typer.secho("\nLatency vs accuracy", bold=True)
        typer.echo(f"  {'Query embedding p50 / p95':<26}{_lat(old)}   {_lat(new)}")
        for side in (old, new):
            if not side.sweep:
                continue
            setting = SETTING_NAMES.get(side.method or "", "exact")
            typer.echo(f"  {side.side} search ({side.method or 'no index'}):")
            for point in side.sweep:
                mark = " *" if point.current else ""
                lat = point.latency
                p50, p95, p99 = (lat.p50_ms, lat.p95_ms, lat.p99_ms) if lat else (None, None, None)
                value = point.setting if point.setting is not None else "—"
                typer.echo(
                    f"    {setting}={value}{mark:<3} index recall {_num(point.index_recall)}"
                    f"  p50 {_ms(p50)}  p95 {_ms(p95)}  p99 {_ms(p99)}"
                )
        samples = next((p.latency.samples for p in new.sweep if p.latency), 0)
        typer.secho(
            f"  * current setting. {samples} queries per setting; search times include the "
            "network round trip.",
            dim=True,
        )
        typer.echo(
            f"  End-to-end p95 (embed + search): old {_ms(old.end_to_end_p95)} · "
            f"new {_ms(new.end_to_end_p95)}"
        )

    for note in report.notes:
        typer.secho(f"\n  {note}", dim=True)
    verdict = report.verdict
    label, color = {
        "go": ("GO", typer.colors.GREEN),
        "no_go": ("NO-GO", typer.colors.RED),
        "inconclusive": ("INCONCLUSIVE", typer.colors.YELLOW),
    }[verdict.status]
    typer.secho(f"\nVerdict: {label}", fg=color, bold=True)
    for reason in verdict.reasons:
        typer.echo(f"  ✖ {reason}" if verdict.status != "go" else f"  {reason}")
    for warning in verdict.warnings:
        typer.secho(f"  ⚠ {warning}", fg=typer.colors.YELLOW)


def _lat(side: SideReport) -> str:
    lat = side.embed_latency
    text = f"{_ms(lat.p50_ms)} / {_ms(lat.p95_ms)}" if lat else "—"
    return f"{side.side} {text}"
