"""What each run does: its options, the checks before it starts, and the work itself.

Every action calls the same service function as its CLI command, and turns the service's
callbacks into events: the engine's own events for apply, ``step`` for eval's progress,
``finding`` for cutover's checks, ``switched`` and ``deleted`` for what changed, ``plan``
for what bench is about to embed, and a final ``result`` holding the same JSON as the
command's ``--json``, or ``error`` holding a problem.

Confirmations are explicit: ``confirm.apply`` for apply, ``confirm.switch`` for cutover and
rollback, and ``confirm.name`` for cleanup, which must be the name the service computes
from the job's own record. Text goes to a model that isn't on this machine only with
``allow_remote``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vecshift.server.problems import scrub
from vecshift.server.runs import Run, _append
from vecshift.service import ServiceError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Confirm(_Strict):
    apply: bool = Field(
        default=False, description="apply: it changes the database and spends money."
    )
    switch: bool = Field(default=False, description="cutover and rollback: they switch searches.")
    name: str | None = Field(
        default=None,
        description="cleanup: the name of the old column or collection it deletes.",
        max_length=256,
    )


class ApplyOptions(_Strict):
    no_index: bool = Field(
        default=False, description="Backfill only; build the index on a later run."
    )
    until: int | None = Field(
        default=None, ge=1, le=99, description="Stop once this percentage has a new vector."
    )


class LabeledQuery(_Strict):
    query: str = Field(min_length=1, max_length=4000)
    relevant: list[str] = Field(min_length=1, max_length=1000)


class EvalOptions(_Strict):
    num_queries: int = Field(default=200, ge=5, le=10_000)
    queries: list[LabeledQuery] | None = Field(
        default=None, max_length=10_000, description="Labeled queries, instead of a file."
    )
    generate_queries: str | None = Field(
        default=None, description="A chat model that writes the queries."
    )
    cross_language: bool = False
    old_model: str | None = None
    group_by: str | None = None
    neighbours: int = Field(default=100, ge=0, le=5_000)
    tolerance: float = Field(default=0.02, ge=0, le=1)
    min_slice: int = Field(default=30, ge=1)
    max_p95_ms: float | None = Field(default=None, gt=0)
    max_slowdown: float | None = Field(default=None, gt=0)
    seed: int = 7


class CutoverOptions(_Strict):
    check_only: bool = Field(default=False, description="Only check that cutover is safe.")
    allow_missing: bool = Field(
        default=False, description="Switch even though the provider rejected some rows."
    )


class Empty(_Strict):
    pass


class _Request(_Strict):
    confirm: Confirm = Field(default_factory=Confirm)
    allow_remote: bool = Field(
        default=False, description="Allow sending text to a model that isn't on this machine."
    )


class ApplyRun(_Request):
    action: Literal["apply"]
    options: ApplyOptions = Field(default_factory=ApplyOptions)


class EvalRun(_Request):
    action: Literal["eval"]
    options: EvalOptions = Field(default_factory=EvalOptions)


class CutoverRun(_Request):
    action: Literal["cutover"]
    options: CutoverOptions = Field(default_factory=CutoverOptions)


class RollbackRun(_Request):
    action: Literal["rollback"]
    options: Empty = Field(default_factory=Empty)


class CleanupRun(_Request):
    action: Literal["cleanup"]
    options: Empty = Field(default_factory=Empty)


RunRequest = Annotated[
    ApplyRun | EvalRun | CutoverRun | RollbackRun | CleanupRun, Field(discriminator="action")
]
Work = Callable[[Run], None]


def prepare(request: Any, job_file: Path, secrets: list[str]) -> tuple[Work, bool]:
    """Check the request before anything starts; return its work, and whether it can stop.

    A missing confirmation, or remote text without ``allow_remote``, is refused here, so
    the caller gets a 422 at once rather than a failed run.
    """
    from vecshift.service import read_job

    job = read_job(job_file)
    if isinstance(request, ApplyRun):
        if not request.confirm.apply:
            raise _unconfirmed("apply changes the database and spends money", "apply")
        _remote(request, [job.model])
        return _apply(request, job_file, secrets), True
    if isinstance(request, EvalRun):
        o = request.options
        _remote(request, [job.model, o.old_model or job.source.model, o.generate_queries])
        return _eval(request, job_file, secrets), False
    if isinstance(request, CutoverRun):
        if not request.options.check_only and not request.confirm.switch:
            raise _unconfirmed("cutover switches searches to the new vectors", "switch")
        return _cutover(request, job_file, secrets), False
    if isinstance(request, RollbackRun):
        if not request.confirm.switch:
            raise _unconfirmed("rollback switches searches back to the old vectors", "switch")
        return _rollback(request, job_file, secrets), False
    if not request.confirm.name:
        raise ServiceError(
            "cleanup deletes the old vectors for good: send confirm.name, the name of the "
            "old column or collection.",
            "GET /jobs/{name}/status shows it as previous.",
            "confirm.required",
        )
    return _cleanup(request, job_file, secrets), False


def _unconfirmed(what: str, field: str) -> ServiceError:
    return ServiceError(
        f"{what[0].upper()}{what[1:]}: send confirm.{field}=true to go ahead.",
        code="confirm.required",
    )


def _remote(request: _Request | BenchRun, models: list[str | None]) -> None:
    """Refuse a request that would send text away without ``allow_remote``."""
    from vecshift.embeddings import SpecError, parse_spec
    from vecshift.service.plan import destination

    if request.allow_remote:
        return
    urls = []
    for raw in models:
        if not raw:
            continue
        try:
            where = destination(parse_spec(raw))
        except SpecError:
            continue  # the service reports a bad spec in its own words
        if where and where not in urls:
            urls.append(where)
    if urls:
        raise ServiceError(
            f"This sends text to {', '.join(urls)}.",
            "Send allow_remote=true to allow it.",
            "consent.required",
        )


class Reporter:
    """Records a run's events and outcome, with secret values scrubbed out."""

    def __init__(self, run: Run, secrets: list[str]) -> None:
        self.run = run
        self.secrets = secrets

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        self.run.emit(kind, self.clean(data))

    def clean(self, data: Any) -> Any:
        if isinstance(data, str):
            return scrub(data, self.secrets)
        if isinstance(data, dict):
            return {k: self.clean(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.clean(v) for v in data]
        return data

    def done(
        self,
        result: dict[str, Any],
        state: str = "completed",
        problem: dict[str, Any] | None = None,
    ) -> None:
        self.run.result = self.clean(result)
        self.run.emit("result", self.run.result or {})
        if problem is not None:
            self.run.problem = self.clean(problem)
            self.run.emit("error", self.run.problem or {})
        self.run.finish(state)

    def failed(self, exc: ServiceError) -> None:
        problem: dict[str, Any] = {"code": exc.code, "detail": exc.message}
        if exc.hint:
            problem["hint"] = exc.hint
        plan = getattr(exc, "plan", None)
        if plan is not None:
            problem["findings"] = [f.to_dict() for f in plan.findings if f.severity.rank >= 2]
        self.run.problem = self.clean(problem)
        self.run.emit("error", self.run.problem or {})
        self.run.finish("failed")


def _guarded(job_file: Path, secrets: list[str], body: Callable[[Reporter], None]) -> Work:
    """The run's work: problems become its ``error`` event, without the jobs folder's path."""

    def work(run: Run) -> None:
        report = Reporter(run, secrets)
        try:
            body(report)
        except ServiceError as exc:
            exc.message = exc.message.replace(f"{job_file.parent}{os.sep}", "")
            report.failed(exc)

    return work


def _apply(request: ApplyRun, job_file: Path, secrets: list[str]) -> Work:
    from vecshift.service.apply import ApplyOptions as Options
    from vecshift.service.apply import run_apply

    def body(report: Reporter) -> None:
        run = report.run
        outcome = run_apply(
            job_file,
            Options(no_index=request.options.no_index, until=request.options.until),
            confirm=lambda notice: request.allow_remote or notice.destination == "your machine",
            on_event=lambda event: report.emit(event.kind, event.data),
            should_stop=lambda: run.stop_requested,
        )
        status = outcome.result.status
        state = {"complete": "completed", "failed": "failed"}.get(status, "stopped")
        report.done(outcome.to_dict(), state)

    return _guarded(job_file, secrets, body)


def _eval(request: EvalRun, job_file: Path, secrets: list[str]) -> Work:
    from vecshift import __version__
    from vecshift.eval.html import render_html
    from vecshift.server.runs import _write
    from vecshift.service import read_job
    from vecshift.service.evaluation import EvalOptions as Options
    from vecshift.service.evaluation import run_eval

    def body(report: Reporter) -> None:
        run, o = report.run, request.options
        queries = None
        if o.queries:
            queries = run.folder / f"{run.run_id}.queries.jsonl"
            for q in o.queries:
                _append(queries, json.dumps({"query": q.query, "relevant": q.relevant}))
        try:
            result = run_eval(
                job_file,
                Options(
                    num_queries=o.num_queries,
                    queries=queries,
                    generate_queries=o.generate_queries,
                    cross_language=o.cross_language,
                    old_model=o.old_model,
                    group_by=o.group_by,
                    neighbours=o.neighbours,
                    tolerance=o.tolerance,
                    min_slice=o.min_slice,
                    max_p95_ms=o.max_p95_ms,
                    max_slowdown=o.max_slowdown,
                    seed=o.seed,
                ),
                consent=lambda sending: request.allow_remote,
                on_step=lambda message: report.emit("step", {"message": message}),
            )
        finally:
            if queries is not None:
                queries.unlink(missing_ok=True)
        name = read_job(job_file).name
        _write(run.report_path, render_html(result, job=name, version=__version__))
        report.done(result.to_dict())

    return _guarded(job_file, secrets, body)


def _cutover(request: CutoverRun, job_file: Path, secrets: list[str]) -> Work:
    from vecshift.service.switching import SwitchNotice, cutover

    def body(report: Reporter) -> None:
        shown = False

        def confirm(notice: SwitchNotice) -> bool:
            nonlocal shown
            ready = notice.readiness
            for finding in ready.findings if ready is not None else []:
                report.emit("finding", finding.to_dict())
            shown = True
            if ready is not None and ready.pending and not request.allow_remote:
                _remote(request, [_model(job_file)])
            return True

        outcome = cutover(
            job_file,
            check=request.options.check_only,
            allow_missing=request.options.allow_missing,
            confirm=confirm,
        )
        if not shown:
            for finding in outcome.readiness.findings:
                report.emit("finding", finding.to_dict())
        if outcome.status == "cut_over":
            report.emit(
                "switched", {"live": outcome.live, "new": outcome.new, "previous": outcome.previous}
            )
        report.done(outcome.to_dict())

    return _guarded(job_file, secrets, body)


def _rollback(request: RollbackRun, job_file: Path, secrets: list[str]) -> Work:
    from vecshift.service.switching import rollback

    def body(report: Reporter) -> None:
        outcome = rollback(job_file, confirm=lambda notice: True)
        live = outcome.previous if outcome.store == "qdrant" else outcome.live
        report.emit("switched", {"live": outcome.live, "points_at": live, "new": outcome.new})
        report.done(outcome.to_dict())

    return _guarded(job_file, secrets, body)


def _cleanup(request: CleanupRun, job_file: Path, secrets: list[str]) -> Work:
    from vecshift.service.switching import cleanup

    def body(report: Reporter) -> None:
        outcome = cleanup(job_file, confirm=lambda notice: request.confirm.name)
        report.emit("deleted", {"name": outcome.deleted})
        report.done(outcome.to_dict())

    return _guarded(job_file, secrets, body)


def _model(job_file: Path) -> str:
    from vecshift.service import read_job

    return read_job(job_file).model


class JobDocuments(_Strict):
    type: Literal["job"]
    job: str = Field(description="Sample the source of this job.", max_length=128)


class TableDocuments(_Strict):
    type: Literal["pgvector"]
    dsn_env: str = Field(default="VECSHIFT_DSN", description="The connection string's variable.")
    table: str | None = None
    column: str | None = Field(default=None, description="The vector column, if there are several.")
    text_column: str | None = None


class CollectionDocuments(_Strict):
    type: Literal["qdrant"]
    url_env: str = Field(default="VECSHIFT_QDRANT_URL", description="The URL's variable.")
    key_env: str = Field(default="QDRANT_API_KEY", description="The API key's variable.")
    collection: str | None = Field(default=None, description="A collection or alias.")
    vector: str | None = None
    text_field: str | None = None


class Document(_Strict):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(max_length=100_000)


class InlineDocuments(_Strict):
    type: Literal["documents"]
    documents: list[Document] = Field(min_length=1, max_length=100_000)


BenchDocuments = Annotated[
    JobDocuments | TableDocuments | CollectionDocuments | InlineDocuments,
    Field(discriminator="type"),
]


class BenchRun(_Strict):
    documents: BenchDocuments = Field(description="Where the documents come from.")
    models: list[str] = Field(
        default=["hash/256", "hash/1024"],
        min_length=1,
        max_length=20,
        description="The models to compare, as provider/model[,option=value].",
    )
    sample: int = Field(default=1000, ge=10, le=100_000, description="Documents to sample.")
    num_queries: int = Field(default=100, ge=5, le=10_000)
    queries: list[LabeledQuery] | None = Field(
        default=None, max_length=10_000, description="Labeled queries, instead of proxy ones."
    )
    generate_queries: str | None = Field(
        default=None, description="A chat model that writes the queries."
    )
    seed: int = 7
    cache: bool = Field(default=True, description="Reuse vectors from the embedding cache.")
    allow_remote: bool = Field(
        default=False, description="Allow sending text to a model that isn't on this machine."
    )


def prepare_bench(request: BenchRun, source: Any, folder: Path, secrets: list[str]) -> Work:
    """Check a bench request before it starts (its options and ``allow_remote``); return
    its work. ``source`` is where the documents come from, already resolved."""
    from vecshift.bench import Query
    from vecshift.service.bench import BenchOptions, check

    options = BenchOptions(
        models=tuple(request.models),
        sample_size=request.sample,
        num_queries=request.num_queries,
        queries=None
        if request.queries is None
        else tuple(Query(q.query.strip(), frozenset(q.relevant)) for q in request.queries),
        generate_queries=request.generate_queries,
        seed=request.seed,
        cache=request.cache,
    )
    specs, generator = check(options)
    for spec in [*specs, *([generator] if generator else [])]:
        if spec.safe_raw != spec.raw:
            raise ServiceError(
                f"{spec.safe_raw}: a URL's query string can hold a key, so the API doesn't "
                "take one.",
                "Read the key from a variable instead, with key_env=NAME.",
                "options.invalid",
            )
    _remote(request, [*request.models, request.generate_queries])
    return _bench(options, source, folder, secrets)


def _bench(options: Any, source: Any, folder: Path, secrets: list[str]) -> Work:
    from vecshift import __version__
    from vecshift.bench.html import render_html
    from vecshift.server.runs import _write
    from vecshift.service.bench import BenchNotice, run_bench

    def body(report: Reporter) -> None:
        def confirm(notice: BenchNotice) -> bool:
            report.emit("plan", notice.to_dict())
            return True  # allow_remote was checked before the run started

        result = run_bench(
            source,
            options,
            confirm=confirm,
            on_step=lambda message: report.emit("step", {"message": message}),
        )
        _write(report.run.report_path, render_html(result, version=__version__))
        if result.models and all(m.error for m in result.models):
            problem = {
                "code": "model.failed",
                "detail": "Every model failed: each one's error is in the result.",
            }
            report.done(result.to_dict(), "failed", problem)
        else:
            report.done(result.to_dict())

    return _guarded(folder / "bench", secrets, body)
