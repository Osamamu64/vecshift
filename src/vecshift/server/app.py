"""The REST API: the same operations as the CLI, through the same service layer.

Endpoints are plain functions, so FastAPI runs them in its thread pool: the service layer
blocks on databases and models, and one slow request mustn't stall the others.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from vecshift import __version__
from vecshift.server.folder import JobFolder
from vecshift.server.guard import Guard
from vecshift.server.problems import environment_values, from_service, problem
from vecshift.server.settings import ServerSettings
from vecshift.service import Sending, ServiceError

log = logging.getLogger("vecshift.server")

API = "/api/v1"


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Version(BaseModel):
    version: str
    api: Literal["v1"] = "v1"


class Variable(BaseModel):
    name: str
    kind: Literal["connection", "url", "key"]
    set: bool
    display: str | None = Field(
        default=None, description="Where a connection string or URL points, without a password."
    )
    problem: str | None = Field(default=None, description="Why a value that's set can't be used.")
    used_by: list[str] = Field(description="The jobs that read this variable.")


class Environment(BaseModel):
    variables: list[Variable]


class Side(BaseModel):
    type: Literal["pgvector", "qdrant"]
    name: str = Field(description="The table, column, or collection.")


class Problem(BaseModel):
    code: str
    detail: str
    hint: str | None = None


class JobSummary(BaseModel):
    name: str
    valid: bool = Field(description="Whether the file is a valid job.")
    source: Side | None = None
    target: Side | None = None
    model: str | None = None
    stage: str | None = Field(
        default=None,
        description="From a quick status without counting rows: `ready` shows as `filling` "
        "until GET /jobs/{name}/status?full=true has counted.",
    )
    updated_at: str | None = Field(default=None, description="When the job's state last changed.")
    problem: Problem | None = Field(
        default=None, description="Why the job or its stage is unknown."
    )


class Jobs(BaseModel):
    jobs: list[JobSummary]


class JobState(BaseModel):
    runs: int
    rows_written: int
    tokens: int
    spent_usd: float
    failed: int = Field(description="Rows or points the model provider rejected.")
    history: list[dict[str, str]]
    collections: list[str]
    updated_at: str | None


class Job(BaseModel):
    name: str
    spec: dict[str, Any] = Field(description="The job file as JSON.")
    yaml: str = Field(description="The job file as written.")
    state: JobState


class DoctorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store: Literal["pgvector", "qdrant"]
    dsn_env: str = Field("VECSHIFT_DSN", description="pgvector: the connection string's variable.")
    url_env: str = Field("VECSHIFT_QDRANT_URL", description="Qdrant: the URL's variable.")
    key_env: str = Field("QDRANT_API_KEY", description="Qdrant: the API key's variable.")
    table: str | None = None
    column: str | None = None
    collection: str | None = Field(default=None, description="Qdrant: a collection or alias.")
    vector: str | None = None
    text_field: str | None = Field(default=None, description="The text column or payload field.")
    sample: int = Field(2000, ge=1, le=100_000)
    timeout: int = Field(60, ge=1, le=600, description="Seconds.")


PROBLEMS: dict[int | str, dict[str, Any]] = {
    status: {"description": "A problem (RFC 9457).", "content": {"application/problem+json": {}}}
    for status in (404, 409, 422, 502)
}


def create_app(settings: ServerSettings) -> FastAPI:
    folder = JobFolder(settings.jobs)
    app = FastAPI(
        title="vecshift",
        version="1",
        summary="Safe, observable embedding migrations for any vector store.",
        openapi_url=f"{API}/openapi.json",
        # The interactive docs load scripts from a CDN: not served, to keep the CSP strict.
        docs_url=None,
        redoc_url=None,
    )

    def secrets() -> list[str]:
        from vecshift.service.environment import variables

        return environment_values(v.name for v in variables(_jobs(folder)))

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        exc.message = folder.relative(exc.message)
        return from_service(exc, secrets())

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Without the input values: they're echoed back otherwise.
        errors = [
            {"loc": list(e.get("loc", ())), "message": e.get("msg", "")} for e in exc.errors()
        ]
        return problem(422, "The request isn't valid.", code="request.invalid", errors=errors)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return problem(exc.status_code, str(exc.detail), code=f"http.{exc.status_code}")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The type only: messages and tracebacks could carry a connection string.
        log.error("%s on %s %s", type(exc).__name__, request.method, request.url.path)
        return problem(500, "Something went wrong in vecshift.", code="internal")

    router = APIRouter(prefix=API, responses=PROBLEMS)

    @router.get("/health")
    def health() -> Health:
        return Health()

    @router.get("/version")
    def version() -> Version:
        return Version(version=__version__)

    @router.get("/environment")
    def environment() -> Environment:
        """Each variable vecshift may read, and whether it's set. Never a value."""
        from vecshift.service.environment import variables

        found = variables(_jobs(folder))
        return Environment(variables=[Variable(**v.to_dict()) for v in found])

    @router.get("/jobs")
    def jobs() -> Jobs:
        """Every job file in the folder, with a quick stage (no row counts)."""
        return Jobs(jobs=[_summary(folder, name) for name in folder.names()])

    @router.get("/jobs/{name}")
    def job(name: str) -> Job:
        from vecshift.migrate import JobState as State
        from vecshift.service import read_job

        path = folder.path(name)
        spec = read_job(path)
        state = State.for_job(path, spec.name)
        return Job(
            name=name,
            spec=_masked(spec.model_dump(mode="json"), spec),
            yaml=_masked_text(path.read_text(encoding="utf-8"), spec),
            state=JobState(
                runs=state.runs,
                rows_written=state.rows_written,
                tokens=state.tokens,
                spent_usd=state.spent_usd,
                failed=len(state.failed),
                history=state.history,
                collections=state.collections,
                updated_at=state.updated_at,
            ),
        )

    @router.get("/jobs/{name}/plan")
    def plan(
        name: str,
        probe: bool = False,
        sample: int = Query(2000, ge=50, le=100_000),
        allow_remote: bool = False,
    ) -> dict[str, Any]:
        """The same JSON as `vecshift plan --json`. Read-only. A probe that would send sample
        rows to a remote model needs `allow_remote=true`."""
        from vecshift.service import prepare_plan

        def consent(sending: Sending) -> bool:
            if allow_remote:
                return True
            urls = ", ".join(url for _, url in sending.remote)
            raise ServiceError(
                f"The probe would send {sending.count} sample rows to {urls}.",
                "Pass allow_remote=true to allow it.",
                "consent.required",
            )

        prepared = prepare_plan(folder.path(name), sample_size=sample, probe=probe, consent=consent)
        result: dict[str, Any] = prepared.plan.to_dict()
        return result

    @router.get("/jobs/{name}/status")
    def status(name: str, full: bool = False) -> dict[str, Any]:
        """The same JSON as `vecshift status --json`. Read-only. Without `full`, rows aren't
        counted (counting reads the whole table or source)."""
        from vecshift.service import status as collect

        return collect(folder.path(name), full=full).to_dict()

    @router.post("/doctor")
    def doctor(request: DoctorRequest) -> dict[str, Any]:
        """The same JSON as `vecshift doctor --json`. Read-only. The request names the
        variables holding the connection, never their values."""
        import os

        from vecshift.service.doctor import doctor_pgvector, doctor_qdrant
        from vecshift.service.environment import variables

        known = {v.name: v.kind for v in variables(_jobs(folder))}
        wanted = (
            [(request.dsn_env, "connection")]
            if request.store == "pgvector"
            else [(request.url_env, "url"), (request.key_env, "key")]
        )
        for variable, kind in wanted:
            if known.get(variable) != kind:
                raise ServiceError(
                    f"{variable} isn't a {kind} variable vecshift reads here.",
                    "Use one GET /environment lists, such as a job's dsn_env.",
                    "options.invalid",
                )
        value = os.environ.get(wanted[0][0])
        if not value:
            raise ServiceError(f"{wanted[0][0]} isn't set.", code="env.missing")
        if request.store == "pgvector":
            found = doctor_pgvector(
                value,
                table=request.table,
                column=request.column,
                text_column=request.text_field,
                sample_size=request.sample,
                timeout=request.timeout,
            )
        else:
            found = doctor_qdrant(
                value,
                collection=request.collection,
                vector=request.vector,
                text_field=request.text_field,
                sample_size=request.sample,
                timeout=request.timeout,
                key_env=request.key_env,
            )
        return found.to_dict()

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> Response:
        return JSONResponse({"api": f"{API}/", "openapi": f"{API}/openapi.json"})

    app.add_middleware(Guard, settings=settings)
    return app


def _jobs(folder: JobFolder) -> dict[str, Any]:
    """The valid jobs in the folder, by name."""
    from vecshift.service import read_job

    found = {}
    for name in folder.names():
        try:
            found[name] = read_job(folder.path(name))
        except ServiceError:
            continue
    return found


def _summary(folder: JobFolder, name: str) -> JobSummary:
    from vecshift.embeddings import parse_spec
    from vecshift.jobs.spec import QdrantSource
    from vecshift.migrate import JobState as State
    from vecshift.service import read_job, status

    path = folder.path(name)
    try:
        spec = read_job(path)
    except ServiceError as exc:
        return JobSummary(
            name=name,
            valid=False,
            problem=Problem(code=exc.code, detail=folder.relative(exc.message)),
        )
    if isinstance(spec.source, QdrantSource):
        source = Side(type="qdrant", name=spec.source.collection)
    else:
        source = Side(type="pgvector", name=spec.pg_source.table)
    target = (
        Side(type="qdrant", name=spec.target_collection)
        if spec.target.type == "qdrant"
        else Side(type="pgvector", name=spec.pg_target.column)
    )
    summary = JobSummary(
        name=name,
        valid=True,
        source=source,
        target=target,
        model=parse_spec(spec.model).safe_raw,
        updated_at=State.for_job(path, spec.name).updated_at,
    )
    try:
        summary.stage = status(path, full=False).stage
    except ServiceError as exc:
        summary.problem = Problem(code=exc.code, detail=exc.message, hint=exc.hint)
    return summary


def _models(spec: Any) -> list[tuple[str, str]]:
    """(as written, as shown) for each model in a job: shown without URL query strings."""
    from vecshift.embeddings import SpecError, parse_spec

    pairs = []
    for raw in (spec.model, spec.source.model):
        if not raw:
            continue
        try:
            shown = parse_spec(raw).safe_raw
        except SpecError:
            shown = raw.split("?", 1)[0]
        if shown != raw:
            pairs.append((raw, shown))
    return pairs


def _masked_text(text: str, spec: Any) -> str:
    for raw, shown in _models(spec):
        text = text.replace(raw, shown)
    return text


def _masked(data: dict[str, Any], spec: Any) -> dict[str, Any]:
    pairs = dict(_models(spec))
    if data.get("model") in pairs:
        data["model"] = pairs[data["model"]]
    source = data.get("source") or {}
    if source.get("model") in pairs:
        source["model"] = pairs[source["model"]]
    return data


def openapi(settings: ServerSettings | None = None) -> dict[str, Any]:
    """The OpenAPI schema, as committed in docs/openapi.json."""
    schema: dict[str, Any] = create_app(settings or ServerSettings(jobs=Path("."))).openapi()
    return schema
