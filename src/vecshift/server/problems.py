"""Errors as RFC 9457 ``application/problem+json``, from the service layer's ``ServiceError``.

The ``detail`` and ``hint`` are the same sentences the CLI shows. ``code`` is the service's
stable code. Nothing here ever includes a traceback, and every text is scrubbed of the
values of the variables vecshift reads, in case one ended up in a message.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from http import HTTPStatus
from typing import Any

from starlette.responses import JSONResponse

from vecshift.service import ServiceError

MEDIA_TYPE = "application/problem+json"
QUERY = re.compile(r"(https?://[^\s?#'\",()]+)\?[^\s'\",()]*")
"""A URL's query string, which can carry a key, as in a model spec's url=."""

STATUS = {
    "job.missing": 404,
    "job.exists": 409,
    "job.name_taken": 409,
    "job.renamed": 409,
    "run.missing": 404,
    "report.missing": 404,
    "run.finished": 409,
    "run.not_stoppable": 409,
    "source.not_found": 404,
    "job.invalid": 422,
    "job.wrong_store": 422,
    "options.invalid": 422,
    "consent.required": 422,
    "confirm.required": 422,
    "confirm.mismatch": 422,
    "source.unsupported": 422,
    "source.no_text": 422,
    "documents.invalid": 422,
    "env.missing": 409,
    "source.ambiguous": 409,
    "job.running": 409,
    "plan.errors": 409,
    "state.no_record": 409,
    "eval.not_applied": 409,
    "connect.failed": 502,
    "database.error": 502,
    "qdrant.error": 502,
    "model.failed": 502,
    "database.busy": 503,
}


def status_for(code: str) -> int:
    return STATUS.get(code, 400)


def problem(
    status: int,
    detail: str,
    *,
    code: str,
    hint: str | None = None,
    secrets: Iterable[str] = (),
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"urn:vecshift:error:{code}",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": scrub(detail, secrets),
        "code": code,
    }
    if hint:
        body["hint"] = scrub(hint, secrets)
    body.update(extra)
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(body, status_code=status, media_type=MEDIA_TYPE, headers=headers)


def from_service(exc: ServiceError, secrets: Iterable[str] = ()) -> JSONResponse:
    secrets = list(secrets)
    extra: dict[str, Any] = {}
    candidates = getattr(exc, "candidates", None)
    if candidates is not None:
        extra["candidates"] = [c.to_dict() for c in candidates]
    errors = getattr(exc, "errors", None)
    if errors is not None:
        extra["errors"] = [
            {k: scrub(str(v), secrets) for k, v in error.items()} for error in errors
        ]
    return problem(
        status_for(exc.code), exc.message, code=exc.code, hint=exc.hint, secrets=secrets, **extra
    )


def scrub(text: str, secrets: Iterable[str]) -> str:
    """``text`` with any of ``secrets``, and URLs' query strings, replaced by an ellipsis."""
    for value in sorted({s for s in secrets if len(s) >= 6}, key=len, reverse=True):
        text = text.replace(value, "…")
    return QUERY.sub(r"\1?…", text)


def environment_values(names: Iterable[str]) -> list[str]:
    """The values of these variables that are set: what must never appear in a response."""
    return [v for v in (os.environ.get(n) for n in names) if v]
