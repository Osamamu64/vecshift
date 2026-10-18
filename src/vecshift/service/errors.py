"""What the service layer raises instead of exiting: a message for people, and a code."""

from __future__ import annotations


class ServiceError(Exception):
    """Something stopped the operation. Nothing was changed unless the message says so.

    ``message`` and ``hint`` are plain sentences shown to people, the same in the CLI, the
    API, and the UI. ``code`` is stable, for scripts: ``area.reason``, such as
    ``job.invalid`` or ``env.missing``.
    """

    def __init__(self, message: str, hint: str | None = None, code: str = "error") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


class Declined(ServiceError):
    """The person said no when asked. Nothing was changed or sent."""

    def __init__(self, message: str = "Declined.") -> None:
        super().__init__(message, code="declined")


def first_line(exc: BaseException) -> str:
    """The first line of an error's text, or its type: database errors run long."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
