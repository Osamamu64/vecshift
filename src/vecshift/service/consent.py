"""Asking before text leaves the machine.

The service never prompts. It describes what it's about to send and asks the caller's
``Consent`` callback: the CLI asks at the terminal (or takes ``--yes``), and the API
answers from the request's ``allow_remote``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vecshift.service.errors import Declined, ServiceError


@dataclass(frozen=True, slots=True)
class Sending:
    """Text about to go to models.

    ``to`` lists (model, URL) for every model involved; the URL is empty for one that runs
    on this machine. ``count`` is how many texts, ``what`` what they are.
    """

    action: str
    """``probe`` (plan --probe) or ``eval``."""
    to: tuple[tuple[str, str], ...]
    count: int
    what: str

    @property
    def remote(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, url) for name, url in self.to if url)


Consent = Callable[[Sending], bool]
"""Asked before anything is sent to a remote model. Returning False stops with ``Declined``.

It may also raise ``ServiceError`` itself, for instance when there's nobody to ask."""


def ask(consent: Consent | None, sending: Sending) -> None:
    """Go ahead only if nothing leaves the machine, or the caller agrees."""
    if not sending.remote:
        return
    if consent is None:
        raise ServiceError(
            "Not sending data without confirmation.",
            "Allow sending text to: " + ", ".join(url for _, url in sending.remote),
            code="consent.required",
        )
    if not consent(sending):
        raise Declined("Not sending anything.")
