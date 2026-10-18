"""Checks every request passes before it reaches an endpoint.

- **Host:** without a token the server only answers to ``localhost`` names, so a web page
  can't reach it through DNS rebinding (a hostile name that resolves to 127.0.0.1).
- **Token:** with ``VECSHIFT_API_TOKEN`` set, every request but ``/health`` needs
  ``Authorization: Bearer <token>``, compared in constant time.
- **Header:** requests that aren't ``GET`` or ``HEAD`` need ``X-Vecshift: 1``. A browser
  can't add a custom header to a cross-site request without CORS, which the server never
  allows, so another web page can't trigger them.
- **Size:** request bodies are capped.
- **Headers:** every response says not to cache, sniff, frame, or run anything.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from vecshift.server.problems import problem
from vecshift.server.settings import LOOPBACK, ServerSettings

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]

SAFE_METHODS = frozenset({"GET", "HEAD"})
OPEN_PATHS = frozenset({"/api/v1/health"})
HEADERS = [
    (b"cache-control", b"no-store"),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
]


class TooLarge(Exception):
    pass


class Guard:
    def __init__(self, app: App, settings: ServerSettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        refused = self.refusal(scope, headers)
        if refused is not None:
            await refused(scope, receive, self._secure(send))
            return

        size = int(headers.get("content-length") or 0)
        if size > self.settings.max_body:
            await self._too_large()(scope, receive, self._secure(send))
            return
        started = False
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.settings.max_body:
                    raise TooLarge
            return message

        async def sending(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await self._secure(send)(message)

        try:
            await self.app(scope, counted, sending)
        except TooLarge:
            if started:
                raise
            await self._too_large()(scope, receive, self._secure(send))

    def refusal(self, scope: Scope, headers: dict[str, str]) -> Any:
        """A problem response if the request mustn't go further, else None."""
        token = self.settings.token
        if token is None and _hostname(headers.get("host", "")) not in LOOPBACK:
            return problem(
                421,
                "This server only answers requests addressed to localhost.",
                code="request.host",
                hint="Call it as http://localhost:PORT or http://127.0.0.1:PORT.",
            )
        if token is not None and scope["path"] not in OPEN_PATHS:
            given = headers.get("authorization", "")
            expected = f"Bearer {token}"
            if not hmac.compare_digest(given.encode(), expected.encode()):
                return problem(
                    401,
                    "This server needs a token.",
                    code="request.unauthorized",
                    hint="Send Authorization: Bearer <VECSHIFT_API_TOKEN>.",
                )
        if scope["method"] not in SAFE_METHODS and headers.get("x-vecshift") != "1":
            return problem(
                403,
                "Requests that aren't GET need the header X-Vecshift: 1.",
                code="request.header",
            )
        return None

    def _too_large(self) -> Any:
        return problem(
            413,
            f"The request body is larger than {self.settings.max_body:,} bytes.",
            code="request.too_large",
        )

    def _secure(self, send: Send) -> Send:
        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *HEADERS]
            await send(message)

        return wrapped


def _hostname(host: str) -> str:
    """The host name from a Host header, without the port."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host
