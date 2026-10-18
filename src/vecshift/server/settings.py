"""How the API server runs: where the jobs are, where it listens, and who may call it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
TOKEN_ENV = "VECSHIFT_API_TOKEN"  # noqa: S105 - a variable's name, not a token
MIN_TOKEN = 16


@dataclass(frozen=True, slots=True)
class ServerSettings:
    jobs: Path
    """The folder of job files (``*.yaml``). Each job's state is in ``.vecshift/`` next to it."""
    host: str = "127.0.0.1"
    port: int = 8765
    token: str | None = None
    """When set, every request needs ``Authorization: Bearer <token>``."""
    max_body: int = 1_000_000
    """The largest request body accepted, in bytes."""

    @property
    def local(self) -> bool:
        """Whether only this machine can reach the server."""
        return self.host.strip("[]").lower() in LOOPBACK

    def problem(self) -> str | None:
        """Why the server mustn't start like this, if it mustn't."""
        if self.token is not None and len(self.token) < MIN_TOKEN:
            return f"{TOKEN_ENV} is too short: use at least {MIN_TOKEN} random characters."
        if not self.local and self.token is None:
            return (
                f"Listening on {self.host} lets other machines call the API, so it needs a "
                f"token: set {TOKEN_ENV}."
            )
        if not self.jobs.is_dir():
            return f"{self.jobs} isn't a folder."
        return None
