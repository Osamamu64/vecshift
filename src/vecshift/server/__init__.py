"""The REST API server: ``vecshift serve``. Needs ``pip install "vecshift[server]"``.

It's a presentation layer, like the CLI: it may import ``vecshift.service`` and below, and
nothing imports it but ``vecshift serve``.
"""

from vecshift.server.app import create_app, openapi
from vecshift.server.settings import ServerSettings

__all__ = ["ServerSettings", "create_app", "openapi"]
