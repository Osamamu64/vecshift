"""The folder of job files the server serves. A job's name is its file's stem."""

from __future__ import annotations

import os
import re
from pathlib import Path

from vecshift.service import ServiceError

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SUFFIXES = (".yaml", ".yml")


class JobFolder:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def names(self) -> list[str]:
        """Every job file's name, sorted. Files whose names a URL can't carry are skipped."""
        found = {
            path.stem
            for path in self.root.iterdir()
            if path.suffix in SUFFIXES and path.is_file() and NAME.fullmatch(path.stem)
        }
        return sorted(found)

    def path(self, name: str) -> Path:
        """The job file called ``name``, which must be directly in this folder."""
        if NAME.fullmatch(name) and ".." not in name:
            for suffix in SUFFIXES:
                path = self.root / f"{name}{suffix}"
                # Through a symlink, a name could reach a file elsewhere: refuse it.
                if path.is_file() and path.resolve().parent == self.root:
                    return path
        raise ServiceError(f"There's no job called {name!r} here.", code="job.missing")

    def relative(self, text: str) -> str:
        """``text`` with this folder's path taken out of file names: callers needn't know
        where the server keeps its files."""
        return text.replace(f"{self.root}{os.sep}", "")
