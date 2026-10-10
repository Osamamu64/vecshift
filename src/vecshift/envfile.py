"""Reading and writing ``.env`` files, so connection strings and keys needn't be exported.

A value already in the environment always wins over the file. Values are never printed:
callers learn only which names were loaded.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

DEFAULT = Path(".env")
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "$": "$"}


class EnvFileError(Exception):
    """A line couldn't be read. The message names the line, never its value."""


def parse(text: str) -> dict[str, str]:
    """``NAME=value`` lines, with optional ``export``, quotes, and ``#`` comments."""
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            raise EnvFileError(f"line {number} isn't NAME=value")
        name, raw = match.groups()
        values[name] = _value(raw, number)
    return values


def _value(raw: str, number: int) -> str:
    if raw[:1] in {"'", '"'}:
        quote = raw[0]
        end = _closing(raw, quote)
        if end is None:
            raise EnvFileError(f"line {number} has an unclosed {quote}")
        rest = raw[end + 1 :].strip()
        if rest and not rest.startswith("#"):
            raise EnvFileError(f"line {number} has text after the closing {quote}")
        inner = raw[1:end]
        if quote == "'":
            return inner
        return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(0)), inner)
    # Unquoted: a comment starts at " #".
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def _closing(raw: str, quote: str) -> int | None:
    i = 1
    while i < len(raw):
        if quote == '"' and raw[i] == "\\":
            i += 2
            continue
        if raw[i] == quote:
            return i
        i += 1
    return None


def readable_by_others(path: Path) -> bool:
    """Whether other users on this machine can read the file."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return os.name == "posix" and bool(mode & (stat.S_IRGRP | stat.S_IROTH))


def load(path: Path = DEFAULT) -> list[str]:
    """Set variables from ``path`` that aren't set yet. Returns the names it set.

    A missing file is not an error: the file is optional.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise EnvFileError(f"couldn't read it: {exc.strerror}") from exc
    loaded = []
    for name, value in parse(text).items():
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def quote(value: str) -> str:
    """A value as a ``.env`` line would hold it, safe to ``source`` from a shell too."""
    if "'" not in value and "\n" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    return '"' + escaped.replace("\n", "\\n") + '"'


def save(path: Path, name: str, value: str) -> None:
    """Set ``name`` in the file, replacing an earlier line, readable by its owner only."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"not a variable name: {name!r}")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    entry = f"{name}={quote(value)}"
    kept = [line for line in lines if (m := _LINE.match(line)) is None or m.group(1) != name]
    kept.append(entry)
    # Create it private, then tighten an existing file before writing the secret to it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        os.write(fd, ("\n".join(kept) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
