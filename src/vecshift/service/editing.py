"""Writing job files: a new one from a spec, or an existing one changed in place.

A new file starts from ``vecshift init``'s commented template. An existing file is edited
line by line, so its comments and layout stay: a changed setting keeps its line and its
comment, a setting that's no longer set is commented out, and a new one uses its
commented-out example when there is one. The result is read back and must equal the spec;
when the file is laid out in a way this can't follow (flow style, block text, anchors), it's
written afresh instead, without the comments, and the caller is told.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vecshift.service.errors import ServiceError

if TYPE_CHECKING:
    from vecshift.jobs import JobSpec

SECTIONS = ("source", "target", "limits")
SETTING = re.compile(r"(?P<indent> *)(?P<hash>#\s*)?(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>.*)")
BARE = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]*")


class InvalidJob(ServiceError):
    """The spec isn't a valid job. ``errors`` lists each problem as {loc, message}."""

    def __init__(self, message: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(message, code="job.invalid")
        self.errors = errors


def validate(data: dict[str, Any]) -> JobSpec:
    """The spec as a job, or ``InvalidJob`` in the same words as a job file's problems."""
    from pydantic import ValidationError

    from vecshift.jobs import JobSpec
    from vecshift.jobs.spec import format_errors

    try:
        return JobSpec.model_validate(data)
    except ValidationError as exc:
        errors = [
            {"loc": line.strip().split(": ", 1)[0], "message": line.strip().split(": ", 1)[-1]}
            for line in format_errors(exc).splitlines()
        ]
        raise InvalidJob(f"The job has problems:\n{format_errors(exc)}", errors) from exc


@dataclass(frozen=True, slots=True)
class Written:
    text: str
    comments_kept: bool
    """False when the file had to be written afresh, without its comments."""


def render(spec: JobSpec) -> Written:
    """A new job file for ``spec``, commented like the one ``vecshift init`` writes."""
    from vecshift.jobs.spec import QdrantSource, qdrant_template, template

    source = spec.source
    budget = spec.limits.budget_usd
    if spec.target.type == "qdrant":
        target = spec.qdrant_target
        if isinstance(source, QdrantSource):
            start = qdrant_template(
                spec.model,
                collection=source.collection,
                vector=source.vector,
                text_field=source.text_field,
                source_model=source.model,
                alias=target.alias,
                new_collection=target.collection,
                budget_usd=budget,
            )
        else:
            start = qdrant_template(
                spec.model,
                table=source.table,
                text_field=source.text_column,
                vector_column=source.vector_column,
                source_model=source.model,
                alias=target.alias,
                new_collection=target.collection,
                budget_usd=budget,
            )
    else:
        pg = spec.pg_source
        start = template(
            pg.table,
            spec.model,
            spec.pg_target.column,
            vector_column=pg.vector_column,
            text_column=pg.text_column,
            source_model=pg.model,
            budget_usd=budget,
        )
    return update(start, spec)


def update(text: str, spec: JobSpec) -> Written:
    """``text`` changed so that it reads as ``spec``, keeping comments where it can."""
    edited = _edit(text, spec)
    if edited is not None and _reads_as(edited, spec):
        return Written(edited, True)
    return Written(_plain(spec), False)


def write(path: Path, text: str, *, new: bool) -> None:
    """Write a job file in one step: a crash leaves the old file or the new one, whole.

    ``new`` refuses to replace a file that's already there.
    """
    if new:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise ServiceError(
                f"There's already a job called {path.stem!r}.", code="job.exists"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _reads_as(text: str, spec: JobSpec) -> bool:
    import yaml
    from pydantic import ValidationError

    from vecshift.jobs import JobSpec

    try:
        return JobSpec.model_validate(yaml.safe_load(text)) == spec
    except (yaml.YAMLError, ValidationError):
        return False


def _plain(spec: JobSpec) -> str:
    import yaml

    data = spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    data = {"version": 1, **data}
    return (
        "# vecshift job. Check it with `vecshift plan`; nothing changes until "
        "`vecshift apply`.\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )


def _value(value: Any) -> str:
    """A YAML scalar: a plain word when it reads back as itself, quoted otherwise."""
    import yaml

    if isinstance(value, str) and BARE.fullmatch(value) and yaml.safe_load(value) == value:
        return value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return json.dumps(value, ensure_ascii=False)


def _split(rest: str) -> tuple[str, str]:
    """A line's value and its trailing comment (with the spaces before it)."""
    quote = None
    for i, char in enumerate(rest):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (i == 0 or rest[i - 1] in " \t"):
            value = rest[:i].rstrip()
            return value.strip(), rest[len(value) :]
    return rest.strip(), ""


@dataclass(slots=True)
class _Setting:
    index: int
    section: str | None
    key: str
    commented: bool
    value: str
    comment: str
    indent: str


def _scan(lines: list[str]) -> tuple[list[_Setting], dict[str, int], str] | None:
    """The settings in a job file, active and commented out, and where each section starts.

    ``None`` when the file uses YAML this can't edit line by line.
    """
    settings: list[_Setting] = []
    headers: dict[str, int] = {}
    section: str | None = None
    child = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        found = SETTING.fullmatch(line)
        if stripped.startswith("#"):
            # A commented-out setting is something like "# key: value"; prose isn't.
            if found and found["hash"] and (section is None or found["indent"]):
                value, comment = _split(found["rest"])
                where = section if found["indent"] else None
                settings.append(
                    _Setting(i, where, found["key"], True, value, comment, found["indent"])
                )
            continue
        if not found or found["hash"] or "\t" in found["indent"]:
            return None
        value, comment = _split(found["rest"])
        if value[:1] in ("[", "{", "|", ">", "&", "*", "!") or value[:1] == "-":
            return None
        if not found["indent"]:
            if value and found["key"] in SECTIONS:
                return None  # such as "target: null"
            if value:
                section = None
            elif found["key"] in SECTIONS:
                section = found["key"]
                headers[section] = i
            else:
                return None
            settings.append(_Setting(i, None, found["key"], False, value, comment, ""))
            continue
        if section is None or not value or (child and found["indent"] != child):
            return None
        child = found["indent"]
        settings.append(_Setting(i, section, found["key"], False, value, comment, child))
    return settings, headers, child or "  "


def _edit(text: str, spec: JobSpec) -> str | None:
    import yaml

    lines = text.split("\n")
    scanned = _scan(lines)
    if scanned is None:
        return None
    settings, headers, child = scanned
    full = spec.model_dump(mode="json", exclude_none=True)
    wanted = spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    inserts: list[tuple[int, int, str]] = []

    def put(s: _Setting, value: Any) -> None:
        head = f"{s.indent}{s.key}: {_value(value)}"
        if s.comment:
            note = s.comment.lstrip()
            column = len(lines[s.index]) - len(note)
            head = head.ljust(column) + note if len(head) < column else f"{head}  {note}"
        lines[s.index] = head

    def comment_out(s: _Setting) -> None:
        line = lines[s.index]
        head = line[: len(line) - len(s.comment)]
        gap = s.comment[: len(s.comment) - len(s.comment.lstrip())]
        # Keep the comment where it was: take back the two columns "# " adds.
        if len(gap) > 2:
            gap = gap[2:]
        lines[s.index] = f"{s.indent}# {head[len(s.indent) :]}{gap}{s.comment.lstrip()}"

    def apply(section: str | None, desired: dict[str, Any], minimal: dict[str, Any]) -> None:
        mine = [s for s in settings if s.section == section]
        active = {s.key: s for s in mine if not s.commented and (section or s.key not in SECTIONS)}
        for key, s in active.items():
            if key not in desired:
                comment_out(s)
                continue
            try:
                same = yaml.safe_load(s.value) == desired[key]
            except yaml.YAMLError:
                same = False
            if not same:
                put(s, desired[key])
        for key, value in minimal.items():
            if key in active or isinstance(value, dict):
                continue
            examples = [s for s in mine if s.commented and s.key == key]
            if examples:
                put(examples[0], value)
                continue
            if section is None:
                first = min((s.index for s in settings if not s.commented), default=len(lines))
                inserts.append((first, len(inserts), f"{key}: {_value(value)}"))
            elif section in headers:
                last = max([headers[section], *(s.index for s in mine)])
                inserts.append((last + 1, len(inserts), f"{child}{key}: {_value(value)}"))

    top = {k: v for k, v in full.items() if not isinstance(v, dict)}
    apply(None, top, {k: v for k, v in wanted.items() if not isinstance(v, dict)})
    for section in SECTIONS:
        apply(section, full.get(section, {}), wanted.get(section, {}))

    for index, _, line in sorted(inserts, reverse=True):
        lines.insert(index, line)
    missing = [s for s in SECTIONS if wanted.get(s) and s not in headers]
    if missing:
        while lines and not lines[-1].strip():
            lines.pop()
        for section in missing:
            lines += ["", f"{section}:"]
            lines += [f"{child}{k}: {_value(v)}" for k, v in wanted[section].items()]
        lines.append("")
    return "\n".join(lines)
