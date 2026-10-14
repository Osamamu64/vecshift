"""The ``vecshift init`` command: write a job file, from flags or by asking.

With ``--table`` (or ``--collection``) and ``--model`` it writes the file and asks nothing,
for scripts. At a terminal without them, it asks what to migrate, connects, lists the
vectors it finds, and asks about the rest: a pgvector column into a new column, a Qdrant
collection into a new collection, or a pgvector table into Qdrant. Secrets are typed
hidden, never echoed, and never written to the job file. They're saved to ``.env`` only if
the user says so.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from vecshift import __version__, ui
from vecshift.cli_plan import DEFAULT_JOB, _fail

if TYPE_CHECKING:
    import psycopg

    from vecshift.connectors.pgvector import ConnectionSettings, VectorColumn
    from vecshift.connectors.qdrant import QdrantClient, VectorField

DSN_VARIABLES = ("VECSHIFT_DSN", "DATABASE_URL")
QDRANT_URL_ENV = "VECSHIFT_QDRANT_URL"
QDRANT_URL_EXAMPLE = "http://localhost:6333"
CONNECT_TRIES = 3
TEXT_SAMPLE = 64

# Offered in this order; the first is the default. Anything else can be typed as a spec.
MODEL_CHOICES = (
    ("openai/text-embedding-3-small", "OpenAI · 1,536 dims · $0.02 per 1M tokens"),
    ("openai/text-embedding-3-large,dims=1024", "OpenAI · 1,024 dims · $0.13 per 1M tokens"),
    ("ollama/bge-m3", "Ollama on this machine · 1,024 dims · multilingual · free"),
    ("ollama/nomic-embed-text", "Ollama on this machine · 768 dims · free"),
    ("hash/256", "built-in test model · offline · free · for trying vecshift out"),
)


def interactive() -> bool:
    """Whether a person is at the terminal to answer questions."""
    return sys.stdin.isatty() and sys.stdout.isatty()


class Store(StrEnum):
    PGVECTOR = "pgvector"
    QDRANT = "qdrant"


Route = tuple[Store, Store]
"""Where the vectors are, and where the new ones go."""

# Offered in this order; the first is the default.
ROUTES: tuple[tuple[Route, str, str], ...] = (
    ((Store.PGVECTOR, Store.PGVECTOR), "A pgvector column", "into a new column of the same table"),
    ((Store.QDRANT, Store.QDRANT), "A Qdrant collection", "into a new collection, behind an alias"),
    ((Store.PGVECTOR, Store.QDRANT), "A pgvector table, into Qdrant", "into a new collection"),
)


@dataclass(slots=True)
class _Answers:
    model: str
    route: Route = (Store.PGVECTOR, Store.PGVECTOR)
    table: str | None = None
    collection: str | None = None
    """For a Qdrant source: the alias (or collection) the application searches."""
    vector_column: str | None = None
    vector: str | None = None
    text_column: str | None = None
    """The text column, or for a Qdrant source the payload field."""
    source_model: str | None = None
    alias: str | None = None
    new_collection: str | None = None
    budget_usd: float | None = None
    dsn_env: str = "VECSHIFT_DSN"


@dataclass(slots=True)
class _Steps:
    total: int
    done: int = 0

    def next(self, title: str) -> None:
        self.done += 1
        ui.step(self.done, self.total, title)


def init(
    ctx: typer.Context,
    table: Annotated[
        str | None,
        typer.Option(help="Table holding the vectors, as table or schema.table. Asked if missing."),
    ] = None,
    collection: Annotated[
        str | None,
        typer.Option(
            help="Qdrant collection or alias holding the vectors, instead of a table. "
            "Asked if missing."
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            help="The new model, e.g. openai/text-embedding-3-large,dims=1024. Asked if missing."
        ),
    ] = None,
    to: Annotated[
        Store | None,
        typer.Option(
            "--to",
            help="Where the new vectors go: pgvector (a new column next to the old one) or "
            "qdrant (a new collection). Defaults to where they are now.",
            show_default=False,
        ),
    ] = None,
    column: Annotated[
        str, typer.Option(help="Column for the new vectors, for a pgvector target.")
    ] = "embedding_v2",
    alias: Annotated[
        str | None,
        typer.Option(
            help="For a Qdrant target: the alias cutover points at the new collection. "
            "Asked if needed."
        ),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the job.")] = (
        DEFAULT_JOB
    ),
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Write a job file to start a migration from. At a terminal, it guides you through it."""
    if table and collection:
        raise _fail("Pass --table or --collection, not both.")
    route = _route(table, collection, to)
    if model and route is not None:
        _check_flags(ctx, route, alias)
        answers = _Answers(model=model, route=route, table=table, collection=collection)
        answers.alias = alias
        _write(output, _template(column, answers), force=force)
        typer.echo(f"Wrote {output}. Review it, then run: vecshift plan {output}")
        return
    if not interactive():
        raise _fail(
            "init needs --model and --table (or --collection) when it can't ask.",
            "Pass them, or run vecshift init in a terminal to be guided through it.",
        )
    _guided(ctx, table, collection, model, to, column, alias, output, force)


def _route(table: str | None, collection: str | None, to: Store | None) -> Route | None:
    """The migration the flags decide, or None to ask."""
    if collection:
        if to is Store.PGVECTOR:
            raise _fail(
                "A Qdrant collection can only be migrated into Qdrant.",
                "Drop --to pgvector.",
            )
        return Store.QDRANT, Store.QDRANT
    if table:
        return Store.PGVECTOR, to or Store.PGVECTOR
    return None


def _check_flags(ctx: typer.Context, route: Route, alias: str | None) -> None:
    source = ctx.get_parameter_source("column")
    if route[1] is Store.QDRANT and source is not None and source.name == "COMMANDLINE":
        raise _fail(
            "--column is for a pgvector target; the new vectors go in a new Qdrant collection.",
            "Drop --column. To name the collection, set target.collection in the job file.",
        )
    if route[1] is Store.PGVECTOR and alias:
        raise _fail(
            "--alias is for a migration into Qdrant.",
            "Add --to qdrant to copy the table into a new Qdrant collection.",
        )


def _template(column: str, answers: _Answers) -> str:
    import yaml
    from pydantic import ValidationError

    from vecshift.jobs.spec import JobSpec, format_errors, qdrant_template, template

    source, target = answers.route
    if target is Store.PGVECTOR:
        text = template(
            answers.table or "",
            answers.model,
            column,
            vector_column=answers.vector_column,
            text_column=answers.text_column,
            source_model=answers.source_model,
            budget_usd=answers.budget_usd,
        )
    else:
        text = qdrant_template(
            answers.model,
            collection=answers.collection if source is Store.QDRANT else None,
            table=answers.table if source is Store.PGVECTOR else None,
            vector=answers.vector,
            text_field=answers.text_column,
            vector_column=answers.vector_column,
            source_model=answers.source_model,
            alias=answers.alias,
            new_collection=answers.new_collection,
            budget_usd=answers.budget_usd,
        )
    if source is Store.PGVECTOR and answers.dsn_env != "VECSHIFT_DSN":
        text = text.replace("dsn_env: VECSHIFT_DSN", f"dsn_env: {answers.dsn_env}", 1)
    try:
        JobSpec.model_validate(yaml.safe_load(text))
    except ValidationError as exc:
        raise _fail(f"Those settings aren't valid:\n{format_errors(exc)}") from exc
    return text


def _write(output: Path, text: str, *, force: bool) -> None:
    if output.exists() and not force:
        raise _fail(f"{output} already exists.", "Pass --force to overwrite it.")
    try:
        output.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise _fail(f"Couldn't write {output}: {exc.strerror}") from exc


# --- the guided flow


def _guided(
    ctx: typer.Context,
    table: str | None,
    collection: str | None,
    model: str | None,
    to: Store | None,
    column: str,
    alias: str | None,
    output: Path,
    force: bool,
) -> None:
    ui.banner(
        __version__,
        "Let's set up a migration.",
        [("Job file", str(output)), ("Changes", "none until you run vecshift apply")],
    )
    if output.exists() and not force:
        if not ui.confirm(f"{output} already exists. Replace it?", default=False):
            raise typer.Exit(1)
        force = True

    route = _route(table, collection, to) or _choose_route(to)
    _check_flags(ctx, route, alias)
    source, target = route
    answers = _Answers(model="", route=route)
    typed: list[tuple[str, str, str]] = []  # (variable, value, what), saved only on request
    steps = _Steps(5 if source != target else 4)

    if source is Store.PGVECTOR:
        steps.next("Database")
        dsn_env, pasted, settings = _connection()
        answers.dsn_env = dsn_env
        if pasted is not None:
            typed.append((dsn_env, pasted, "connection string"))
        steps.next("Vectors")
        vectors, text = _pg_vectors(settings, table)
        answers.table = f"{vectors.schema}.{vectors.table}"
        answers.vector_column = vectors.column
        answers.text_column = text
        if target is Store.QDRANT:
            steps.next("Qdrant")
            client = _qdrant_connection(typed)
            with client:
                _into_qdrant(client, vectors.table, alias, answers)
    else:
        steps.next("Qdrant")
        client = _qdrant_connection(typed)
        steps.next("Vectors")
        with client:
            _qdrant_vectors(client, collection, alias, answers)

    steps.next("Models")
    answers.source_model = _ask_spec(
        "Which model made the current vectors? (optional, lets eval compare; Enter to skip)",
        optional=True,
    )
    answers.model = model or _choose_model()
    answers.budget_usd = _ask_budget(answers.model)

    steps.next("Save")
    _write(output, _template(column, answers), force=force)
    ui.success(f"Wrote {output}")

    env_file = _env_file(ctx)
    for name, value, what in typed:
        _offer_to_save(env_file, name, value, what)
        os.environ.setdefault(name, value)  # for this process, so plan can run below
    _model_key(env_file, answers.model)

    if ui.confirm("Check the plan now? It only reads, and changes nothing.", default=True):
        from vecshift.cli_plan import plan

        typer.echo()
        ctx.invoke(plan, job_file=output)
    else:
        ui.note(f"Next: vecshift plan {output}")


def _choose_route(to: Store | None) -> Route:
    routes = [r for r in ROUTES if to is None or r[0][1] is to]
    if len(routes) == 1:
        return routes[0][0]
    # Someone with only Qdrant set up most likely has their vectors there.
    only_qdrant = os.environ.get(QDRANT_URL_ENV) and not any(
        os.environ.get(v) for v in DSN_VARIABLES
    )
    default = next((i for i, r in enumerate(routes) if only_qdrant and r[0][0] is Store.QDRANT), 0)
    picked = ui.select(
        "What do you want to migrate?", [(label, hint) for _, label, hint in routes], default
    )
    return routes[picked][0]


def _connection() -> tuple[str, str | None, ConnectionSettings]:
    """The variable to read the database from, the string if it was typed here, and settings.

    The typed string is held in memory only; the caller decides whether to save it.
    """
    from vecshift.connectors import pgvector

    for variable in DSN_VARIABLES:
        value = os.environ.get(variable)
        if value:
            try:
                settings = pgvector.prepare(value)
            except pgvector.ConnectError as exc:
                raise _fail(f"{variable} isn't usable: {exc}", exc.hint) from exc
            ui.success(f"Using {settings.display} from {variable}")
            return variable, None, settings

    ui.note(
        "Paste your PostgreSQL connection string, such as the session pooler string from "
        "Supabase's Connect button. It's hidden as you type and never written to the job file."
    )
    for attempt in range(1, CONNECT_TRIES + 1):
        value = ui.secret("Connection string")
        try:
            settings = pgvector.prepare(value)
            with ui.working("Connecting…"):
                conn = pgvector.connect(settings)
        except pgvector.ConnectError as exc:
            ui.error(f"Couldn't connect: {exc}", exc.hint)
            if attempt == CONNECT_TRIES:
                raise typer.Exit(2) from exc
            continue
        conn.close()
        ui.success(f"Connected to {settings.display}")
        return "VECSHIFT_DSN", value, settings
    raise typer.Exit(2)  # pragma: no cover - the loop always returns or raises


def _pg_vectors(settings: ConnectionSettings, table: str | None) -> tuple[VectorColumn, str]:
    """The vector column to re-embed, and the text column it comes from."""
    from vecshift.connectors import pgvector

    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        source = _choose_column(conn, table)
        return source, _choose_text(conn, source)
    finally:
        conn.rollback()
        conn.close()


def _estimate(conn: psycopg.Connection, relid: int) -> str:
    row = conn.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = %s", (relid,)).fetchone()
    return f"~{int(row[0]):,} rows" if row and row[0] >= 0 else "rows unknown"


def _choose_column(conn: psycopg.Connection, table: str | None) -> VectorColumn:
    from vecshift.connectors.pgvector import find_vector_columns

    columns = find_vector_columns(conn)
    if table:
        schema, _, name = table.rpartition(".")
        columns = [c for c in columns if c.table == name and (not schema or c.schema == schema)]
    if not columns:
        where = f" in {table}" if table else ""
        raise _fail(
            f"No vector columns found{where}.",
            "vecshift migrates pgvector columns. Check the database and the role's access.",
        )
    choices = [
        (c.qualified, f"{c.type}({c.dimensions or '?'}) · {_estimate(conn, c.relid)}")
        for c in columns
    ]
    if len(columns) == 1:
        ui.success(f"Found {choices[0][0]}  {choices[0][1]}")
        return columns[0]
    return columns[ui.select("Which vector column do you want to re-embed?", choices)]


def _choose_text(conn: psycopg.Connection, source: VectorColumn) -> str:
    from vecshift.connectors.pgvector.inspect import TEXT_COLUMNS, TEXT_TYPES, _columns, _pick

    columns = _columns(conn, source.relid)
    texts = [name for name, kind in columns.items() if kind in TEXT_TYPES]
    if not texts:
        raise _fail(
            f"{source.schema}.{source.table} has no text column to re-embed from.",
            "vecshift needs the source text next to the vectors.",
        )
    guess = _pick(columns, TEXT_COLUMNS, TEXT_TYPES) or texts[0]
    if len(texts) == 1:
        ui.success(f"Text comes from {guess}")
        return guess
    choices = [(name, "detected" if name == guess else "") for name in texts]
    picked = ui.select(
        "Which column holds the text the vectors were made from?",
        choices,
        default=texts.index(guess),
    )
    return texts[picked]


# --- Qdrant


def _qdrant_url_problem(value: str) -> str | None:
    from vecshift.connectors import qdrant

    try:
        qdrant.prepare(value)
    except qdrant.QdrantError as exc:
        return str(exc)
    return None


def _qdrant_connection(typed: list[tuple[str, str, str]]) -> QdrantClient:
    """A client for the Qdrant to use, asking for the URL and key when they aren't set.

    A URL or key typed here is added to ``typed``; the caller decides whether to save it.
    """
    from vecshift.connectors import qdrant

    from_env = os.environ.get(QDRANT_URL_ENV) or None
    url, key = from_env, None
    if from_env is None:
        ui.note(f"Type your Qdrant's REST URL, such as {QDRANT_URL_EXAMPLE} or a cloud cluster's.")
    for attempt in range(1, CONNECT_TRIES + 1):
        if url is None:
            url = ui.text("Qdrant URL", default=QDRANT_URL_EXAMPLE, validate=_qdrant_url_problem)
        try:
            settings = qdrant.prepare(url, key=key)
        except qdrant.QdrantError as exc:
            if from_env:
                raise _fail(f"{QDRANT_URL_ENV} isn't usable: {exc}", exc.hint) from exc
            ui.error(str(exc), exc.hint)
            url = key = None
            continue
        client = qdrant.QdrantClient(settings)
        try:
            with ui.working("Connecting…"):
                client.collections()
        except qdrant.QdrantError as exc:
            client.close()
            needs_key = exc.status in {401, 403} and not os.environ.get(settings.key_env)
            if needs_key and attempt < CONNECT_TRIES:
                if key is None:
                    ui.note(
                        "This Qdrant needs an API key. It's hidden as you type and never "
                        "written to the job file."
                    )
                else:
                    ui.error("Qdrant refused that key.")
                key = ui.secret("Qdrant API key")
                continue
            if from_env and not needs_key:
                raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
            ui.error(f"Couldn't connect: {exc}", exc.hint)
            if attempt == CONNECT_TRIES:
                raise typer.Exit(2) from exc
            if not from_env:
                url = key = None
            continue
        where = f" from {QDRANT_URL_ENV}" if from_env else ""
        ui.success(f"Connected to Qdrant at {settings.display}{where}")
        if from_env is None:
            typed.append((QDRANT_URL_ENV, settings.url, "Qdrant URL"))
        if key:
            typed.append((settings.key_env, key, "Qdrant API key"))
        return client
    raise typer.Exit(2)


def _free(base: str, taken: set[str]) -> str:
    """``base``, or ``base_2``, ``base_3``, ... if that name is taken."""
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}_{n}"
    return name


def _name_problem(collections: set[str], aliases: set[str]) -> Callable[[str], str | None]:
    from vecshift.jobs.spec import _COLLECTION

    def check(value: str) -> str | None:
        if not _COLLECTION.match(value):
            return "Use letters, digits, and _ . - only."
        if value in collections:
            return f"{value} is already a collection's name."
        if value in aliases:
            return f"{value} is already an alias."
        return None

    return check


def _qdrant_vectors(
    client: QdrantClient, collection: str | None, alias: str | None, answers: _Answers
) -> None:
    """Choose the vectors to re-embed, the name searches go through, and the text field."""
    from vecshift.connectors.qdrant.inspect import _fields

    collections = client.collections()
    pairs = client.aliases()
    resolved = dict(pairs).get(collection, collection) if collection else None
    found: list[VectorField] = []
    points: dict[str, int | None] = {}
    for name in collections:
        if resolved and name != resolved:
            continue
        info = client.collection(name)
        points[name] = info.get("points_count")
        found.extend(_fields(name, info))
    if not found:
        if collection and resolved not in collections:
            raise _fail(
                f"No collection or alias named {collection!r}.",
                "Check the name: vecshift doctor --qdrant URL lists the vectors it finds.",
            )
        where = f" in {collection}" if collection else " in this Qdrant"
        raise _fail(f"No dense vectors found{where}.", "vecshift migrates dense vectors.")

    def describe(f: VectorField) -> str:
        count = points.get(f.collection)
        rows = f"~{count:,} points" if count is not None else "points unknown"
        names = [a for a, c in pairs if c == f.collection]
        via = f" · alias {', '.join(names)}" if names else ""
        return f"{f.dimensions or '?'} dims · {f.distance or '?'} · {rows}{via}"

    if len(found) == 1:
        field = found[0]
        ui.success(f"Found {field.qualified}  {describe(field)}")
    else:
        choices = [(f.qualified, describe(f)) for f in found]
        field = found[ui.select("Which vectors do you want to re-embed?", choices)]
    siblings = [f for f in found if f.collection == field.collection]
    if len(siblings) > 1:
        answers.vector = field.name
        ui.warn(
            f"{field.collection} has {len(siblings)} named vectors. vecshift can't migrate "
            "a collection with several yet, because the copy would drop the others: plan "
            "will stop."
        )

    names = [a for a, c in pairs if c == field.collection]
    if collection and collection != field.collection:
        searched = collection
    elif len(names) == 1:
        searched = names[0]
    elif names:
        searched = names[
            ui.select("Which alias does your application search?", [(n, "") for n in names])
        ]
    else:
        searched = field.collection
    answers.collection = searched
    if searched != field.collection:
        ui.success(f"Searches go through the alias {searched}")
        answers.alias = alias
    else:
        ui.warn(
            f"Your application searches {field.collection} by name, so there's no alias "
            "for cutover to move."
        )
        answers.alias = _ask_alias(
            alias, f"{field.collection}_live", set(collections), {a for a, _ in pairs}
        )
    answers.new_collection = _new_collection(
        f"{answers.alias or searched}_v2", set(collections), {a for a, _ in pairs}
    )
    answers.text_column = _qdrant_text(client, field)


def _ask_alias(
    alias: str | None,
    suggestion: str,
    collections: set[str],
    aliases: set[str],
) -> str:
    """The alias cutover will create: from --alias, or asked."""
    check = _name_problem(collections, set())
    if alias:
        problem = check(alias)
        if problem:
            raise _fail(f"--alias {alias}: {problem}")
        return alias
    ui.note(
        "Cutover creates an alias and points it at the new collection; from then on, your "
        "application searches the alias. Qdrant can't give an alias the name of an existing "
        "collection."
    )
    return ui.text(
        "Name for the alias", default=_free(suggestion, collections | aliases), validate=check
    )


def _new_collection(default: str, collections: set[str], aliases: set[str]) -> str | None:
    """None to keep the default name for the new collection, or another when it's taken."""
    if default not in collections and default not in aliases:
        return None
    ui.warn(f"A collection or alias named {default} already exists.")
    return ui.text(
        "Name for the new collection",
        default=_free(default, collections | aliases),
        validate=_name_problem(collections, aliases),
    )


def _qdrant_text(client: QdrantClient, field: VectorField) -> str:
    """The payload field holding the text, from a sample of points."""
    from vecshift.connectors.qdrant.inspect import (
        MODEL_KEYS,
        TEXT_KEYS,
        UPDATED_AT_KEYS,
    )

    sample, _ = client.scroll(field.collection, TEXT_SAMPLE, vector=False)
    skip = set(MODEL_KEYS) | set(UPDATED_AT_KEYS)
    counts: Counter[str] = Counter()
    for point in sample:
        for key, value in (point.get("payload") or {}).items():
            if isinstance(value, str) and value.strip() and key not in skip:
                counts[key] += 1
            elif isinstance(value, dict):
                for inner, text in value.items():
                    if isinstance(text, str) and text.strip() and inner not in skip:
                        counts[f"{key}.{inner}"] += 1
    if not counts:
        raise _fail(
            f"No text found in the payloads of {field.collection}.",
            "vecshift re-embeds the text stored with each point, so it needs it in the payload.",
        )
    fields = [name for name, _ in counts.most_common()]
    guess = next((k for k in TEXT_KEYS if k in counts), fields[0])
    if len(fields) == 1:
        ui.success(f"Text comes from the payload field {guess}")
        return guess
    choices = [
        (
            name,
            f"in {counts[name]} of {len(sample)} points" + (" · detected" if name == guess else ""),
        )
        for name in fields
    ]
    picked = ui.select(
        "Which payload field holds the text the vectors were made from?",
        choices,
        default=fields.index(guess),
    )
    return fields[picked]


def _into_qdrant(client: QdrantClient, table: str, alias: str | None, answers: _Answers) -> None:
    """The alias and collection a pgvector table is copied into."""
    collections = set(client.collections())
    aliases = {a for a, _ in client.aliases()}
    answers.alias = _ask_alias(alias, table, collections, aliases)
    answers.new_collection = _new_collection(f"{answers.alias}_v2", collections, aliases)


# --- models


def _spec_problem(value: str) -> str | None:
    from vecshift.embeddings import SpecError, parse_spec

    try:
        parse_spec(value)
    except SpecError as exc:
        return str(exc)
    return None


def _ask_spec(prompt: str, *, optional: bool = False) -> str | None:
    def check(value: str) -> str | None:
        if not value:
            return None if optional else "Type a model spec."
        return _spec_problem(value)

    value = ui.text(prompt, validate=check)
    return value or None


def _choose_model() -> str:
    choices = [*MODEL_CHOICES, ("Another model", "type its spec, as for vecshift bench")]
    picked = ui.select("Which model should make the new vectors?", choices)
    if picked < len(MODEL_CHOICES):
        return MODEL_CHOICES[picked][0]
    return _ask_spec("Model spec, e.g. compat/my-model,url=http://localhost:8080/v1") or ""


def _ask_budget(model: str) -> float | None:
    from vecshift.embeddings import parse_spec

    if parse_spec(model).price == 0:
        return None

    def check(value: str) -> str | None:
        if not value:
            return None
        try:
            ok = float(value.lstrip("$")) > 0
        except ValueError:
            ok = False
        return None if ok else "Enter an amount above zero, such as 25, or leave it empty."

    value = ui.text(
        "Spending limit in USD? plan refuses a bigger estimate and apply stops there "
        "(Enter for none)",
        validate=check,
    )
    return float(value.lstrip("$")) if value else None


# --- secrets


def _env_file(ctx: typer.Context) -> Path | None:
    """The .env file the CLI reads, or None if --no-env-file was given."""
    params = ctx.find_root().params
    if params.get("no_env_file"):
        return None
    path = params.get("env_file")
    return Path(path) if path else Path(".env")


def _offer_to_save(env_file: Path | None, name: str, value: str, what: str) -> None:
    from vecshift import envfile

    if env_file is not None and ui.confirm(
        f"Save the {what} to {env_file} so later commands find it? Only your user can read it.",
        default=False,
    ):
        try:
            envfile.save(env_file, name, value)
        except OSError as exc:
            ui.error(f"Couldn't write {env_file}: {exc.strerror}")
        else:
            ui.success(f"Saved {name} to {env_file}")
            _keep_out_of_git(env_file)
            return
    ui.note(f"Before the next command, set it in your shell: export {name}='…'")


def _keep_out_of_git(env_file: Path) -> None:
    """Warn, and offer to fix it, if git would commit the file."""
    git = shutil.which("git")
    if git is None:
        return
    try:
        result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "check-ignore", "--quiet", "--", str(env_file)],
            cwd=env_file.parent if str(env_file.parent) else None,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return
    if result.returncode != 1:  # 0: ignored; 128: not in a repository
        return
    ui.warn(f"{env_file} isn't in .gitignore, so git could commit it.")
    ignore = env_file.parent / ".gitignore"
    if ui.confirm(f"Add it to {ignore}?", default=True):
        existing = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
        separator = "" if not existing or existing.endswith("\n") else "\n"
        ignore.write_text(f"{existing}{separator}{env_file.name}\n", encoding="utf-8")
        ui.success(f"Added {env_file.name} to {ignore}")


def _model_key(env_file: Path | None, model: str) -> None:
    """Offer to store the model's API key if it isn't set; apply can't run without it."""
    from vecshift.embeddings import parse_spec

    spec = parse_spec(model)
    if not spec.key_env or os.environ.get(spec.key_env):
        return
    ui.note(f"{spec.name} needs an API key in {spec.key_env}, which isn't set.")
    key = ui.secret(
        f"Paste your {spec.key_env} to save it, or press Enter to set it yourself later",
        allow_empty=True,
    )
    if key:
        _offer_to_save(env_file, spec.key_env, key, "API key")
        os.environ.setdefault(spec.key_env, key)  # for this process only
    else:
        ui.note(f"Set {spec.key_env} before running vecshift apply.")
