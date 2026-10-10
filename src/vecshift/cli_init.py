"""The ``vecshift init`` command: write a job file, from flags or by asking.

With ``--table`` and ``--model`` it writes the file and asks nothing, for scripts. At a
terminal without them, it connects, lists the vector columns it finds, and asks about the
rest. Secrets are typed hidden, never echoed, and never written to the job file. They're
saved to ``.env`` only if the user says so.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from vecshift import __version__, ui
from vecshift.cli_plan import DEFAULT_JOB, _fail

if TYPE_CHECKING:
    import psycopg

    from vecshift.connectors.pgvector import ConnectionSettings, VectorColumn

DSN_VARIABLES = ("VECSHIFT_DSN", "DATABASE_URL")
CONNECT_TRIES = 3

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


STEPS = 4


@dataclass(slots=True)
class _Answers:
    table: str
    vector_column: str | None
    text_column: str | None
    source_model: str | None
    model: str
    budget_usd: float | None
    dsn_env: str


def init(
    ctx: typer.Context,
    table: Annotated[
        str | None,
        typer.Option(help="Table holding the vectors, as table or schema.table. Asked if missing."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            help="The new model, e.g. openai/text-embedding-3-large,dims=1024. Asked if missing."
        ),
    ] = None,
    column: Annotated[str, typer.Option(help="Column for the new vectors.")] = "embedding_v2",
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the job.")] = (
        DEFAULT_JOB
    ),
    force: Annotated[bool, typer.Option(help="Overwrite an existing file.")] = False,
) -> None:
    """Write a job file to start a migration from. At a terminal, it guides you through it."""
    if table and model:
        _write(output, _template(table, model, column), force=force)
        typer.echo(f"Wrote {output}. Review it, then run: vecshift plan {output}")
        return
    if not interactive():
        raise _fail(
            "init needs --table and --model when it can't ask.",
            "Pass both, or run vecshift init in a terminal to be guided through it.",
        )
    _guided(ctx, table, model, column, output, force)


def _template(table: str, model: str, column: str, answers: _Answers | None = None) -> str:
    import yaml
    from pydantic import ValidationError

    from vecshift.jobs.spec import JobSpec, format_errors, template

    text = template(
        table,
        model,
        column,
        vector_column=answers.vector_column if answers else None,
        text_column=answers.text_column if answers else None,
        source_model=answers.source_model if answers else None,
        budget_usd=answers.budget_usd if answers else None,
    )
    if answers and answers.dsn_env != "VECSHIFT_DSN":
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
    model: str | None,
    column: str,
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

    ui.step(1, STEPS, "Database")
    dsn_env, pasted, settings = _connection()
    answers = _ask(settings, table, model, dsn_env)

    ui.step(4, STEPS, "Save")
    _write(output, _template(answers.table, answers.model, column, answers), force=force)
    ui.success(f"Wrote {output}")

    env_file = _env_file(ctx)
    if pasted is not None:
        _offer_to_save(env_file, dsn_env, pasted, "connection string")
        os.environ.setdefault(dsn_env, pasted)  # for this process, so plan can run below
    _model_key(env_file, answers.model)

    if ui.confirm("Check the plan now? It reads the database and changes nothing.", default=True):
        from vecshift.cli_plan import plan

        typer.echo()
        ctx.invoke(plan, job_file=output)
    else:
        ui.note(f"Next: vecshift plan {output}")


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


def _ask(
    settings: ConnectionSettings, table: str | None, model: str | None, dsn_env: str
) -> _Answers:
    from vecshift.connectors import pgvector

    try:
        conn = pgvector.connect(settings)
    except pgvector.ConnectError as exc:
        raise _fail(f"Couldn't connect: {exc}", exc.hint) from exc
    try:
        ui.step(2, STEPS, "Vectors")
        source = _choose_column(conn, table)
        text = _choose_text(conn, source)
    finally:
        conn.rollback()
        conn.close()

    ui.step(3, STEPS, "Models")
    old = _ask_spec(
        "Which model made the current vectors? (optional, lets eval compare; Enter to skip)",
        optional=True,
    )
    new = model or _choose_model()
    budget = _ask_budget(new)
    return _Answers(
        table=f"{source.schema}.{source.table}",
        vector_column=source.column,
        text_column=text,
        source_model=old,
        model=new,
        budget_usd=budget,
        dsn_env=dsn_env,
    )


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
