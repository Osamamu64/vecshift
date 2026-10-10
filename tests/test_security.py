"""Tests that pin down vecshift's security properties, so a regression fails CI."""

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tests.test_doctor_checks import make_profile
from vecshift.bench.generate import GenerationError, QueryGenerator
from vecshift.cli import app
from vecshift.connectors.pgvector.target import quote_ident
from vecshift.doctor import run_checks
from vecshift.doctor.html import render_html
from vecshift.embeddings import EmbeddingCache, EmbeddingError, SpecError, parse_spec
from vecshift.embeddings.providers import OpenAICompatEmbedder
from vecshift.html_kit import asset

KEY = "sk-test-never-print-me"


# --- Credentials never travel in the clear or end up in reports


def test_credentials_in_urls_are_rejected() -> None:
    with pytest.raises(SpecError, match="Don't put credentials in url="):
        parse_spec("compat/m,url=https://user:pass@api.example.com/v1")


def test_url_query_strings_are_hidden_in_displayed_specs() -> None:
    spec = parse_spec("compat/m,url=https://api.example.com/v1?api_key=abc123")
    assert "abc123" not in spec.safe_raw and "api.example.com/v1?" in spec.safe_raw


def test_compat_sends_no_key_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECSHIFT_API_KEY", KEY)
    spec = parse_spec("compat/m,url=https://api.example.com/v1")
    assert spec.key_env is None
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAICompatEmbedder(
        spec, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    import asyncio

    asyncio.run(embedder.embed(["x"]))
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("cls", [OpenAICompatEmbedder, QueryGenerator])
def test_keys_are_never_sent_over_plain_http_to_remote_hosts(
    monkeypatch: pytest.MonkeyPatch, cls: type
) -> None:
    monkeypatch.setenv("REMOTE_KEY", KEY)
    remote = parse_spec("compat/m,url=http://api.example.com/v1,key_env=REMOTE_KEY")
    with pytest.raises((EmbeddingError, GenerationError), match="unencrypted"):
        cls(remote)
    local = parse_spec("compat/m,url=http://localhost:8080/v1,key_env=REMOTE_KEY")
    cls(local)  # local traffic doesn't leave the machine


def test_redirects_are_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    embedder = OpenAICompatEmbedder(parse_spec("openai/text-embedding-3-small"))
    client = embedder._client
    assert client.follow_redirects is False


def test_tracebacks_never_show_local_variables() -> None:
    assert app.pretty_exceptions_show_locals is False


def test_password_on_command_line_warns(tmp_path: Path) -> None:
    runner = CliRunner()
    dsn = "postgresql://u:hunter2@127.0.0.1:1/db"
    flagged = runner.invoke(app, ["doctor", "--dsn", dsn])
    assert "visible to other users" in flagged.output and "hunter2" not in flagged.output
    quiet = runner.invoke(app, ["doctor"], env={"VECSHIFT_DSN": dsn})
    assert "visible to other users" not in quiet.output and "hunter2" not in quiet.output


# --- Data at rest


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_cache_is_private_to_its_owner(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache" / "embeddings.sqlite")
    cache.put_many([(b"k", [1.0])])
    assert stat.S_IMODE((tmp_path / "cache").stat().st_mode) == 0o700
    assert stat.S_IMODE(cache.path.stat().st_mode) == 0o600


def test_cache_does_not_store_text(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    secret_text = "patient 4471 diagnosis"
    cache.put_many([(EmbeddingCache.key("m", "document", secret_text), [0.5])])
    cache.close()
    assert secret_text.encode() not in (tmp_path / "c.sqlite").read_bytes()


# --- Generated SQL


@pytest.mark.parametrize(
    ("name", "quoted"),
    [
        ("embedding_v2", "embedding_v2"),
        ("order", '"order"'),
        ("user", '"user"'),
        ("Embedding", '"Embedding"'),
        ('we"ird', '"we""ird"'),
        ("has space", '"has space"'),
    ],
)
def test_identifiers_are_quoted_when_needed(name: str, quoted: str) -> None:
    assert quote_ident(name) == quoted


# --- HTML reports


def _eval_page() -> str:
    from tests.test_eval import run_fake
    from vecshift.eval.html import render_html as render_eval

    return render_eval(run_fake(), job="docs")


@pytest.mark.parametrize("make_page", [lambda: render_html(run_checks(make_profile())), _eval_page])
def test_reports_have_a_strict_content_security_policy(make_page: Any) -> None:
    page = make_page()
    assert not re.search(r'(src|href)="(https?:)?//', page), "nothing loads from the network"
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page)
    assert policy, "missing CSP"
    rules = dict(r.strip().split(" ", 1) for r in policy.group(1).split(";"))
    assert rules["default-src"] == "'none'"
    assert "unsafe-inline" not in rules["script-src"]
    script = asset("report.js")
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert rules["script-src"] == f"'sha256-{digest}'"
    assert f"<script>{script}</script>" in page  # the hash covers exactly what's served


def test_job_files_never_hold_connection_strings() -> None:
    from vecshift.jobs import JobSpec

    with pytest.raises(ValueError, match=r"unknown setting|Extra inputs"):
        JobSpec.model_validate(
            {"source": {"table": "t", "dsn": "postgresql://u:p@h/db"}, "model": "ollama/m"}
        )


def test_json_reports_hold_no_password() -> None:
    result = CliRunner().invoke(
        app, ["doctor", "--json"], env={"VECSHIFT_DSN": "postgresql://u:hunter2@127.0.0.1:1/db"}
    )
    assert "hunter2" not in result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stdout or "x")  # connection failed: nothing (secret or not) on stdout


def test_sql_is_never_built_with_string_formatting() -> None:
    """Statements are composed with psycopg.sql and bound parameters, never f-strings."""
    import ast

    import vecshift

    reviewed = {
        "cache.py": "a run of '?' placeholders; the values are bound",
        "writer.py": "a random hex dollar-quote tag for the trigger function body",
    }
    offenders = []
    for path in Path(vecshift.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            first = node.args[0]
            formatted = isinstance(first, ast.JoinedStr) or (
                isinstance(first, ast.BinOp) and isinstance(first.op, ast.Mod | ast.Add)
            )
            if name in {"execute", "executemany", "SQL"} and formatted:
                offenders.append(path.name)
    assert sorted(offenders) == sorted(reviewed), "review new string-built SQL"


# --- Apply


def test_apply_asks_before_changing_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --yes, apply needs a person at a terminal before it writes or sends text."""
    from types import SimpleNamespace

    import vecshift.cli_apply as cli_apply
    from vecshift.connectors import pgvector
    from vecshift.planning.plan import Estimates

    ready = SimpleNamespace(
        plan=SimpleNamespace(
            ok=True,
            dimensions=8,
            findings=[],
            changes=[],
            estimates=Estimates(),
            source="public.documents (embedding)",
            metric="cosine",
        ),
        job=SimpleNamespace(name="docs", limits=SimpleNamespace(budget_usd=None)),
        spec=SimpleNamespace(price=None, is_local=False, url="https://api.example.com/v1"),
        target=SimpleNamespace(primary_key=("id",)),
        profile=SimpleNamespace(text_field="content"),
    )
    monkeypatch.setattr(cli_apply, "prepare", lambda job_file: ready)

    def connect(settings: object) -> None:
        raise AssertionError("connected without confirmation")

    monkeypatch.setattr(pgvector, "connect_writer", connect)
    result = CliRunner().invoke(app, ["apply", "j.yaml"])
    assert result.exit_code == 2 and "--yes" in result.output
    assert "sends row text to https://api.example.com/v1" in result.output


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_apply_state_is_private_and_holds_no_text(tmp_path: Path) -> None:
    import asyncio

    from tests.test_migrate import FakeModel, FakeTable
    from vecshift.migrate import JobState, apply

    table = FakeTable(5)
    table.rows[3][0] = "patient 4471 diagnosis"
    state = JobState.for_job(tmp_path / "j.yaml", "docs")
    asyncio.run(
        apply(
            table,
            FakeModel(reject="patient"),
            state,
            dims=4,
            price_per_million=1.0,
            budget_usd=None,
            chunk_rows=10,
            index=None,
        )
    )
    assert state.failed, "the rejected row is recorded by ID"
    assert stat.S_IMODE(state.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state.path.parent.stat().st_mode) == 0o700
    assert "patient" not in state.path.read_text() and "text " not in state.path.read_text()


def test_trigger_functions_pin_their_search_path() -> None:
    """vecshift's trigger functions can't be redirected by objects in other schemas."""
    from unittest.mock import MagicMock

    from vecshift.connectors.pgvector.writer import Layout, PgWriter

    conn = MagicMock(autocommit=True)
    lay = Layout("public", "docs", "id", "content", "embedding_v2", "vector", 8, "extensions")
    text = PgWriter(conn, lay).sync_function_sql().as_string()
    assert 'SET search_path = pg_catalog, "extensions"' in text
    assert text.count("$vs_") == 2, "the body is dollar-quoted with a random tag"


@pytest.mark.parametrize(
    ("command", "changes"),
    [
        ("cutover", ["_catch_up(", "switch.cutover("]),
        ("rollback", ["switch.rollback("]),
        ("cleanup", ["switch.cleanup("]),
    ],
)
def test_switching_asks_before_changing_or_sending(command: str, changes: list[str]) -> None:
    """cutover, rollback, and cleanup confirm before embedding text or touching the table.

    The integration tests check that, without --yes, nothing changes.
    """
    import vecshift.cli_cutover as cli_cutover

    source = Path(cli_cutover.__file__).read_text(encoding="utf-8")
    run = source.split(f"def {command}(", 1)[1].split("def run(", 1)[1]
    asks = run.index("_confirm(") if "_confirm(" in run else run.index("typer.prompt(")
    for change in changes:
        assert asks < run.index(change)


def test_eval_asks_before_sending_queries() -> None:
    """Query text goes to remote models only after confirmation (or --yes)."""
    import typer

    from vecshift.cli_eval import _confirm

    sends = [("openai/text-embedding-3-large", "https://api.openai.com/v1")]
    with pytest.raises(typer.Exit):
        _confirm(sends, 10, yes=False)  # no terminal in tests
    _confirm(sends, 10, yes=True)
    _confirm([("ollama/nomic-embed-text", "")], 10, yes=False)


def test_eval_only_reads() -> None:
    """eval connects read-only, so the database itself rejects any write."""
    import vecshift.cli_eval as cli_eval

    source = Path(cli_eval.__file__).read_text(encoding="utf-8")
    assert "pgvector.connect(settings)" in source and "connect_writer" not in source


def test_status_only_reads() -> None:
    """status connects read-only, like eval, and never saves job state."""
    import vecshift.cli_status as cli_status

    source = Path(cli_status.__file__).read_text(encoding="utf-8")
    assert "pgvector.connect(settings)" in source and "connect_writer" not in source
    assert ".save(" not in source


def test_secrets_typed_into_init_are_hidden_and_saved_only_on_request() -> None:
    import vecshift.cli_init as cli_init
    import vecshift.ui as ui

    source = Path(cli_init.__file__).read_text(encoding="utf-8")
    assert 'ui.secret("Connection string")' in source
    key_prompt = source[source.index("def _model_key") :]
    assert "ui.secret(" in key_prompt and "typer.prompt(" not in source
    # Saving asks first, and defaults to no.
    save = source[source.index("def _offer_to_save") : source.index("def _keep_out_of_git")]
    assert "ui.confirm(" in save and "default=False" in save
    # ui.secret never echoes, in the arrow-key UI or the plain fallback.
    helper = Path(ui.__file__).read_text(encoding="utf-8")
    secret = helper[helper.index("def secret") : helper.index("def confirm")]
    assert "questionary.password(" in secret and "hide_input=True" in secret


def test_env_files_are_private_and_never_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vecshift import envfile

    path = tmp_path / ".env"
    envfile.save(path, "VECSHIFT_DSN", "postgresql://u:" + KEY + "@h/db")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    monkeypatch.setenv("VECSHIFT_DSN", "from-the-shell")
    assert envfile.load(path) == []
    assert os.environ["VECSHIFT_DSN"] == "from-the-shell"


# --- Images and workflows run pinned, least-privileged code

REPO = Path(__file__).resolve().parent.parent
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def test_docker_image_is_pinned_and_runs_as_a_user() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    default = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", dockerfile, re.M)
    assert default and DIGEST.search(default.group(1))
    assert re.findall(r"^FROM (\S+)", dockerfile, re.M) == ["${PYTHON_IMAGE}"] * 2
    users = re.findall(r"^USER (\S+)$", dockerfile, re.M)
    assert users and users[-1] not in {"root", "0"}
    assert "--require-hashes" in dockerfile, "dependencies are checked against the lock file"


def test_demo_keeps_its_database_private() -> None:
    import yaml

    compose = yaml.safe_load((REPO / "demo" / "compose.yaml").read_text(encoding="utf-8"))
    for name, service in compose["services"].items():
        assert "ports" not in service, f"{name} publishes a port"
        assert "privileged" not in service and "network_mode" not in service
    image = re.fullmatch(r"\$\{\w+:-(.+)\}", compose["services"]["db"]["image"])
    assert image and DIGEST.search(image.group(1))


def test_workflow_actions_are_pinned_to_commits() -> None:
    for workflow in (REPO / ".github" / "workflows").glob("*.yml"):
        for line in workflow.read_text(encoding="utf-8").splitlines():
            if match := re.search(r"uses:\s*(\S+)(.*)", line):
                action, comment = match.groups()
                assert re.search(r"@[0-9a-f]{40}$", action), f"{workflow.name}: {action}"
                assert re.match(r"\s*# v\d", comment), f"{workflow.name}: {action} needs # vX"
