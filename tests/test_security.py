"""Tests that pin down vecshift's security properties, so a regression fails CI."""

import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path

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


def test_reports_have_a_strict_content_security_policy() -> None:
    page = render_html(run_checks(make_profile()))
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
