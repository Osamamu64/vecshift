# Notes for Claude

## Commits

- Author commits as `Osama Elshareef <Osamamu64@gmail.com>`.
- Never add `Co-Authored-By` or `Claude-Session` trailers to commit messages, and no
  "Generated with Claude Code" lines in pull request descriptions.

## Checks before pushing

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Integration tests need `VECSHIFT_TEST_PG_DSN`, and the Qdrant ones also
`VECSHIFT_TEST_QDRANT_URL`; see CONTRIBUTING.md.

## Security rules

Read `docs/security.md` first. In particular:

- Secrets come only from environment variables. Never print, log, or put them in reports,
  errors, or job files. Show connection strings via `ConnectionSettings.display`.
- SQL: bind values as parameters; compose identifiers with `psycopg.sql`, or with
  `TargetState.q()` for SQL shown to users. Never format values into SQL strings.
- HTML: escape everything with `html_kit.e` or `rich`; reports keep their CSP, so new
  scripts must go in `assets/report.js`.
- Ask before sending user text to a remote service (see `cli_bench._confirm`).
- Pin new GitHub Actions to commit SHAs with a version comment.
- `tests/test_security.py` pins these properties; extend it with any new one.

## Architecture rule

`src/vecshift/core` and `src/vecshift/service` must not import from `vecshift.cli*`,
`vecshift.ui`, typer, rich, or any other presentation layer (`tests/test_architecture.py`).
Command logic lives in `vecshift.service` and raises `ServiceError`; CLI commands render.
`src/vecshift/server` (the REST API) may import the service layer, never the CLI. Responses
never carry a secret value; `docs/openapi.json` is its contract (see `docs/api.md`).
