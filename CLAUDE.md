# Notes for Claude

## Commits

- Author commits as `Osama Elshareef <Osamamu64@gmail.com>`.
- Never add `Co-Authored-By` or `Claude-Session` trailers to commit messages, and no
  "Generated with Claude Code" lines in pull request descriptions.

## Checks before pushing

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Integration tests need `VECSHIFT_TEST_PG_DSN`; see CONTRIBUTING.md.

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

`src/vecshift/core` must not import from `vecshift.cli` or any other presentation layer.
