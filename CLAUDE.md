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

## Architecture rule

`src/vecshift/core` must not import from `vecshift.cli` or any other presentation layer.
