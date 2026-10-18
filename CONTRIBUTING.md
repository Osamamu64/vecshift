# Contributing to VecShift

Thanks for your interest in VecShift. This guide covers how to set up a development
environment and get a change merged.

## Ground rules

- Be kind. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- For anything larger than a small fix, open an issue first so we can agree on the approach.
- Security issues go through [SECURITY.md](SECURITY.md), not public issues.

## Development setup

You'll need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Osamamu64/vecshift.git
cd vecshift
uv sync
```

## Checks

CI runs these on every pull request. Run them locally before you push:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting (run without --check to fix)
uv run mypy                  # strict type checking
uv run pytest                # tests
```

### Security checks

CI also runs these. Run them before changing dependencies or workflows:

```bash
uv export --frozen --all-extras --all-groups --no-hashes --no-emit-project -o /tmp/req.txt
uvx pip-audit --strict -r /tmp/req.txt      # known vulnerabilities in dependencies
uvx zizmor --offline .github/workflows/     # GitHub Actions security
```

Ruff's security rules (`S`) run as part of `uv run ruff check .`. Read
[docs/security.md](docs/security.md) before touching credentials, SQL, HTML output, or
anything that sends data off the machine; `tests/test_security.py` pins those properties.

### Integration tests

Tests under `tests/integration` run against a real PostgreSQL with pgvector, and the Qdrant
ones also against a real Qdrant. They're skipped unless `VECSHIFT_TEST_PG_DSN` (and, for
Qdrant, `VECSHIFT_TEST_QDRANT_URL`) is set:

```bash
docker run -d --name vecshift-pg -p 55432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
docker run -d --name vecshift-qdrant -p 6333:6333 qdrant/qdrant:v1.19.2
export VECSHIFT_TEST_PG_DSN=postgresql://postgres:postgres@localhost:55432/postgres
export VECSHIFT_TEST_QDRANT_URL=http://localhost:6333
uv run pytest
```

Each test creates and drops its own database, laid out like Supabase (pgvector in an
`extensions` schema), and its own Qdrant collections. CI runs them against PostgreSQL 16
and 17, with Qdrant 1.19.

### JSON contracts

The `--json` output of every command is pinned in `tests/contracts/`: the keys and value
types, not the values ([docs/json.md](docs/json.md) says what they mean). If you add a
field, the contract tests fail until you update the contracts:

```bash
VECSHIFT_UPDATE_CONTRACTS=1 uv run pytest tests/integration/test_json_contracts.py
```

Review the diff and commit it with your change. Adding a field is fine; removing,
renaming, or retyping one is a breaking change and needs a changelog entry saying so.

## Project layout

```
src/vecshift/
  core/        # engine types and contracts; must not import from the CLI or UI
  doctor/      # store-independent diagnosis: profile, checks, findings
  embeddings/  # model specs, providers, and the embedding cache
  bench/       # model benchmarking: corpus, queries, metrics, runner, leaderboard
  jobs/        # the vecshift.yaml job spec
  planning/    # turning a job into a plan: changes, estimates, findings
  migrate/     # the apply loop and its state file, independent of any store
  eval/        # comparing old and new vectors: queries, scores, latency, the verdict
  connectors/  # one package per store: pgvector and qdrant
  service/     # what each command does, without a terminal; under the CLI and the API
  cli.py       # command-line interface
tests/         # pytest suite
docs/          # architecture, roadmap, prior art
```

The boundary matters: `vecshift.core` and `vecshift.service` never import from
presentation layers (typer, rich, `vecshift.ui`, the `cli*` modules), so the same engine can
back the CLI, an API, and a UI. `tests/test_architecture.py` enforces it. New command logic
goes in `vecshift.service`, raising `ServiceError` rather than exiting; the CLI command
only renders the result.

## Adding a connector or provider

Connectors and embedding providers implement the protocols in
[`src/vecshift/core/contracts.py`](src/vecshift/core/contracts.py) and declare their
[`Capability`](src/vecshift/core/capabilities.py) flags. Read the
[architecture doc](docs/architecture.md) first. Each new connector needs integration tests
that run against a real instance in a container.

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests for any behavior change.
- Update [CHANGELOG.md](CHANGELOG.md) under **Unreleased**.
- Write commit messages in the imperative mood ("Add pgvector source", not "Added").

Maintainers publish releases as described in [RELEASING.md](RELEASING.md).

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
