# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `doctor` no longer warns "No way to track writes during a migration" for pgvector
  tables with a primary key: `vecshift apply` tracks edits with its own trigger. It
  reports `sync.trigger` (OK) instead. Other stores, and tables without a primary key,
  still get `sync.none`.
- `apply` leaves a blank line between its summary and its progress at a colour terminal.

## [0.3.0] - 2026-10-10

### Changed

- A new look at the terminal, across every command: a gradient block wordmark, numbered
  steps and arrow-key menus in `vecshift init` (with hidden input for secrets), a panel
  with the migration's progress in `vecshift status`, a live progress bar in `apply`,
  highlighted SQL in `plan`, tables in `eval` and `bench`, and consistent headers,
  findings, verdicts, and next steps everywhere. Off a terminal, output stays plain and
  prompts fall back to typed answers, as before.
- New dependency: `questionary`, for the arrow-key menus.

## [0.2.0] - 2026-10-10

### Added

- `vecshift init` guides you at a terminal: it asks for the connection string (hidden),
  lists the vector columns it finds, picks the text column, and offers a short list of
  models and a spending limit, then shows the plan. Flags still work without prompts.
- Commands read settings such as `VECSHIFT_DSN` and `OPENAI_API_KEY` from a `.env` file in
  the current folder. Variables already set take precedence; `--env-file` and
  `--no-env-file` choose another file or none. `init` can save to it, privately.
- `vecshift status` shows where a migration stands and what to run next (`--json` too).
- Running `vecshift` alone, and `vecshift init`, show the logo at a colour terminal.

## [0.1.0] - 2026-10-10

The first release.

### Security

- API keys are never sent over unencrypted `http://` to remote hosts, redirects are never
  followed, and credentials inside model URLs are rejected. URL query strings are hidden
  wherever a spec is shown.
- `compat` servers only receive an API key when the spec names one with `key_env=`.
  Previously `VECSHIFT_API_KEY` was sent to any compat URL.
- The embedding cache is now readable by its owner only (directory `0700`, file `0600`).
- HTML reports carry a strict Content Security Policy: no network access, and only their
  own script, pinned by hash, may run.
- SQL generated for `apply` quotes identifiers using the server's keyword list.
- Crash tracebacks never print local variables, and `--dsn` with a password on the
  command line prints a warning.
- CI runs `pip-audit`, `zizmor`, and Ruff's security rules on every change and weekly.
  Actions are pinned to commit SHAs, and checkout no longer persists credentials.

### Fixed

- On tables with constant writes, `apply` never finished (a few rows always changed during
  each catch-up pass, so it stopped without building the index) and `cutover` could never
  pass its final check. `apply` now finishes once at most 0.1% of rows (at least 100) are
  left, counting them after the index build; each pass covers only what was pending when it
  began; and `cutover` embeds up to 500 late rows while it holds the write lock.
- `eval` treated a busy table's few waiting rows as a partial migration and skipped latency.
- `eval`'s index recall now counts ties: a row exactly as close as the exact 10th result is
  a correct answer. Tables with near-duplicate rows read far too low before.
- Schema changes wait at most 2 seconds (was 5) for a lock, which bounds how long
  application writes can queue behind them.
- `apply` explains `could not resize shared memory segment` during the index build
  (Docker's default 64 MB `/dev/shm`), and `plan`'s memory hint mentions `--shm-size`.
- `plan` described cutover with the wrong column name (`_previous`; it's `_old`).

- `plan` (and so `apply`) and `cutover --check` refuse a table whose row-level security binds
  the connecting role, such as one with `FORCE ROW LEVEL SECURITY`. `apply` used to report
  success while skipping the rows the role couldn't see.
- They also refuse partitioned tables, where `apply` embedded every row and then failed to
  build the index, since PostgreSQL can't build one concurrently on a partitioned table.

- `vecshift doctor` sampled only the start of large tables, so rows written later (often by a
  newer model) could be missed. Samples are now spread across the table.

### Added

- A Docker image that runs as a non-root user, and an offline demo:
  `cd demo && docker compose up` runs doctor, plan, apply, eval, and cutover on a sample
  table of English and Arabic articles, with no API key. See `docs/demo.md`.
- Job files accept the built-in `hash/N` models, for trying vecshift out; `plan` warns
  that they're test models (`plan.baseline_model`).
- `vecshift eval` compares the old and new vectors on your data before cutover, without
  re-embedding documents. It reports recall@1, recall@10, and MRR@10 overall and for each
  query → document script pair (Arabic and Latin), how much the top results changed,
  query embedding latency, and search latency (p50, p95, p99) with index recall across
  `hnsw.ef_search` or `ivfflat.probes` settings. It ends with GO, NO-GO, or INCONCLUSIVE
  (exit status 0, 1, or 3), with optional latency limits. If the old model is unavailable
  it judges the old side from stored vectors, by how well each row's nearest rows stay in
  the same document. Queries come from the rows themselves, a labeled file, or an LLM,
  optionally in the other language (`--cross-language`). It's read-only, and `--dsn-env`
  can point it at a replica.
- `vecshift eval --html FILE` writes a self-contained report: the verdict, headline numbers
  with their change, recall@10 per language as a before → after chart, and a latency vs
  accuracy chart for both sides, each with a table view, in light and dark themes.
- `source.model` in the job file names the model that made the current vectors.

- `vecshift cutover` switches searches to the new vectors: in one transaction it renames
  the live column to `<name>_old` and the new column to `<name>`, so the application's SQL
  doesn't change. It first embeds rows added or edited since the last `apply`, then
  confirms under a brief write lock that every row has a new vector. `--check` reports
  whether the switch is safe without changing anything: missing vectors, an unbuilt index,
  views bound to the column, and a vector size change the application must match.
- `vecshift rollback` renames the columns back, and reports rows added or edited since
  cutover that have no old-model vector.
- `vecshift cleanup` drops the old column, its index, and its trigger once you're sure.
- `vecshift apply --until PERCENT` stops once that share of rows has a new vector.

### Changed

- The sync trigger keeps a new vector the application writes itself in the same update,
  and pins its function's `search_path`. Dropping the column it guards by hand is refused
  instead of leaving a broken trigger behind.
- `plan` reports a table that was already cut over and not cleaned up.

- `vecshift apply` runs a migration on pgvector and Supabase: it adds the new column and a
  trigger that clears a row's new vector when its text changes, embeds every row in
  resumable batches, makes catch-up passes for rows edited during the run, then builds the
  index concurrently and checks every vector's size. Writes are guarded so a vector only
  lands if the row's text is unchanged. It stops cleanly on Ctrl-C or before passing
  `limits.budget_usd` (counted across runs), isolates rows the provider rejects, and
  `--json` streams progress as JSON lines. Exit status 3 means "stopped; run again".
- `plan` now lists the sync trigger among the changes, and reports tables with a
  composite primary key, which `apply` doesn't support yet.

- `vecshift init` writes a commented `vecshift.yaml` job file describing a migration: the
  source table, the side-by-side target column, the new model, and limits such as a budget.
- `vecshift plan` checks a job against the database without changing anything. It shows
  the SQL apply would run, estimates rows, tokens, cost, duration, storage, and index build
  memory, and reports anything that would make the migration fail, exiting with status 1 on
  errors. `--probe` measures the real model's size, token counts, and speed on 16 rows.
- `vecshift bench` compares embedding models on a sample of your documents, from a JSONL
  file or a pgvector table. It reports recall@1, recall@10, MRR@10, query latency,
  throughput, cost per million documents, and storage, as a terminal table, `--json`, or an
  `--html` leaderboard. Queries come from the documents themselves (free), from an LLM
  (`--generate-queries`), or from your labeled file (`--queries`).
- Embedding providers: OpenAI, Ollama, any OpenAI-compatible server, and a free hashing
  baseline, with batching, retries that respect rate limits, known query and document
  prefixes, and an on-disk embedding cache. Installed with `pip install 'vecshift[bench]'`.
- `bench` shows estimated tokens and cost and asks before sending text to a remote API.
- Project logo, in light and dark versions, with a GitHub social preview image. The HTML
  report shows it in its header and as its browser-tab icon. `scripts/make_logo.py`
  regenerates the logo files.
- `vecshift doctor --html FILE` writes a self-contained HTML report: a verdict, key numbers,
  filterable findings, and charts of vector lengths, models, and vector sizes. It works
  offline, supports light and dark themes, and contains statistics only.
- The JSON report gains a `facts` section with the measurements behind the findings.

- `vecshift doctor` for pgvector, including Supabase: a read-only check of an existing index
  for missing source text, mixed vector sizes or models, zero and unnormalized vectors,
  duplicates, missing ANN indexes, row-level security gaps, and how live writes could be
  tracked during a migration. Supports `--json` and `--fail-on` for CI.
- PostgreSQL connections that work with every Supabase mode (direct, session pooler,
  transaction pooler), require TLS for Supabase hosts, and never print passwords.
- Canonical `Record` type with tombstones and `updated_at` conflict resolution.
- `EmbeddingFingerprint` and model tags that identify a vector space.
- `Capability` flags and plugin contracts for sources, targets, and embedding providers.
- `vecshift fingerprint` and `vecshift --version` commands.

[Unreleased]: https://github.com/Osamamu64/vecshift/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Osamamu64/vecshift/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Osamamu64/vecshift/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Osamamu64/vecshift/releases/tag/v0.1.0
