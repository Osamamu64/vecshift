# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- `vecshift doctor` sampled only the start of large tables, so rows written later (often by a
  newer model) could be missed. Samples are now spread across the table.

### Added

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
