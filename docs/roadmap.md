# Roadmap

VecShift is built in the order that delivers value with the least risk: understand the
index first, prove the new model is better second, migrate third.

Dates aren't promised. Each phase ships when it works.

## Phase 0: Foundations ✅ in progress

- [x] Repository, Apache 2.0 license, CI, contributor guide, code of conduct
- [x] Canonical `Record` with tombstones and `updated_at` ordering
- [x] `EmbeddingFingerprint` and model tags
- [x] Capability flags and plugin contracts
- [x] `vecshift fingerprint`
- [ ] Plugin loading through entry points
- [x] A free first run: the built-in hashing model and an offline Docker demo

## Phase 1: Know your index ✅ in progress

Read-only tools. No writes to your stores, no production risk.

- [x] **`vecshift doctor`** for pgvector and Supabase: inspect an index and report
  - [x] mixed models or vector sizes in one index
  - [x] zero and unnormalized vectors
  - [x] exact duplicate texts and vectors
  - [x] missing source text (the index can't be re-embedded as-is)
  - [x] missing ANN indexes, and vectors too large to index
  - [x] which change-capture strategy the source supports
  - [x] row-level security hiding rows
  - [x] HTML report (`--html`)
  - [ ] near-duplicate chunks
  - [ ] dimension waste (where Matryoshka truncation could save storage)
- [ ] `vecshift doctor` for Qdrant
- [x] **`vecshift bench`**: compare embedding models on a sample of your own data, reporting
  recall@k, latency, cost per million documents, and storage, with an HTML leaderboard
  - [x] documents from JSONL or pgvector
  - [x] proxy, LLM-generated, and labeled queries
  - [x] embedding cache and cost confirmation
  - [ ] the index's existing vectors as a baseline, without re-embedding the documents
- Connectors (read only): **pgvector** ✅, **Qdrant**
- Providers: **OpenAI-compatible** (covers OpenAI, vLLM, Ollama, TEI, and most hosted
  inference) ✅ and a **hashing baseline** for free runs ✅

## Phase 2: Migrate safely

- [x] `vecshift init` and the `vecshift.yaml` job file, with a guided setup at a terminal
  that finds the vector columns, offers models, and can save secrets to a private `.env`
- [x] `vecshift status`: where a migration stands and the command to run next
- [x] `vecshift plan`: validate the job against the database and estimate tokens, cost, time,
  and storage, with an optional `--probe` of the real model
- [x] `vecshift apply` for pgvector and Supabase: re-embed into a side-by-side column
  - [x] resume from the database itself, with no row embedded twice
  - [x] a trigger and guarded writes that keep up with edits made during the run
  - [x] rejected rows isolated and retried on the next run
  - [x] a spend cap across runs that stops before the budget is passed
  - [x] concurrent index build, with lock timeouts that never block the application
  - [x] `--until PERCENT` to stop part way, check, and continue
  - [ ] adaptive rate limiting that backs off on rising latency, not only on 429s
  - [ ] partitioned tables, indexing each partition concurrently and attaching it
- [x] `vecshift cutover`, `rollback`, and `cleanup` for pgvector and Supabase: the new
  vectors take the column name the application already uses, in one transaction
- [x] `vecshift eval`: compare old and new vectors on your data, with a go/no-go verdict
  - [x] recall@k per language pair (Arabic and Latin scripts), with cross-language queries
  - [x] latency vs accuracy: query embedding and search percentiles across index settings
  - [x] works without the old model, from stored vectors and document grouping
  - [x] works on a partial (`--until`) migration
  - [x] an HTML report with the latency vs accuracy chart
  - [ ] an LLM judge for results the two sides disagree on
- [x] PyPI package, Docker image, and an offline docker compose demo
- [x] A terminal design for every command: arrow-key prompts, progress bars, tables, and
  plain output for scripts and CI

### Two kinds of migration

Every connector will support both, behind the same commands:

- **Within a store**: a new model, side by side in the same database, then an atomic
  switch. pgvector renames columns; OpenSearch, Elasticsearch, Qdrant, and Milvus build a
  new index or collection and move an alias, since they can't change a vector field in
  place.
- **To another store**: for example pgvector to Qdrant. Pick the source table or index and
  vecshift creates the target's schema (fields, metadata, vector size, and metric), then
  copies and embeds in one pass. Re-embedding is optional, so it can also move existing
  vectors as they are.

Stores without aliases (such as Pinecone) need a one-line application config change at
cutover; `doctor` will say so up front, along with whether an alias exists for rollback.

## Phase 3: Stay in sync, stay honest

- Change capture: Postgres logical replication, `updated_at` polling, delete reconciliation
- A retrieval regression gate for CI: fail a pull request when a model or chunking change
  drops recall
- Prometheus metrics
- More connectors: OpenSearch, Elasticsearch, Milvus, Weaviate, and others, driven by demand
- More providers: Gemini, Azure OpenAI, Cohere, Voyage, Bedrock, sentence-transformers

## Phase 4: API and web UI

Starts once the CLI covers the full workflow (`doctor`, `bench`, `plan`, `apply`, `eval`,
`cutover`).

- REST API over the same engine the CLI uses, so both always behave the same
- Web UI for running and watching jobs: progress, throughput, errors, and cost
- Visual views of `doctor` and `bench` reports
- Guided cutover and rollback

To keep this cheap later, the CLI stays a thin layer over the engine, and the `--json`
output is treated as a stable contract that the API can reuse.

## Not planned

These were considered and set aside. They may be revisited if users ask for them.

- **A query routing proxy.** Putting a new service in the hot path of production search is a
  large ask for latency and reliability. Alias swaps and the application's own feature flags
  cover gradual cutover.
- **Adapter mode** (mapping old vectors into the new space with a learned transform).
  Published results retain roughly 86–92% of retrieval quality, which most teams won't accept
  permanently.

## Ideas

Things that would be good but aren't scheduled:

- A static migration cost calculator page
- A self-contained HTML report for migrations, like the one `doctor` has
- "Storage diet": truncate or quantize vectors and report the recall trade-off
- An MCP server so coding agents can run `doctor`, `bench`, and `plan`
