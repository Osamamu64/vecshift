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
- [ ] Mock embedding provider and local store for a free first run

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

- [x] `vecshift init` and the `vecshift.yaml` job file
- [x] `vecshift plan`: validate the job against the database and estimate tokens, cost, time,
  and storage, with an optional `--probe` of the real model
- `vecshift apply`: re-embed into a shadow index or named vector
  - checkpoints and resume
  - idempotent upserts guarded by `updated_at`
  - dead-letter queue with `vecshift dlq retry`
  - adaptive rate limiting that backs off on 429s and rising latency
  - a hard spend cap that stops the run before it exceeds budget
- `vecshift eval`: compare old and new indexes on overlap and recall@k, with a go/no-go report
- `vecshift cutover` and `vecshift rollback` through alias swap
- Connectors gain write support; `copy` mode for moving vectors without re-embedding
- Docker image and a docker-compose demo

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
