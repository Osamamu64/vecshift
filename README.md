# VecShift

**Safe, observable embedding migrations for any vector store, with any embedding model.**

[![CI](https://github.com/Osamamu64/vecshift/actions/workflows/ci.yml/badge.svg)](https://github.com/Osamamu64/vecshift/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)

> [!WARNING]
> VecShift is in early development. `vecshift doctor` works today for pgvector and Supabase;
> migrations aren't built yet. Watch the repo or read the [roadmap](docs/roadmap.md) to
> follow along.

---

## Why

Every team running RAG or semantic search eventually wants a better or cheaper embedding
model, different dimensions, new chunking, or a different vector store. Today each of those
is a one-off project: a re-embed script, rate-limit workarounds, a new index, a manual
cutover, and no real answer to two questions:

1. **Is the new model actually better on *our* data?**
2. **What's actually in our index right now?** Which model made these vectors? Is it mixed?
   Do we still have the text to re-embed it?

VecShift answers those first, then makes the migration itself safe and repeatable.

## What it will do

| Command | Purpose | Status |
|---|---|---|
| `vecshift fingerprint` | Print a stable tag identifying an embedding configuration's vector space | ✅ Available |
| `vecshift doctor` | Inspect an index: mixed models or sizes, zero and unnormalized vectors, duplicates, missing text, indexing and change-tracking gaps | ✅ pgvector and Supabase |
| `vecshift bench` | Compare embedding models on a sample of *your* data: recall, latency, cost, storage | 🚧 Next |
| `vecshift plan` | Dry run: validate schemas, estimate tokens, cost, time, and storage | 📋 Planned |
| `vecshift apply` | Re-embed into a shadow index with checkpoints, resume, and rate limiting | 📋 Planned |
| `vecshift eval` | Compare old and new indexes and produce a go/no-go report | 📋 Planned |
| `vecshift cutover` / `rollback` | Alias swap with instant rollback | 📋 Planned |

Initial targets are **pgvector** and **Qdrant**, with an **OpenAI-compatible** embedding
provider (which covers OpenAI, vLLM, Ollama, TEI, and most hosted inference) plus a free
mock provider for trying things out.

## Design principles

- **Look before you leap.** Read-only diagnosis and benchmarking come before any write.
- **Never lose data.** Idempotent upserts guarded by `updated_at`, checkpoints, a dead-letter queue.
- **Know your vector space.** Every vector carries a fingerprint of the model configuration
  that produced it, so mixed or stale vectors are detectable.
- **Data stays home.** Runs inside your network. Nothing leaves except to the embedding
  provider you choose.
- **Free first run.** Mock embeddings and a local store, so you can try it without an API key.

## Quick start (from source)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Osamamu64/vecshift.git
cd vecshift
uv sync
uv run vecshift --help
```

### Check an existing pgvector or Supabase index

`doctor` is read-only and never downloads your vectors, so it's safe against production.

```bash
export VECSHIFT_DSN='postgresql://postgres.<project-ref>:<password>@<region>.pooler.supabase.com:5432/postgres'
uv run vecshift doctor --table public.documents
```

Abridged output:

```text
vecshift doctor · pgvector via Supabase session pooler
Target  public.documents.embedding  vector
Rows    ~1,001 (inspected 1,001, full table)

✖ ERROR    Vectors of different sizes in one index
           Sampled vectors have 2 different sizes: 1536 (901), 3072 (100). They
           come from different models and can't be compared with each other.
✖ ERROR    Vectors from more than one model
           `metadata->>'model'` records 2 different models: text-embedding-ada-002
           (900), text-embedding-3-small (100). ...
⚠ WARNING  No way to track writes during a migration
           There's no logical replication and no updated-at timestamp column, ...
✔ OK       Source text available
           Every sampled row has text in `content`, so the index can be re-embedded.
```

Add `--json` for scripts, or `--fail-on error` to fail a CI job. See the
[pgvector and Supabase guide](docs/connectors/pgvector.md) for connection strings, row-level
security, and a read-only role recipe.

### Fingerprint an embedding configuration

Get the fingerprint of an embedding configuration:

```console
$ uv run vecshift fingerprint --provider openai --model text-embedding-3-small --dimensions 1536
openai/text-embedding-3-small@1536#a3ac94a82eca
```

Any change to the provider, model, dimensions, version, task type, prefix, or normalization
produces a different tag, because it's a different vector space.

## Documentation

- [pgvector and Supabase](docs/connectors/pgvector.md): connecting, what `doctor` checks, and safety
- [Architecture](docs/architecture.md): the canonical record, plugin contracts, and capability flags
- [Roadmap](docs/roadmap.md): what's being built, in what order, and why
- [Prior art](docs/prior-art.md): related tools and how VecShift relates to them

## Contributing

Contributions are welcome, especially new connectors and embedding providers once the
contracts settle. Start with [CONTRIBUTING.md](CONTRIBUTING.md). Everyone taking part is
expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

To report a security issue, see [SECURITY.md](SECURITY.md). Please don't open a public issue.

## License

Licensed under the [Apache License 2.0](LICENSE).
