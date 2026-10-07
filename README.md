# VecShift

**Safe, observable embedding migrations for any vector store, with any embedding model.**

[![CI](https://github.com/Osamamu64/vecshift/actions/workflows/ci.yml/badge.svg)](https://github.com/Osamamu64/vecshift/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)

> [!WARNING]
> VecShift is in early development. The core types and CLI are taking shape, but it cannot
> migrate an index yet. Watch the repo or read the [roadmap](docs/roadmap.md) to follow along.

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
| `vecshift doctor` | Inspect an index: mixed models, unnormalized vectors, duplicates, missing text, wasted dimensions | 🚧 Next |
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

Get the fingerprint of an embedding configuration:

```console
$ uv run vecshift fingerprint --provider openai --model text-embedding-3-small --dimensions 1536
openai/text-embedding-3-small@1536#a3ac94a82eca
```

Any change to the provider, model, dimensions, version, task type, prefix, or normalization
produces a different tag, because it's a different vector space.

## Documentation

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
