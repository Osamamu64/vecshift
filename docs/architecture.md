# Architecture

This document describes how VecShift is put together. Parts marked *(planned)* don't exist
in code yet.

## Overview

```
┌──────────────────────────────────────────────────────────┐
│ Interfaces          CLI  ·  REST API (planned)           │
└──────────────────────────────────────────────────────────┘
                            │  same YAML job spec
┌──────────────────────────────────────────────────────────┐
│ Control plane       Planner · State store · Checkpoints  │  (planned)
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│ Data plane          Source ──► Embed ──► Target          │  (planned)
│                     bounded queues, backpressure, DLQ    │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│ Core                Record · Fingerprint · Capabilities  │
│                     · Contracts                          │
└──────────────────────────────────────────────────────────┘
```

`vecshift.core` sits at the bottom and never imports from the layers above it.

## Canonical record

Every connector converts to and from one type,
[`Record`](../src/vecshift/core/record.py). Supporting N stores then takes N connectors,
not N × N pairwise bridges.

| Field | Purpose |
|---|---|
| `id` | Stable identifier from the source. |
| `text` | Original text. Needed to re-embed; `None` for vector-only stores. |
| `vectors` | Dense vectors by name (`"default"` for single-vector stores). |
| `sparse` | Sparse vectors by name, for hybrid search. |
| `metadata` | Payload, carried across unchanged unless remapped. |
| `updated_at` | Last-modified time. Guards upserts so a backfill never overwrites a newer live write. |
| `model_tag` | Fingerprint of the vector space the vectors belong to. |
| `deleted` | Tombstone for deletes seen through a change feed. |

### Write ordering

`Record.is_newer_than` decides upsert races. A record without `updated_at` never beats one
that has it. That keeps a timestamp-less backfill from clobbering a timestamped live write.

## Fingerprints

An [`EmbeddingFingerprint`](../src/vecshift/core/fingerprint.py) captures everything that
determines a vector space:

- provider and model
- dimensions
- version
- task type (for providers with document and query modes)
- text prefix (for example `passage: ` for E5-style models)
- normalization

Its `model_tag` looks like `openai/text-embedding-3-small@1536#a3ac94a82eca`: a readable
prefix plus a hash over every field. Two vectors are comparable only when their tags match.

Fingerprints are the foundation of `doctor` (finding mixed or stale vectors in one index)
and of migration safety (never writing vectors from two spaces into one field).

## Capabilities

Connectors declare [`Capability`](../src/vecshift/core/capabilities.py) flags, and the
planner chooses a strategy from them. It never assumes a capability that wasn't declared.

| Capability | Enables |
|---|---|
| `alias` | Atomic cutover and rollback by repointing an alias. |
| `named_vectors` | Old and new vectors side by side in one collection. |
| `change_feed` | Live sync during and after backfill. |
| `scroll` | Resumable full scans. |
| `sparse` | Hybrid search migration. |
| `delete_detection` | Direct deletes instead of reconcile-by-diff. |
| `stores_text` | Re-embedding without a separate text source. |

## Plugin contracts

Defined as `typing.Protocol`s in [`contracts.py`](../src/vecshift/core/contracts.py):

| Contract | Responsibility |
|---|---|
| `SourceConnector` | Count records and read them in resumable batches. |
| `TargetConnector` | Idempotent batched upserts that honour tombstones and `updated_at`. |
| `EmbeddingProvider` | Embed text in document or query mode; expose its fingerprint. |

Third-party plugins will register through the `vecshift.plugins` entry-point group, so
connectors can ship without changes to core.

## Change capture

Keeping a shadow index in sync with live writes depends on what the source offers:

1. **A real change feed** (for example Postgres logical replication): stream changes directly.
2. **An `updated_at` field**: poll for rows changed since the last watermark. This can't see
   deletes, so it needs periodic reconciliation by ID.
3. **Neither**: the application has to dual-write during the migration.

`doctor` will report which of these applies before a migration starts, rather than leaving
it to be discovered mid-run.

## Language

Python, for the embedding and database client ecosystem and because most users and
contributors are AI engineers. Components talk over clear interfaces, so a
latency-sensitive piece can be rewritten in another language later if benchmarks justify it.
