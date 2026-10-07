# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
