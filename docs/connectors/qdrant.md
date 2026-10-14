# Qdrant

VecShift works with [Qdrant](https://qdrant.tech) over the REST API: `doctor` and `bench`
read collections, and `plan`, `apply`, and `status` [migrate into Qdrant](#migrating-into-qdrant),
from another Qdrant collection or from a pgvector table. `cutover`, `rollback`, `cleanup`,
and `eval` for Qdrant are next on the [roadmap](../roadmap.md).

## Connecting

Pass the REST URL with `--qdrant`, or set `VECSHIFT_QDRANT_URL`. An API key, if your
Qdrant needs one, is read from `QDRANT_API_KEY` only, never from the command line:

```bash
export VECSHIFT_QDRANT_URL=https://xyz.eu-central-1-0.aws.cloud.qdrant.io:6333
export QDRANT_API_KEY=...          # or put both in .env
vecshift doctor --collection documents
```

Both can live in a `.env` file next to where you run vecshift, like the PostgreSQL settings.

The key is sent only over `https://`, or over plain `http://` to this machine (for a local
Qdrant). Redirects are never followed, credentials in the URL are rejected, and no message
ever includes the key.

## Running `doctor`

```bash
vecshift doctor --qdrant http://localhost:6333 --collection documents
vecshift doctor --qdrant http://localhost:6333 --collection documents --vector dense
```

`--collection` takes a collection or an alias. Without it, `doctor` inspects the only
dense vector in the instance, or lists them all so you can choose. Collections with several
named vectors need `--vector`.

### What it detects

The same checks as for pgvector (vector sizes, norms, duplicates, models, source text,
and the index), plus one that matters for migrations in Qdrant:

- **Aliases.** A migration builds a new collection and switches searches by moving an
  alias, so it needs your application to query through one. Qdrant can't give an alias the
  name of an existing collection, so if your application uses the collection name
  directly, `doctor` warns that the first migration needs a one-time change: create an
  alias (for example `documents_live`) and point the application at it. After that, every
  migration is just a model switch.

These payload fields are recognized automatically, at the top level or, for the model,
inside `metadata`:

| Purpose | Recognized payload fields |
|---|---|
| Source text | `page_content` (LangChain), `content`, `text`, `document`, `chunk`, `chunk_text`, `body` |
| Model | `model_tag`, `embedding_model`, `model`, `model_name`, also under `metadata` or `meta` |
| Change tracking | `updated_at`, `modified_at`, `last_modified`, `last_modified_at`, `changed_at` |

For text anywhere else, pass its path with `--text-column`, such as `metadata.text`.

## Safety

- `doctor` and `bench` only read. They never create, change, or delete anything.
- `apply` writes only to the new collection. The source collection or table is only read,
  and `status` only reads both.
- Qdrant can't compute norms or hashes itself, so `doctor` reads a sample of points with
  their vectors (a random sample on Qdrant 1.11 and later, otherwise the first points),
  turns them into counts, dimensions, norms, and hashes in memory, and discards them.
  Nothing about a single point is stored or shown, and the HTML report holds statistics
  only.
- Only dense vectors are inspected; sparse and multi-vectors are skipped.

## `bench` from Qdrant

```bash
vecshift bench --qdrant http://localhost:6333 --collection documents \
  -m openai/text-embedding-3-small -m ollama/bge-m3
```

It samples documents (ID and text) from the collection's payloads, and labeled queries
can refer to point IDs, integers or UUIDs.

## Migrating into Qdrant

Qdrant can't change a collection's vectors in place, so a migration fills a **new
collection** next to the current one, with the same payloads and vectors from the new model,
and later points your application's **alias** at it. The source is only ever read.

### From a Qdrant collection

```yaml
# vecshift.yaml
name: docs-reembed
source:
  type: qdrant
  collection: documents          # the alias your application searches
  # vector: dense                # when the collection has several named vectors
  # text_field: metadata.body    # found automatically when it's a common name
target:
  collection: documents_2026_10  # defaults to documents_v2
  # metric: cosine               # defaults to the current collection's distance
model: openai/text-embedding-3-large,dims=1024
limits:
  budget_usd: 50
```

The new collection gets the same vector name as the old one, so queries keep working, and
the source's payload indexes, so filters stay fast. Both sides use `VECSHIFT_QDRANT_URL`
and `QDRANT_API_KEY` unless the source sets `url_env` and `key_env`, or the target sets its
own to copy between two Qdrant servers.

If `source.collection` is a collection rather than an alias, `plan` warns that there's no
alias to switch: set `target.alias` and have the application search that alias instead.

### From a pgvector table

```yaml
source:
  table: public.documents        # read with VECSHIFT_DSN, as for any pgvector job
target:
  type: qdrant
  collection: documents          # defaults to <table>_v2
  alias: documents_live          # what cutover will point at the new collection
model: openai/text-embedding-3-small
```

Every row with text becomes a point. Its payload holds every column except vector
columns, as JSON. Its ID is the primary key when that's a non-negative integer or a UUID,
and otherwise a UUID derived from the key, so the same row always maps to the same point.
The table needs a single-column primary key. vecshift reads it in short read-only
transactions on a read-only connection, so a transaction pooler works and the table is
never locked or changed.

### Running it

```bash
vecshift plan      # checks both sides and estimates cost, without changing anything
vecshift apply     # creates the collection and fills it
vecshift status    # how far along it is, and what to run next
```

Run `apply` again at any time to catch up. The new collection itself is the record of
progress: a point is (re-)embedded when it's missing there or its text differs from the
source, a payload-only edit is copied without calling the model, and points deleted from
the source (or whose text was removed) are deleted from the new collection at the end of
a complete run. After filling, `apply` waits for Qdrant to finish indexing.

### What `plan` checks for Qdrant

| Finding | Severity | Meaning |
|---|---|---|
| `plan.no_alias` | warning | Nothing to switch at cutover: set `target.alias` |
| `plan.alias_conflict` | error | The alias name is already a collection's |
| `plan.target_is_source` | error | `target.collection` is the collection being replaced |
| `plan.collection_exists` | info | The new collection exists with matching vectors; apply fills the gaps |
| `plan.collection_conflict` | error | It exists with a different size or distance |
| `plan.cut_over` | error | The alias already points at the new collection |
| `plan.other_vectors` | error | The source has more vectors than the one being replaced, which a copy would drop |
| `plan.no_primary_key` | error | A pgvector source without a single-column primary key |
| `plan.partial_text` | warning | Points without text, which aren't copied |

### Limits for now

- Each `apply` compares the whole source with the new collection, page by page. That's
  cheap next to embedding, but it reads every point once per pass.
- Only one `apply` per job should run at a time. Qdrant has no locks, so vecshift guards
  with a lock file in `.vecshift/`, which only covers runs from the same directory.
- Collections with several named vectors, or with sparse vectors, aren't supported yet.
