# Qdrant

VecShift reads [Qdrant](https://qdrant.tech) collections over the REST API, for `doctor`
and `bench`. Migrations within Qdrant (a new collection, then an alias switch) are next on
the [roadmap](../roadmap.md).

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
