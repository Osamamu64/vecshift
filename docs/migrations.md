# Planning a migration

A migration re-embeds a pgvector column with a new model, side by side with the old one,
then switches searches over in one step. This page covers the job file and
`vecshift plan`. `apply` and `cutover` come next.

```bash
vecshift init --table public.documents --model openai/text-embedding-3-large,dims=1024
export VECSHIFT_DSN='postgresql://...'
vecshift plan                 # reads vecshift.yaml; changes nothing
```

## How the migration works

1. **Add a column.** The new vectors go in a new column, `embedding_v2` by default, on the
   same table. A nullable column with no default is added instantly at any table size.
2. **Backfill.** Each row's text is embedded with the new model and written to the new
   column, in resumable batches, while the table keeps serving reads and writes.
3. **Index.** The vector index is built on the new column after the backfill, concurrently.
4. **Cut over.** The old and new columns are swapped by renaming them in one transaction,
   so searches switch atomically. Rollback renames them back.

Your application keeps querying the same column name throughout.

## The job file

`vecshift init` writes a commented `vecshift.yaml`. The full set of settings:

```yaml
version: 1
name: documents-reembed

source:
  type: pgvector
  dsn_env: VECSHIFT_DSN        # environment variable holding the connection string
  table: public.documents
  vector_column: embedding     # only needed if the table has several vector columns
  text_column: content         # detected automatically for common names

target:
  column: embedding_v2
  vector_type: vector          # or halfvec
  index: hnsw                  # hnsw, ivfflat, or none
  metric: cosine               # defaults to the current index's metric, or cosine

model: openai/text-embedding-3-large,dims=1024   # a model spec, as for vecshift bench

limits:
  budget_usd: 50               # plan fails above this; apply will stop at it
  tokens_per_minute: 1000000   # your provider's rate limits, for the duration estimate
  requests_per_minute: 3000
```

The file never contains credentials: the connection string comes from the environment
variable that `dsn_env` names. Sections whose settings are all commented out use the
defaults, and unknown settings are rejected so typos don't go unnoticed.

## What `plan` checks

`plan` connects read-only, samples the table, and reports what `apply` would do, what it
would cost, and anything that would make it fail. It exits with status 1 when there are
errors, so it can gate a CI job.

| Check | Severity |
|---|---|
| No text column to re-embed | error |
| No primary key (apply needs a stable ID to write back and resume) | error |
| The connecting role doesn't own the table, which adding a column requires | error |
| The target column exists with a different type or size | error |
| More dimensions than pgvector can index for the vector type (`halfvec` is suggested when it fits) | error |
| `dims=` on a model that can't shorten its vectors, or larger than its output | error |
| No API key for the model | error |
| Estimated cost above `limits.budget_usd` | error |
| Some rows have no text, so they'd have no new vector | warning |
| Vector size unknown (add `dims=` or use `--probe`) | warning |
| The table already uses the target model | warning |
| The HNSW build needs more memory than `maintenance_work_mem` | warning |
| No way to track writes made during the migration (from `doctor`) | warning |
| No budget set, or no price for the model | info |

## Estimates

- **Rows** come from the table's statistics and the share of sampled rows with text.
- **Tokens** use the sampled text length at about four characters per token, or the
  ratio measured by `--probe`.
- **Cost** is tokens times the model's price per million tokens.
- **Duration** is the slowest of your rate limits and the speed measured by `--probe`.
- **Storage** counts the new column (4 bytes per dimension for `vector`, 2 for `halfvec`),
  next to the old one.
- **Index build memory** is a rough HNSW estimate, compared with `maintenance_work_mem`.

## Probing the model

`vecshift plan --probe` embeds 16 sample rows with the real model to measure its output
size, tokens per character, and speed. That replaces guesses with measurements for models
vecshift doesn't know. For a remote API it asks before sending anything; `--yes` skips the
question.
