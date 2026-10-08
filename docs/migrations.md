# Planning a migration

A migration re-embeds a pgvector column with a new model, side by side with the old one,
then switches searches over in one step. This page covers the job file,
`vecshift plan`, and `vecshift apply`. `cutover` and `rollback` come next.

```bash
vecshift init --table public.documents --model openai/text-embedding-3-large,dims=1024
export VECSHIFT_DSN='postgresql://...'
vecshift plan                 # reads vecshift.yaml; changes nothing
vecshift apply                # adds the column, embeds every row, builds the index
```

## How the migration works

1. **Add a column.** The new vectors go in a new column, `embedding_v2` by default, on the
   same table. A nullable column with no default is added instantly at any table size.
   A small trigger clears a row's new vector whenever its text changes, so edits made
   during the migration are never left with a vector of the old text.
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

## Running `apply`

```bash
vecshift apply                # asks first; --yes skips the question
vecshift apply --json         # progress as JSON lines, for scripts and dashboards
vecshift apply --no-index     # backfill only; build the index on a later run
```

`apply` re-runs the plan first and refuses to start if it has errors. It then shows the
changes, the estimated cost, and where row text will be sent, and asks before doing
anything. Without a terminal it needs `--yes`.

What it does, in order:

1. Adds the target column, then the sync trigger, each in its own short transaction.
2. Embeds every row that has text but no new vector, in primary-key order, and writes the
   vectors back in batches. Each write only lands if the row's text is still the text that
   was embedded, so a row edited mid-batch is picked up again instead of getting a stale
   vector.
3. Makes catch-up passes for rows that were edited or inserted during the run.
4. Builds the vector index with `CREATE INDEX CONCURRENTLY`, so writes continue. An
   invalid index left by an interrupted build is dropped and rebuilt.
5. Checks that every new vector has the expected size.

### Stopping, resuming, and catching up

Every finished batch is committed, so `apply` can stop at any point and pick up where it
left off. The database itself records which rows are done (their new vector is filled in),
so nothing is embedded twice.

- **Ctrl-C** once finishes the current batch and stops; a second Ctrl-C aborts at once.
- **Run it again** at any time. A re-run embeds only rows that are new or whose text
  changed since, which makes it the way to catch up just before cutover.

Until you cut over, your application still writes vectors from the old model to the old
column. New and edited rows get their new vector the next time `apply` runs, so switch
your application's ingestion to the new model before cutover and run `apply` once more
right before it.

### Budget

With `limits.budget_usd` set, `apply` estimates each batch's cost before sending it and
stops before the total across runs would pass the budget. Spend so far is kept in the
state file, so a budget covers the whole migration, not each run. Raise the budget and
run `apply` again to continue. A budget needs the model's price (`price=` in the spec for
models vecshift doesn't know).

### Rows the provider rejects

If the provider rejects a batch, `apply` splits it to find the rows at fault and writes
the rest. Rejected rows are recorded by ID (never their text) in the state file and
retried on the next run after the others.

### Exit status

| Status | Meaning |
|---|---|
| 0 | Every row with text has a new vector, and the index is built |
| 1 | Failed: the plan has errors, the model returned the wrong size, or the database reported an error |
| 2 | Not started: bad arguments, no confirmation, or the job can't run as configured |
| 3 | Stopped on purpose (Ctrl-C, the budget, or rows still changing); run `apply` again |

### Locks and concurrency

- Schema changes wait at most 5 seconds for their lock, then back off and retry, instead
  of queueing behind your application's queries and blocking them.
- Statements run under timeouts. Only the concurrent index build runs without one.
- A session advisory lock keeps two `apply` runs off the same column at once.
- Because of that lock and the concurrent index build, `apply` needs a real session: on
  Supabase use the direct connection or the session pooler (port 5432), not the
  transaction pooler (port 6543).

### The state file

`apply` keeps a small state file at `.vecshift/<job name>.state.json` next to the job
file: money and tokens spent across runs, rows written, and the IDs of rejected rows. It
never holds row text or credentials, and only its owner can read it. Deleting it resets
the spend count and retries rejected rows; the rows already done stay done.

### Permissions

`apply` adds a column, a trigger, and an index, which PostgreSQL only allows the table's
owner to do. Run it as the owner (on Supabase, `postgres`), or as a role that is a member
of the owning role:

```sql
create role vecshift_migrator with login password 'use-a-strong-password';
grant usage on schema extensions to vecshift_migrator;
grant postgres to vecshift_migrator;   -- or whichever role owns the table
```

Table owners bypass row-level security unless the table uses `FORCE ROW LEVEL SECURITY`,
so `apply` sees every row. Revoke the membership once the migration is done.
