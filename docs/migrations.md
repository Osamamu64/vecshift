# Running a migration

A migration re-embeds a pgvector column with a new model, side by side with the old one,
then switches searches over in one step. This page covers the job file and the four
commands: `plan`, `apply`, `cutover`, and `rollback`. [Evaluating a migration](eval.md)
covers `eval`, which compares the old and new vectors before you switch.

```bash
vecshift init --table public.documents --model openai/text-embedding-3-large,dims=1024
export VECSHIFT_DSN='postgresql://...'
vecshift plan                 # reads vecshift.yaml; changes nothing
vecshift apply                # adds the column, embeds every row, builds the index
vecshift eval                 # is the new model better on your data? how fast? read-only
vecshift cutover --check      # is it safe to switch? changes nothing
vecshift cutover              # searches now use the new vectors, under the same name
vecshift rollback             # if needed: the old vectors are back, instantly
vecshift cleanup              # when you're sure: drop the old vectors for good
```

## How the migration works

1. **Add a column.** The new vectors go in a new column, `embedding_v2` by default, on the
   same table. A nullable column with no default is added instantly at any table size.
   A small trigger clears a row's new vector whenever its text changes, so edits made
   during the migration are never left with a vector of the old text.
2. **Backfill.** Each row's text is embedded with the new model and written to the new
   column, in resumable batches, while the table keeps serving reads and writes.
3. **Index.** The vector index is built on the new column after the backfill, concurrently.
4. **Cut over.** The live column is renamed to `embedding_old` and the new one to
   `embedding`, in one transaction that takes milliseconds, so every search switches at
   once. Rollback renames them back.

Your application keeps querying the same column name throughout. The one thing it has to
change is the model it embeds queries and new rows with, at cutover time: a query embedded
by one model can't be compared with vectors from another.

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
  model: openai/text-embedding-3-small   # the model that made the current vectors, for eval

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
| A partitioned table (not supported yet) | error |
| Row-level security applies to the connecting role, so it can't reach every row | error |
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
vecshift apply --until 50     # stop once half the rows have a new vector
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
- **`--until PERCENT`** stops once that share of rows has a new vector, so you can check
  the new vectors on part of the data before paying for the rest. Searches keep using the
  old vectors the whole time.
- **Run it again** at any time. A re-run embeds only rows that are new or whose text
  changed since. `cutover` does this one last time itself.

Until you cut over, your application keeps writing the old model's vectors to the live
column, and the sync trigger clears the new vector of any row whose text it changes. If
your application starts writing the new model's vectors to the new column itself, the
trigger keeps them.

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
| 3 | Stopped on purpose (Ctrl-C, the budget, `--until`, or rows still changing); run `apply` again |

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

`apply` adds a column, a trigger, and an index, `cutover` and `rollback` rename columns,
and `cleanup` drops one, which PostgreSQL only allows the table's owner to do. Run it as the owner (on Supabase, `postgres`), or as a role that is a member
of the owning role:

```sql
create role vecshift_migrator with login password 'use-a-strong-password';
grant usage on schema extensions to vecshift_migrator;
grant postgres to vecshift_migrator;   -- or whichever role owns the table
```

Table owners bypass row-level security, so `apply` reaches every row. If the table uses
`FORCE ROW LEVEL SECURITY`, policies bind the owner too, and `apply` could skip rows it can't
see while reporting success, so `plan` refuses. Connect as a role with `BYPASSRLS` (or a
superuser), or turn off `FORCE ROW LEVEL SECURITY` for the migration. Revoke any membership
or attribute you granted once the migration is done.

## Cutting over

```bash
vecshift cutover --check      # changes nothing; exits 1 if it isn't safe
vecshift cutover              # asks first; --yes skips the question
```

`cutover --check` reports whether the switch is safe:

| Check | Severity |
|---|---|
| `apply` hasn't run, or the table was already cut over | error |
| A partitioned table, or row-level security hiding rows from the connecting role | error |
| `<column>_old` already exists (left from an earlier migration) | error |
| Rows with text but no new vector | error for `--check`; `cutover` embeds them first |
| The vector index on the new column isn't built | error |
| The new column holds vectors of different sizes | error |
| Views or SQL-standard functions bound to a vector column | error |
| Rows with a vector but no text, which will have none after cutover | warning |
| The vector size changes, so old-model queries and inserts will fail until the app switches | warning |

Then `cutover`:

1. Embeds rows added or edited since the last `apply`, after showing how many and where
   their text goes.
2. Briefly holds off writes (reads continue) while it confirms every row has a new vector.
   A temporary index makes this check instant, even on large tables.
3. In one transaction: renames `embedding` to `embedding_old` and `embedding_v2` to
   `embedding`, renames vecshift's index to match, and moves the sync trigger to the old
   column.
4. Records the cutover in the state file.

If rows keep arriving between the catch-up and the switch, it catches up again, up to
three times. Rows the provider rejected block the switch; fix their text, or pass
`--allow-missing` to switch without them.

**At cutover, switch your application to the new model** for the queries it embeds and
the rows it writes. Its SQL doesn't change. If the vector size changes (say 1536 to 1024),
searches and inserts that still use the old model fail until it switches; if the size
stays the same, they run but match poorly. Either way, deploy the model change together
with the cutover.

### Views and functions

PostgreSQL ties views, materialized views, and SQL-standard functions (`BEGIN ATOMIC`) to
a column itself, not its name, so after a rename they would keep reading the old vectors.
Cutover refuses to run while any exist; drop them first and recreate them afterwards.
Ordinary SQL and PL/pgSQL functions, such as Supabase's `match_documents`, look columns up
by name each time they run and need no change.

## Rolling back

```bash
vecshift rollback             # asks first; --yes skips the question
```

Rollback renames the columns back in one transaction: the old vectors are live again
under the original name, and the new ones return to `embedding_v2`, where the sync
trigger keeps them current for another cutover. Switch your application back to the old
model at the same time.

After cutover, the sync trigger guards the old column: when a row's text changes, its old
vector is cleared. So rollback can tell you exactly how many rows were added or edited
since cutover and have no old-model vector. Your application needs to embed those with
the old model.

## Cleaning up

```bash
vecshift cleanup              # asks you to type the column name; --yes skips that
```

When you're sure you won't roll back, `cleanup` drops `embedding_old`, its index, and the
sync trigger guarding it, in one transaction. It can't be undone. Dropping a column is
instant; its space is reused as rows are updated, or reclaimed at once with `VACUUM FULL`
(which locks the table).

Use `cleanup` rather than dropping the column by hand: the sync trigger depends on it, so
PostgreSQL refuses a plain `DROP COLUMN` instead of leaving a trigger behind that would
break your application's updates.
