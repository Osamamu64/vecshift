# pgvector and Supabase

VecShift reads from PostgreSQL tables with [pgvector](https://github.com/pgvector/pgvector)
columns, including Supabase projects.

The PostgreSQL driver comes with VecShift:

```bash
pip install vecshift
```

## Running `doctor`

Put the connection string in an environment variable rather than on the command line, so
the password stays out of your shell history:

```bash
export VECSHIFT_DSN='postgresql://user:password@host:5432/dbname'
vecshift doctor
```

`doctor` finds every `vector` and `halfvec` column it can see. If there's exactly one, it
inspects it. If there are several, it lists them and asks you to choose:

```bash
vecshift doctor --table public.documents
vecshift doctor --table rag.chunks --column embedding
```

| Option | Purpose |
|---|---|
| `--table` | `table` or `schema.table` to inspect. |
| `--column` | The vector column, when a table has more than one. |
| `--text-column` | The column holding source text, if it isn't detected. |
| `--sample-size` | Maximum rows to inspect (default 2,000). |
| `--timeout` | Statement timeout in seconds (default 60). |
| `--json` | Machine-readable output. |
| `--html FILE` | Also write a self-contained HTML report: verdict, key numbers, findings, and charts. |
| `--fail-on` | `warning` or `error`: exit with status 1 if a finding is this severe. Useful in CI. |

### What it detects

`doctor` looks at the schema and at a sample of rows. These columns are recognized
automatically:

| Purpose | Recognized columns |
|---|---|
| Source text | `content`, `text`, `page_content`, `document`, `chunk`, `chunk_text`, `body` |
| Model | `model_tag`, `embedding_model`, `model`, `model_name`, or the same keys inside a `metadata` / `cmetadata` / `meta` JSON column |
| Change tracking | `updated_at`, `modified_at`, `last_modified`, `changed_at` (timestamp types) |

That covers the default layouts of Supabase's vector guides, LangChain, and LlamaIndex.

## Safety

`doctor` is safe to run against production:

- The connection is **read-only**; the database rejects any write.
- Every statement has a **timeout** (`--timeout`).
- **Vectors never leave the database.** Dimensions, norms, and hashes are computed in SQL,
  and only those numbers are returned.
- On large tables it reads a **page-level sample** (`TABLESAMPLE SYSTEM`) rather than
  scanning the whole table, then keeps rows spread across that sample, so rows written
  recently are represented as well as old ones.
- The **HTML report** contains statistics, column names, and model names only. It never
  includes vectors or row text, and it loads nothing from the network.
- Passwords are never printed. Connection strings are shown with the password removed.

## Supabase

### Which connection string to use

Copy a connection string from your project's **Connect** button in the Supabase dashboard.

| Connection | Host and port | Works with VecShift | Notes |
|---|---|---|---|
| Direct | `db.<ref>.supabase.co:5432` | Yes | IPv6 only unless your project has the IPv4 add-on. |
| Session pooler | `<region>.pooler.supabase.com:5432` | Yes, **recommended** | Works over IPv4. User is `postgres.<ref>`. |
| Transaction pooler | `<region>.pooler.supabase.com:6543` | Yes | User is `postgres.<ref>`. |
| Dedicated pooler | `db.<ref>.supabase.co:6543` | Yes | Paid plans. |

If you're on a network without IPv6 (most home and office networks, many CI runners), use
the **session pooler**. If the direct connection fails with "Network is unreachable",
VecShift will suggest that switch.

VecShift handles the Supabase-specific details for you:

- **TLS** is required for Supabase hosts unless your string sets `sslmode` itself.
- **Prepared statements are disabled**, because the transaction poolers don't support them.
- All work happens inside **one transaction**, so session settings like the statement
  timeout behave the same through a transaction pooler.
- pgvector lives in Supabase's **`extensions` schema**. VecShift finds it wherever it's
  installed, even when it isn't on your role's search path.
- Supabase's internal schemas (`auth`, `storage`, `realtime`, and so on) are skipped.

### Row-level security

Tables exposed through Supabase's API usually have row-level security (RLS) turned on. A
role that doesn't bypass RLS only sees the rows its policies allow, so `doctor` would
inspect a partial table. When that's the case, the report warns you.

For a complete picture, connect as the `postgres` user, which bypasses RLS.

### A dedicated read-only role

If you'd rather not use `postgres`, create a role for VecShift that can only read. Run this
in the Supabase SQL editor, replacing the password and table:

```sql
create role vecshift_reader with login password 'use-a-strong-password';
grant usage on schema public to vecshift_reader;
grant usage on schema extensions to vecshift_reader;
grant select on public.documents to vecshift_reader;

-- If the table has RLS enabled, let the role read every row:
create policy "vecshift can read all rows" on public.documents
  for select to vecshift_reader using (true);
```

Then connect through the session pooler as `vecshift_reader.<project-ref>`.
