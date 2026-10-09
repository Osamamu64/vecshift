# Security model

VecShift connects to production databases and sends text to embedding providers, so it is
built to be safe by default. This page says what it touches, what it sends, what it keeps,
and how it protects credentials. To report a vulnerability, see
[SECURITY.md](../SECURITY.md).

## What leaves your machine

| Command | Sends | To |
|---|---|---|
| `doctor` | Nothing. Only statistics come back from the database; vectors and row text never leave it. | — |
| `plan` | Nothing, unless you pass `--probe`, which sends 16 sample rows. | The model's API, after you confirm |
| `bench` | The sampled documents and queries | Each model's API, after you confirm |
| `bench --generate-queries` | The sampled documents | The chat model's API, after you confirm |
| `apply` | Every row's text | The new model's API, after you confirm |
| `cutover` | The text of rows added or edited since the last `apply` | The new model's API, after you confirm |
| `eval` | Short queries: sentences from sampled rows, your labeled queries, or LLM-written ones | The old and new models' APIs (and the chat model), after you confirm |

- Before any text goes to a remote API, vecshift says how much and to where, and asks.
  `--yes` skips the question. Without a terminal it refuses unless `--yes` is given.
  Local models (Ollama, `localhost` URLs, the `hash` baseline) don't need confirmation.
- Reports (`--html`, `--json`) contain statistics, column names, and model names, never
  vectors or row text. HTML reports load nothing from the network.

## Credentials

- **Database connection strings** come from an environment variable (`VECSHIFT_DSN` by
  default). Job files can't hold them: they name the variable instead, and unknown
  settings such as `dsn:` are rejected. Passing `--dsn` with a password on the command
  line prints a warning, because other users can see it with `ps`.
- **Passwords are never printed.** Connections are shown with the password removed, and
  crash tracebacks never include local variables.
- **API keys** come only from environment variables (`OPENAI_API_KEY`, or the variable a
  spec names with `key_env=`). A key is sent only to the provider it belongs to:
  - `compat` servers receive a key only when the spec names one with `key_env=`.
  - A key is never sent over unencrypted `http://` to a remote host.
  - Redirects are never followed, so a key can't be carried to another host.
  - TLS certificates are always verified.
  - Credentials inside URLs (`https://user:pass@...`) are rejected, and URL query strings
    are hidden wherever a spec is displayed.
- Error messages from providers are shortened and never include the request's key.

## The database

- `doctor`, `bench`, `plan`, and `eval` connect read-only, so the database rejects any write, and
  every statement runs under a timeout.
- `apply`, `cutover`, `rollback`, and `cleanup` are the only commands that write. Each
  shows what it will change and asks first (`--yes` skips the question; without a
  terminal it refuses unless `--yes` is given). `cleanup`, the only one that deletes data,
  asks you to type the column's name.
- `apply` only adds things: a column, a trigger and its function, and an index. Its row
  writes touch only the new column. `cutover` and `rollback` only rename columns and
  indexes and move the trigger; no vectors are deleted until `cleanup`.
- vecshift's trigger functions pin their `search_path`, so objects in other schemas can't
  change what they do.
- Schema changes wait at most a few seconds for a lock and then back off, so `apply` never
  queues behind application queries and blocks them. A session advisory lock stops two
  runs from working on the same column at once.
- Large tables are sampled by page (`TABLESAMPLE SYSTEM`) instead of scanned.
- Every query uses bound parameters, or identifiers composed with `psycopg.sql`. The SQL
  that `plan` shows for `apply` quotes identifiers using the server's own keyword list, so
  names like `order` or `user` stay correct.
- Supabase connections require TLS. Row-level security is respected, and reported when it
  hides rows.
- Use a role with only the access a command needs. `doctor`, `bench`, and `plan` need
  read access; [docs/connectors/pgvector.md](connectors/pgvector.md) has a read-only role
  recipe. `apply` needs to own the table; see
  [Permissions](migrations.md#permissions).

## Data kept on disk

- **Embedding cache**: `~/.cache/vecshift/embeddings.sqlite` (or under `$XDG_CACHE_HOME`).
  It holds vectors keyed by a SHA-256 hash of the model configuration and text; the text
  itself isn't stored. Because embeddings can be partly inverted back into text, the
  directory is created readable by its owner only (`0700`) and the file `0600`. Delete
  the directory to clear it, or pass `--no-cache`.
- **Migration state**: `.vecshift/<job>.state.json` next to the job file holds spend,
  token and row counts, and the IDs of rows the provider rejected, with the provider's
  error. Never row text or credentials. The directory is `0700` and the file `0600`, and
  it is replaced atomically so a crash can't leave it half-written.
- **Files you ask for**: reports and saved queries are written only where you point them.
  Saved generated queries contain the query text and document IDs.

## HTML reports

Reports (`doctor --html`, `bench --html`, `eval --html`) escape everything that comes from
the database or a model, and carry a strict Content Security Policy: no network access at
all, and only vecshift's own script, identified by its SHA-256 hash, may run. Even if an escaping bug ever let a name inject
markup, the browser would refuse to run it or send anything anywhere.

## Supply chain

- Dependencies are locked with hashes in `uv.lock`, and CI installs with `--locked`.
- CI checks every change and runs weekly:
  - `pip-audit` for known vulnerabilities in locked dependencies
  - `zizmor` for GitHub Actions security
  - Ruff's flake8-bandit rules (`S`) for code patterns
- GitHub Actions are pinned to full commit SHAs, workflows get read-only tokens, and
  checkout doesn't persist credentials. Dependabot keeps dependencies and pins up to date.

## For maintainers

These repository settings complete the picture (Settings → Code security, and Branches):

- Secret scanning with push protection
- Dependabot alerts and security updates
- Code scanning with CodeQL, default setup
- Private vulnerability reporting
- A branch rule on `main` requiring pull requests and passing CI
