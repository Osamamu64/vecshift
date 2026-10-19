# REST API

`vecshift serve` answers the same questions as the CLI over HTTP, through the same code, so
its JSON is the CLI's `--json` output. It's the base for the web UI, scripts, and other
tools.

> **Preview.** The endpoints may still change before 0.5.0, when the web UI built on them
> ships.

## Running it

```bash
pip install "vecshift[server]"           # FastAPI and uvicorn, kept out of the core install
export VECSHIFT_DSN=postgresql://...     # the same variables the CLI reads, or a .env file
vecshift serve --jobs ./migrations       # http://127.0.0.1:8765/api/v1
```

- `--jobs` is a folder of job files (`*.yaml` or `*.yml`); the default is the current
  folder. A job's name is its file name without the extension: `docs-reembed.yaml` is
  `docs-reembed`. Each job keeps its state in `.vecshift/` next to it, as with the CLI, so
  the CLI and the API can work on the same jobs.
- `--host` and `--port` say where to listen; the default is `127.0.0.1:8765`.

```console
$ curl -s localhost:8765/api/v1/jobs
{"jobs": [{"name": "docs-reembed", "valid": true, "stage": "filling", ...}]}
$ curl -s localhost:8765/api/v1/jobs/docs-reembed/plan
```

## Who can call it

- **Only this machine, by default.** The server listens on `127.0.0.1` and answers only
  requests addressed to `localhost`, `127.0.0.1`, or `[::1]`. A web page can't reach it by
  pointing its own domain name at 127.0.0.1 (DNS rebinding): those requests get **421**.
- **Anywhere else needs a token.** To listen on another address, set `VECSHIFT_API_TOKEN`
  to a random string of at least 16 characters; `vecshift serve` refuses to start without
  one. Every request then needs `Authorization: Bearer <token>` (compared in constant
  time), except `GET /api/v1/health`. Put it behind TLS if it crosses a network.
- **Requests other than GET need `X-Vecshift: 1`.** A browser can't add that header to a
  request from another site, and the server never sends CORS headers, so another web page
  can't make the server act.
- **Request bodies are capped** at 1 MB.
- **Responses** are never cached, sniffed, framed, or allowed to run anything.

## No secrets in, none out

Requests name environment variables (`dsn_env`, `url_env`, `key_env`), never their values,
and responses never contain a value: connection strings appear only as
`postgresql://user@host:port/db`, and keys only as set or not. URL query strings in model
specs (`url=…?token=…`) are shown as `?…`. Error messages are checked for the values of
the variables vecshift reads, and tracebacks never leave the server.

## Endpoints

All paths start with `/api/v1`. The schema is at `/api/v1/openapi.json`, and committed as
[`openapi.json`](openapi.json).

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/version` | `{"version": "0.4.0", "api": "v1"}` |
| GET | `/environment` | Each variable vecshift may read: `name`, `kind` (`connection`, `url`, or `key`), `set`, `display` (where a connection string or URL points), `problem` (why a value can't be used), and `used_by` (jobs) |
| GET | `/jobs` | Every job file: `name`, `valid`, `source` and `target` (`{type, name}`), `model`, `stage`, `updated_at`, and `problem` when the file isn't a valid job or its stage can't be read |
| GET | `/jobs/{name}` | `spec` (the job as JSON), `yaml` (as written), and `state`: runs, rows written, tokens, money spent, rejected rows, history, and the Qdrant collections the job created |
| GET | `/jobs/{name}/plan` | The same JSON as `vecshift plan --json`. Query: `probe`, `sample` (50–100,000), `allow_remote` |
| GET | `/jobs/{name}/status` | The same JSON as `vecshift status --json`. Query: `full` |
| POST | `/doctor` | The same JSON as `vecshift doctor --json` |

**Quick and full status.** `status` counts the rows still missing a vector, which reads the
whole table (or, for Qdrant, the whole source). The API doesn't count unless you pass
`full=true`, and the jobs list never does: until then, `rows` and `missing` are `null`, and
a migration that's ready to cut over shows as `filling`.

**Probing a remote model.** `plan?probe=true` embeds 16 sample rows with the job's model.
If the model isn't on this machine, that sends text away, so it also needs
`allow_remote=true`; without it the answer is **422**, naming where the text would go.

**Doctor.** The body says what to inspect, like the CLI's options:

```json
{"store": "pgvector", "dsn_env": "VECSHIFT_DSN", "table": "public.documents", "column": "embedding", "sample": 2000}
{"store": "qdrant", "url_env": "VECSHIFT_QDRANT_URL", "key_env": "QDRANT_API_KEY", "collection": "docs"}
```

The variables must be ones vecshift reads here (`GET /environment` lists them), so a
request can't make the server read any other variable. When the table has several vector
columns (or the Qdrant collection several vectors), the answer is **409** with
`candidates` to choose from.

## Editing jobs

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/jobs` | `{name, spec, dry_run}` | **201**: the job, as `GET /jobs/{name}` returns it, plus `written` and `comments_kept`. **409** `job.exists` |
| PUT | `/jobs/{name}` | `{spec, dry_run}` | The changed job. **409** `job.running` while the job has a run |
| DELETE | `/jobs/{name}` | | **204**. The job file only: the vectors, the stores, and the job's progress in `.vecshift/` stay. **409** `job.running` while the job has a run |

`spec` is the job as JSON, the same shape `GET /jobs/{name}` returns and the job file
holds. `name` is the file's name (`notes` writes `notes.yaml`), and `spec.name` defaults to
it. With `dry_run: true` nothing is written: the answer shows the file it would write.

```console
$ curl -s -X POST localhost:8765/api/v1/jobs -H 'X-Vecshift: 1' \
    -H 'content-type: application/json' \
    -d '{"name": "notes", "spec": {"source": {"table": "public.notes"}, "model": "openai/text-embedding-3-small"}}'
```

**Checked first.** A job that isn't valid is a **422** (`job.invalid`): `detail` lists the
problems in the words `vecshift plan` uses for a job file, and `errors` has them one by one
(`{"loc": "source.collection", "message": "required"}`).

**Comments stay.** A new file starts from the commented template `vecshift init` writes.
A change edits the file in place: each setting keeps its line and its comment, a setting
that's no longer set is commented out, and a new one fills in its commented-out example
when there is one. If the file is laid out in a way vecshift can't follow line by line
(flow style like `source: {table: docs}`, block text, anchors), it's written afresh and
`comments_kept` is `false`.

**Names.** vecshift keeps a job's progress under its `name`, so two job files can't share
one (**409** `job.name_taken`), and a job that has started can't be renamed (**409**
`job.renamed`): its progress would be lost track of.

**Model URLs.** `GET /jobs/{name}` shows a model URL's query string as `?…`, as it can hold
a key. Sending that back unchanged keeps what the file has; a new query string is refused
(**422**), since requests never carry secrets. Read a key from a variable with
`key_env=NAME` instead, or edit the file.

## Runs

`apply`, `eval`, `cutover`, `rollback`, and `cleanup` take a while, so they run in the
background: you start a run, then follow its events.

| Method | Path | Returns |
|---|---|---|
| POST | `/jobs/{name}/runs` | **202** `{run_id, links}`: the run started. **409** `job.running` if the job already has one |
| GET | `/jobs/{name}/runs` | The job's runs, newest first |
| GET | `/runs/{run_id}` | `run_id`, `job`, `action`, `state`, `started_at`, `finished_at`, `result`, `problem`, and `events` (how many so far) |
| GET | `/runs/{run_id}/events` | The run's events as [server-sent events](#events) |
| POST | `/runs/{run_id}/stop` | **202**: apply stops after the current batch, like the first Ctrl-C |
| GET | `/runs/{run_id}/report` | eval's or bench's HTML report, as `--html` writes it; `?format=json` for its JSON |

A run's `state` is `queued`, `running`, `stopping`, then `completed`, `stopped` (apply
stopped by request or by the budget: run it again to continue), or `failed` (`problem` says
why). When it's finished, `result` is the same JSON as the command's `--json` output:
apply's last line, or the whole report for eval.

### Starting one

```console
$ curl -s -X POST localhost:8765/api/v1/jobs/docs-reembed/runs -H 'X-Vecshift: 1' \
    -H 'content-type: application/json' \
    -d '{"action": "apply", "confirm": {"apply": true}, "allow_remote": true}'
{"run_id": "01JAB…", "links": {"self": "/api/v1/runs/01JAB…", "events": "…/events", "stop": "…/stop"}}
```

The body has the `action`, its `options` (the CLI's flags), and the confirmation it needs.
Nothing starts without it: a missing confirmation is a **422** (`confirm.required`).

| Action | `options` | Needs |
|---|---|---|
| `apply` | `no_index`, `until` (percent) | `confirm.apply: true`: it changes the database and spends money |
| `eval` | `num_queries`, `queries` (labeled queries inline: `[{"query", "relevant": [ids]}]`), `generate_queries`, `cross_language`, `old_model`, `group_by`, `neighbours`, `tolerance`, `min_slice`, `max_p95_ms`, `max_slowdown`, `seed` | nothing: it only reads |
| `cutover` | `check_only`, `allow_missing` | `confirm.switch: true`, unless `check_only` |
| `rollback` | none | `confirm.switch: true` |
| `cleanup` | none | `confirm.name`: the old column or collection it deletes, which must match what vecshift computes from the job's own record (`previous` in the job's status) |

**Text sent away.** apply sends every row's text to the job's model, eval sends queries to
the old and new models (and the chat model with `generate_queries`), and cutover first
embeds the rows that changed since the last apply. When any of those models isn't on this
machine, the request needs `"allow_remote": true`; without it the answer is a **422**
(`consent.required`) naming where the text would go, and nothing starts. (For cutover the
run fails before it changes anything, as it only knows once it has checked.)

**One at a time.** A job has at most one active run. The stores' own locks still keep a
CLI command and a run from working on the same job at once.

### Events

`GET /runs/{run_id}/events` is a stream of server-sent events. Each has an increasing `id`;
to pick up where you left off, reconnect with `Last-Event-ID` (browsers do it on their own)
or `?after=N`. The stream ends once the run has finished and every event was sent, and a
comment line every 15 seconds keeps proxies from closing a quiet stream.

```
id: 3
event: start
data: {"pending": 3000, "spent_before": 0.0}

id: 4
event: batch
data: {"pass": 1, "written": 256, "skipped": 0, "failed": 0, "rows_written": 256, "spent_usd": 0.0}
```

| Event | From | `data` |
|---|---|---|
| `column`, `trigger`, `start`, `batch`, `pass`, `index`, `done` | apply | exactly what `apply --json` prints, line by line |
| `step` | eval, bench | `{"message": "Embedding queries"}`: what it's doing now |
| `plan` | bench | what it's about to embed: the models, tokens, cost, and where each one sends the text |
| `finding` | cutover | a readiness check, as in `cutover --json`'s `findings` |
| `switched` | cutover, rollback | which names now point where |
| `deleted` | cleanup | `{"name": …}` |
| `result` | every action | the same JSON as the command's `--json`; the last event of a run that worked |
| `error` | every action | a problem (`code`, `detail`, `hint`); the last event of a run that failed |

### Runs on disk

Each run's events go to `.vecshift/runs/<run_id>.jsonl` in the jobs folder as they happen,
and its summary to `<run_id>.json` (and eval's or bench's report to `<run_id>.html`), readable by
their owner only. They hold counts, names, and messages, never document text or secrets.
After a restart the server still lists and streams old runs; a run that was still going
when the server stopped is marked `failed` (`run.interrupted`). Every operation is
resumable, so running it again picks up where it stopped.

## Bench

| Method | Path | Returns |
|---|---|---|
| POST | `/bench/runs` | **202** `{run_id, links}`: [`vecshift bench`](bench.md) as a run. **409** `job.running` while another bench runs |
| GET | `/bench/runs` | Bench runs, newest first |

A bench run is followed like any other: `GET /runs/{run_id}` and its events. Its `job` is
`null`, its `result` is the same JSON as `bench --json`, and `/runs/{run_id}/report` is the
HTML leaderboard `bench --html` writes. It can't be stopped halfway. When every model
fails, the run is `failed` (`model.failed`) and still has its result, with each model's
error.

```json
{
  "documents": {"type": "job", "job": "docs-reembed"},
  "models": ["openai/text-embedding-3-small", "ollama/nomic-embed-text"],
  "sample": 1000,
  "num_queries": 100,
  "allow_remote": true
}
```

| Field | |
|---|---|
| `documents` | Where the documents come from: `{"type": "job", "job"}` samples a job's source; `{"type": "pgvector", "dsn_env", "table", "column", "text_column"}` a table; `{"type": "qdrant", "url_env", "key_env", "collection", "vector", "text_field"}` a collection; `{"type": "documents", "documents": [{"id", "text"}]}` takes them from the request |
| `models` | The models to compare (up to 20); two free hashing baselines by default |
| `sample` | Documents to sample (10–100,000), 1,000 by default |
| `num_queries` | Queries to make (5–10,000), 100 by default |
| `queries` | Labeled queries, `[{"query", "relevant": [ids]}]`, instead of proxy ones |
| `generate_queries` | A chat model that writes the queries, such as `openai/gpt-4o-mini` |
| `seed`, `cache` | The sampling seed (7), and whether to reuse vectors from the embedding cache (yes) |
| `allow_remote` | Needed when a model, or the query writer, isn't on this machine |

As for doctor, `dsn_env`, `url_env`, and `key_env` must be variables vecshift reads here.
The events are `plan` (the models, estimated tokens and cost, and where each sends the
text), `step` (`Embedding with …`), then `result` or `error`.

## Errors

Errors are [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) `application/problem+json`.
`detail` and `hint` are the sentences the CLI prints; `code` is stable, for programs.

```json
{
  "type": "urn:vecshift:error:env.missing",
  "title": "Conflict",
  "status": 409,
  "detail": "VECSHIFT_DSN isn't set.",
  "hint": "Export the database connection string as VECSHIFT_DSN, put it in a .env file here, or run vecshift init to set it up.",
  "code": "env.missing"
}
```

| Status | Codes |
|---|---|
| 401 | `request.unauthorized`: the token is missing or wrong |
| 403 | `request.header`: `X-Vecshift: 1` is missing |
| 404 | `job.missing`, `run.missing`, `report.missing`, `source.not_found` |
| 409 | `env.missing`, `source.ambiguous`, `job.running`, `job.exists`, `job.name_taken`, `job.renamed`, `run.finished`, `run.not_stoppable`, `plan.errors`, `eval.not_applied`, `state.no_record` |
| 413 | `request.too_large` |
| 421 | `request.host`: addressed to a name other than localhost, without a token |
| 422 | `request.invalid` (with `errors`), `job.invalid`, `options.invalid`, `consent.required`, `confirm.required`, `confirm.mismatch`, `source.unsupported`, `documents.invalid` |
| 500 | `internal`: a bug; the server logs the error's type, never its message |
| 502 | `connect.failed`, `database.error`, `qdrant.error`, `model.failed` |
| 503 | `database.busy` |

Other codes may appear with status 400; `code` is the field to rely on. A failed run
carries the same `code` in its `problem` and its `error` event.

## Compatibility

`/api/v1` only grows: new endpoints and new fields. Removing or renaming anything means
`/api/v2`. `tests/test_server.py` fails when the code and `docs/openapi.json` differ; after
a deliberate change, regenerate it with `VECSHIFT_UPDATE_CONTRACTS=1` and note it in the
changelog.
