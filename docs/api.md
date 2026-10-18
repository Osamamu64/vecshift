# REST API

`vecshift serve` answers the same questions as the CLI over HTTP, through the same code, so
its JSON is the CLI's `--json` output. It's the base for the web UI, scripts, and other
tools.

> **Preview.** This release has the read-only endpoints: jobs, `plan`, `status`, and
> `doctor`. Running `apply`, `eval`, `cutover`, `rollback`, and `cleanup` over HTTP, with
> live progress, comes next.

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
| 404 | `job.missing`, `source.not_found` |
| 409 | `env.missing`, `source.ambiguous`, `job.running`, `plan.errors`, `eval.not_applied`, `state.no_record` |
| 413 | `request.too_large` |
| 421 | `request.host`: addressed to a name other than localhost, without a token |
| 422 | `request.invalid` (with `errors`), `job.invalid`, `options.invalid`, `consent.required`, `confirm.required`, `source.unsupported` |
| 500 | `internal`: a bug; the server logs the error's type, never its message |
| 502 | `connect.failed`, `database.error`, `qdrant.error`, `model.failed` |
| 503 | `database.busy` |

Other codes may appear with status 400; `code` is the field to rely on.

## Compatibility

`/api/v1` only grows: new endpoints and new fields. Removing or renaming anything means
`/api/v2`. `tests/test_server.py` fails when the code and `docs/openapi.json` differ; after
a deliberate change, regenerate it with `VECSHIFT_UPDATE_CONTRACTS=1` and note it in the
changelog.
