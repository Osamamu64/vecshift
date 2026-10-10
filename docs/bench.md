# Benchmarking embedding models

`vecshift bench` compares embedding models on a sample of your own documents. Public
leaderboards such as MTEB average over other people's data; this tells you which model
retrieves best on yours, how fast it is, and what it costs to run.

```bash
pip install vecshift
vecshift bench --docs docs.jsonl \
  -m openai/text-embedding-3-small \
  -m openai/text-embedding-3-large,dims=256 \
  -m ollama/nomic-embed-text \
  --html leaderboard.html
```

With no `-m`, it compares two free, offline hashing baselines, so a first run costs nothing.

## Where the documents come from

| Source | Options |
|---|---|
| A JSONL file | `--docs FILE`, one `{"id": "...", "text": "..."}` per line (`id` is optional) |
| A pgvector table | `--dsn` (or `VECSHIFT_DSN`), `--table`, and optionally `--column`, `--text-column` |

`--sample` sets how many documents to use (default 1,000). From a database, rows are
sampled spread across the table, the same way `vecshift doctor` samples. Documents longer
than 8,000 characters are cut to that length.

## Choosing models

Each `-m` is a model spec: `provider/model`, optionally followed by `,option=value` pairs.

| Provider | What it calls | API key |
|---|---|---|
| `openai` | OpenAI's API | `OPENAI_API_KEY` |
| `ollama` | Ollama on this machine (`http://localhost:11434/v1`) | none |
| `compat` | Any OpenAI-compatible server: vLLM, TEI, LM Studio, LiteLLM, hosted platforms. Needs `url=`. | Only the variable named by `key_env=` |
| `hash` | Built-in hashing baseline. The model is the dimension count, e.g. `hash/1024`. | none |

| Option | Meaning |
|---|---|
| `url=` | Base URL of the API, e.g. `url=http://localhost:8080/v1` |
| `dims=` | Ask the model for fewer dimensions (Matryoshka truncation), e.g. `dims=256` |
| `price=` | USD per million tokens, for the cost column. Known for OpenAI's models. |
| `key_env=` | Read the API key from this environment variable instead |
| `query_prefix=`, `doc_prefix=` | Text prepended to queries or documents |
| `batch=` | Texts per request (default 64) |

Some models are trained with prefixes and lose a lot of quality without them. VecShift adds
the right ones for `nomic-embed-text` (`search_query: ` / `search_document: `), E5 models
(`query: ` / `passage: `), and BGE English and `mxbai-embed-large` (a query instruction).
Override them with the options above, or set them empty to turn them off.

Examples:

```bash
-m openai/text-embedding-3-large,dims=1024          # compare truncation levels
-m compat/BAAI/bge-m3,url=http://localhost:8080/v1   # Text Embeddings Inference
-m compat/my-model,url=https://llm.example.com/v1,key_env=EXAMPLE_KEY,price=0.05
```

## Where the queries come from

Measuring retrieval needs queries whose right answers are known. There are three options.

**Proxy queries (default, free).** For each chosen document, one sentence is taken out and
used as the query; the model has to find the document from the rest of its text. This is
the inverse cloze task. It costs nothing and is good for *ranking models against each
other*, but it favours models that match on shared words, so the absolute numbers are
optimistic for lexical models and pessimistic for semantic ones.

Only sentences that appear once in the sample are used, because repeated boilerplate would
make every other document look like a better match than the true one. Documents that
differ only by filled-in values, such as generated tickets from one template, still defeat
this method: use one of the options below for those.

**Generated queries.** `--generate-queries openai/gpt-4o-mini` asks a chat model to write
a realistic search query for each document. Any OpenAI-compatible chat endpoint works. Add
`--save-queries queries.jsonl` to keep them and reuse them later with `--queries`, so you
only pay once.

**Labeled queries.** `--queries FILE` reads your own queries, one per line:

```json
{"query": "how do I rotate an API key", "relevant": ["doc-123", "doc-456"]}
```

The relevant documents are always included in the sample. Queries whose documents aren't
found are skipped and counted in the report.

## What's measured

| Column | Meaning |
|---|---|
| Recall@10 | Share of relevant documents found in the top 10 results. The leaderboard is sorted by it. |
| Recall@1 | Share found at the very top. |
| MRR@10 | Average of 1 / rank of the first relevant result (0 if it's outside the top 10). |
| Query p50 | Median time to embed one query, from 20 single requests. This is the latency your search adds. |
| Docs/s | Documents embedded per second in batches. Shows `cached` when nothing needed embedding. |
| Per 1M docs | Cost to embed a million documents of this length, from the provider's reported token counts. `~` marks an estimate. |
| Per 1M vectors | float32 storage for a million vectors of this size. |

Search is exact cosine similarity over every document, so the scores measure the models,
not an ANN index.

## Cost, privacy, and caching

- **You're asked first.** Before any text goes to a remote API, `bench` shows each model's
  estimated tokens and cost, says how many documents will be sent, and asks to continue.
  `--yes` skips the question; without a terminal, it stops unless `--yes` is given. Local
  models (Ollama, `localhost` URLs, `hash`) don't need confirmation.
- **Embeddings are cached** in `~/.cache/vecshift/embeddings.sqlite` (or under
  `$XDG_CACHE_HOME`), keyed by a hash of the model configuration and the text. The text
  itself isn't stored. Re-running a benchmark only embeds what changed. `--no-cache` turns
  this off.
- **Retries are built in.** Requests that hit rate limits or server errors are retried with
  backoff, respecting `Retry-After`.
- **API keys are only read from the environment** and never appear in output or errors.
  A key is never sent over plain `http://` to a remote host, redirects aren't followed,
  and credentials inside `url=` are rejected. See the [security model](security.md).

## Output

The terminal shows the leaderboard. `--json` prints the full results for scripts, and
`--html FILE` writes a self-contained leaderboard page with the best picks (best
retrieval, best value, fastest queries) and charts of quality against storage and query
speed. Like the doctor report, it loads nothing from the network and contains scores and
model names only, never document text.
