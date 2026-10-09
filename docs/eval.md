# Evaluating a migration

`vecshift eval` answers the question to ask before cutover: is the new model better on
*your* data, and how fast is it? It compares the old and new vectors already in your
database, so it only embeds short queries, never your documents.

```bash
vecshift eval                                  # quality + latency, then GO / NO-GO
vecshift eval --json                           # the same, for scripts and CI
vecshift eval --generate-queries openai/gpt-4o-mini --cross-language
vecshift eval --max-p95-ms 80                  # also fail if the new setup is too slow
```

It works on a partial migration too (`apply --until 20`), so you can check a sample
before paying to embed the rest. It only reads: it connects read-only, runs a limited
number of queries under a timeout, and can point at a replica with `--dsn-env`.

## Two modes

Searching the old column needs queries embedded by the old model. Eval tests that model
with one query first, then picks a mode:

| Mode | When | How each side is judged |
|---|---|---|
| **queries** | The old model works (from `source.model` or `--old-model`) | Each query is embedded by both models and searched in both columns. |
| **vectors** | The old model is unknown, retired, or returns the wrong size | The old side is judged from its stored vectors alone (see below), the new side from queries too. |

Name the old model in the job file to get the stronger mode:

```yaml
source:
  table: public.documents
  model: openai/text-embedding-3-small   # the model that made the current vectors
```

## Queries

| Source | Option | Cost |
|---|---|---|
| A sentence taken from a row, which is the right answer | (default) | Free, no LLM |
| Your labeled queries: `{"query": ..., "relevant": [ids]}` per line | `--queries FILE` | Free |
| Realistic questions written by an LLM | `--generate-queries CHAT_MODEL` | Small; asks first |
| Questions in the other language: Arabic for Latin-script rows, English for Arabic ones | `--cross-language` | Same as above |

Sentences taken from rows are still part of those rows' stored text, so they're easy,
known-item searches: absolute scores run high, but the comparison between old and new is
fair. Labeled or generated queries give numbers closer to real search.

## Search quality, per language

For each side eval reports recall@10 (how often the right row is in the top 10),
recall@1, and MRR@10, for all queries and for each **query → document script** pair:
`latin→latin`, `arabic→arabic`, and, with cross-language queries, `arabic→latin` and
`latin→arabic`. Scripts are detected from the letters themselves (Arabic or Latin), so no
language model is needed. A model that's fine on English and poor on Arabic shows up in
its own row instead of hiding in the average.

It also reports how many of the top 10 results the two sides share. A big change isn't
bad by itself, but it's worth knowing before users notice.

## Nearest rows, from stored vectors

For a sample of rows, eval asks each column "which rows are closest to this one?" using
the row's own stored vector, so it needs no model at all. If rows are chunks of larger
documents, it measures how many of a chunk's 10 nearest rows come from the same document:
a model that understands your content keeps a document's chunks together. Eval finds the
document key on its own (columns such as `document_id` or `source`, or the same keys in a
JSON `metadata` column), or you name it with `--group-by document_id` or
`--group-by metadata.file_name`.

This is what makes the **vectors** mode possible, and it's a useful second signal in the
**queries** mode.

## Latency vs accuracy

For each side eval measures:

- **Query embedding** time, p50 and p95, one query at a time, from where vecshift runs.
  Your application may sit in another region, so compare old with new rather than reading
  the numbers as absolute.
- **Search** time in the database, p50, p95, and p99, through the real index.
- **Index recall**: the share of the exact top 10 (found by scanning every row) that the
  approximate index returns.

Search time and index recall are measured at several settings of the index's main knob
(`hnsw.ef_search`, or `ivfflat.probes`), giving a latency/accuracy curve per side. The
current setting is marked. If the new model needs a higher setting to keep recall, the
curve shows what that costs. When the old model can't embed queries, its search latency
is measured with stored row vectors as queries, since speed doesn't depend on the model.

On a partial migration both sides are searched exactly over the rows that have both
vectors, and latency is skipped: an index over half the rows isn't the real thing.

## The verdict

| Exit status | Verdict |
|---|---|
| 0 | **GO**: recall@10 held (within `--tolerance`, default 0.02) or improved, overall and for every language pair with at least `--min-slice` queries (default 30), and any latency limit was met |
| 1 | **NO-GO**: one of those failed; the reasons say which |
| 2 | Couldn't run: no new vectors yet, a bad option, or no confirmation |
| 3 | **INCONCLUSIVE**: not enough evidence, such as too few queries, or no old model and no way to group rows by document |

Latency limits are optional: `--max-p95-ms 80` caps the new end-to-end p95 (query
embedding plus search), and `--max-slowdown 20` fails if it's more than 20% slower than
the old one. Warnings, such as an index finding less than 90% of the exact top 10, don't
change the verdict.

## What leaves your machine

Only query text: sentences from your rows, your labeled queries, or questions an LLM
wrote from sampled rows. They go to the old and new models (and the chat model, when
generating), after you confirm. Local models don't need confirmation. Results hold scores,
timings, and row counts, never document text.
