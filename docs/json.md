# JSON output

Every command that reports something takes `--json`, for scripts and CI. This page lists
what each prints. These shapes are a contract: the REST API planned for 0.5 returns the
same objects.

The exact shapes (every key, and the type of every value) are pinned in
[`tests/contracts/`](../tests/contracts), one file per kind of migration. The tests in
`tests/integration/test_json_contracts.py` run a whole migration against a real database
and Qdrant and compare.

## Stability

- **Adding a field isn't a breaking change.** Read the fields you need and ignore the rest.
- **Removing or renaming a field, or changing its type, is breaking.** It happens only in
  a minor release before 1.0, and the changelog says so.
- Values marked *nullable* below can be `null`, for example before a migration has started.
- With `--json`, standard output holds only JSON, so `vecshift plan --json | jq` works.
  Errors, warnings, and prompts go to standard error.
- The exit code still means what it does without `--json` (see each command).

## `doctor --json`

One object: the report.

| Field | Meaning |
|---|---|
| `connection` | Where it looked, with any password removed |
| `store` | `pgvector` or `qdrant` |
| `target` | The column (`schema.table.column`) or vector (`collection` or `collection:vector`) inspected |
| `vector_type`, `declared_dimensions` | The stored type and its declared size (nullable) |
| `estimated_rows` | Rows or points, estimated or counted (nullable) |
| `sample` | `rows` inspected and the `method` used to choose them |
| `facts` | What was measured: `dimensions` and `models` (counts keyed by size and by model), `norms`, `sample` counts (null vectors, duplicates, texts), `ann_indexes`, `text_field`, `model_field`, `updated_at_field`, and store details |
| `findings` | A list of `{id, severity, title, detail, hint}`; `severity` is `error`, `warning`, `info`, or `ok`, and `hint` is nullable |
| `summary` | The number of findings at each severity |

Exit code: 0, or 1 with `--fail-on` when a finding is that severe; 2 when it can't connect
or the target is ambiguous.

## `bench --json`

| Field | Meaning |
|---|---|
| `source` | Where the documents came from |
| `documents`, `queries` | How many of each were used |
| `query_source` | How the queries were made: proxy sentences, generated, or labeled |
| `models` | One entry per model: `spec`, `name`, `model_tag`, `dimensions`, `scores` (`recall_at_1`, `recall_at_10`, `mrr_at_10`, `ndcg_at_10`), query latency (`query_ms_p50`, `query_ms_p95`), throughput, token and cost estimates, storage per million vectors, `cached_docs`, and `error` (nullable: a model that failed) |
| `notes` | Plain-language caveats about the run |

## `plan --json`

| Field | Meaning |
|---|---|
| `ok` | Whether the plan has no errors; `apply` refuses when it's false |
| `job`, `model`, `source` | The job's name, the new model, and where the vectors are |
| `store` | Where the new vectors go: `pgvector` or `qdrant` |
| `target_column` | The new column, or for Qdrant `collection` or `collection:vector` |
| `vector_type`, `dimensions`, `dimensions_source`, `metric` | The new vectors |
| `changes` | What `apply` and `cutover` will do: `{kind, summary, note, sql}`. `sql` is set for pgvector and null for Qdrant; `note` is nullable |
| `estimates` | `rows`, `rows_exact`, `tokens`, `tokens_method`, `requests`, `cost_usd`, `seconds` and `seconds_method` (nullable without a rate limit), `old_bytes`, `new_bytes`, and for pgvector `index_memory_bytes` and `maintenance_work_mem` (null for Qdrant) |
| `findings` | As for `doctor` |

Exit code: 0 when `ok`, 1 when not; 2 when the job can't be read or the store can't be
reached.

## `apply --json`

JSON lines: one object per event as the run goes, each with an `event` field.

| `event` | Fields |
|---|---|
| `column`, `trigger` | `added`: whether apply created the column or trigger (pgvector) |
| `start` | `pending` rows to embed, and `spent_before` on earlier runs |
| `batch` | `written` and `failed` in this batch, `skipped`, running `rows_written` and `spent_usd`, and the `pass` number |
| `pass` | A new pass over rows edited during the last one: `number`, `remaining` |
| `index` | `state` (`building`, then the result) and `method` |
| `done` | `status` and `message` |
| `result` | Always last: `status` (`complete`, `stopped`, `budget`, or `failed`), `message`, `rows_written`, `rows_failed`, `remaining`, `tokens`, `spent_usd`, `total_spent_usd`, `seconds`, `index`. For Qdrant also `payloads_copied` and `deleted` |

Read the last line for the outcome. Exit code: 0 when complete, 1 when it failed, and 3
when it stopped early (`--until`, Ctrl-C, or the budget).

## `status --json`

| Field | Meaning |
|---|---|
| `stage` | `not_started`, `filling`, `ready`, `cut_over`, `cleaned_up`, or `conflict` |
| `next` | The command to run next, in words |
| `store`, `source_store` | Where the new vectors go, and where the old ones are |
| `live`, `new`, `previous` | The name searches use, the new column or collection, and the old one |
| `rows`, `missing`, `filled` | Rows in the source, rows still to embed, and the fraction done (nullable until there's something to count) |
| `index` | The new vectors' index, once built (nullable) |
| `runs`, `rows_written`, `spent_usd`, `failed` | Totals from the state file |
| `last_event` | The latest cutover, rollback, or cleanup (nullable). Its fields differ by store: see below |
| `table`, `trigger` | pgvector only; present but meaningless for Qdrant |

`status` only reads, and exits 0 at every stage.

## `eval --json`

| Field | Meaning |
|---|---|
| `verdict` | `status` (`go`, `no_go`, or `inconclusive`), with `reasons` and `warnings` |
| `mode` | `queries` when both sides embed the queries, or `vectors` when the old model isn't known and the old side is searched with stored vectors |
| `queries`, `query_source` | How many queries, and where they came from |
| `old`, `new` | Each side: `column` (or collection), `model`, `index_method`, `embedded`, `scores` (keyed by language pair, plus `all`: `recall_at_1`, `recall_at_10`, `mrr_at_10`, `queries`), `embed_latency`, `sweep` (search latency and `index_recall` at each search setting, such as `ef_search` or `hnsw_ef`), `same_group_at_10`, and `note` (nullable) |
| `result_overlap_at_10`, `neighbour_overlap_at_10` | How much the two sides agree |
| `partial` | Whether the new side was still incomplete |
| `notes` | Plain-language caveats |

Exit code: 0 for go, 1 for no-go, 3 for inconclusive.

## `cutover`, `rollback`, and `cleanup --json`

`cutover --check --json` and `cutover --json` print one object:

| Field | Meaning |
|---|---|
| `status` | `ready` or `blocked` (with `--check`, which exits 1 when blocked), or `cut_over` |
| `new`, `previous` | The new column or collection, and the old one |
| `caught_up` | Rows embedded in the final catch-up |
| `pending` | Rows still missing (0 after a cutover) |
| `findings` | As for `doctor` |
| `live` | pgvector: the column name searches use |
| `alias`, `late` | Qdrant: the alias that moved, and writes that reached the old collection too late to copy |

`rollback --json` prints `status: rolled_back`, `live`, and `new`, plus for pgvector
`missing` (rows to re-embed before cutting over again) and for Qdrant `alias` and `changed`
(points added or edited in the new collection since cutover). `cleanup --json` prints
`status: cleaned_up` and the name removed: `dropped` (the pgvector column) or `deleted`
(the Qdrant collection).

## Differences between stores

These are pinned as they are today. The service layer planned for 0.5 will give both
stores the same names, adding the shared names next to the current ones first, so nothing
breaks:

| Command | pgvector | Qdrant |
|---|---|---|
| `cutover` | `live` | `alias`, `late` |
| `rollback` | `missing` | `alias`, `changed` |
| `cleanup` | `dropped` | `deleted` |
| `status` `last_event` | `table`, `live` | `store`, `alias` |
