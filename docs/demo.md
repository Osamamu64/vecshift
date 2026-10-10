# Demo

The demo runs a complete migration on a sample table, start to finish, with Docker. It's
the quickest way to see what each command does before pointing vecshift at your own data.

```bash
git clone https://github.com/Osamamu64/vecshift.git
cd vecshift/demo
docker compose up --build --abort-on-container-exit
```

It needs Docker with Compose v2 and takes a minute or two, most of it building the image.

## What it runs

The demo starts PostgreSQL with pgvector, then [`run.sh`](../demo/run.sh) runs these steps
in the vecshift container:

1. **Seed.** [`seed.py`](../demo/seed.py) creates `documents`, 400 short help-centre
   articles, half in English and half in Arabic, with 64-dimension vectors and an HNSW
   index.
2. **`vecshift doctor`** inspects the index and writes `demo/out/doctor.html`. The sample
   data has a few duplicate articles, so expect warnings about those.
3. **The job file.** The migration moves from a 64-dimension model to a 256-dimension one,
   into the column `embedding_v2`.
4. **`vecshift plan`** shows the SQL it would run and estimates tokens, cost, and storage.
5. **`vecshift apply`** adds the column and the sync trigger, embeds every row, and builds
   the new index concurrently.
6. **`vecshift eval`** compares search quality per language and latency, gives a verdict,
   and writes `demo/out/eval.html`. If the verdict isn't GO, the demo stops here.
7. **`vecshift cutover`** renames the columns in one transaction, so `embedding` now holds
   the new vectors.
8. **`vecshift status`** confirms the stage and says what to do next.

Afterwards you can switch back, or run any other command, against the same database:

```bash
docker compose run --rm --entrypoint vecshift vecshift rollback --yes
docker compose run --rm --entrypoint vecshift vecshift eval --yes
```

Remove everything with `docker compose down --volumes`.

## Offline, with a test model

Both models are `hash/64` and `hash/256`, vecshift's built-in hashing baseline. It runs
locally, costs nothing, and needs no API key, so the demo sends nothing anywhere. It
matches words rather than meaning, which is why `plan` warns that it's a test model. Use a
real embedding model for your own data; the commands and the job file are the same.

## If the reports don't appear

The container runs as user 1000 and writes reports to `demo/out`. If your user has a
different ID, the demo says so and keeps the reports in the container. Make the folder
writable and run it again:

```bash
chmod a+w out
```

## Running the image on your own data

The same image runs any vecshift command. Put the connection string in an environment
variable, and mount a folder for the job file and its progress:

```bash
export VECSHIFT_DSN='postgresql://...'
docker run --rm -e VECSHIFT_DSN ghcr.io/osamamu64/vecshift doctor --table public.documents
docker run --rm -it -e VECSHIFT_DSN -e OPENAI_API_KEY \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work -e HOME=/work \
  ghcr.io/osamamu64/vecshift plan
```

`--user` and `HOME` let vecshift write the job file, reports, its progress, and its
embedding cache into your folder. `apply` keeps that progress in `.vecshift/` next to the job file, so keep using the same
folder to resume. Pass API keys with `-e NAME`, which copies them from your environment,
never as values on the command line.
