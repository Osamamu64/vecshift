import json

import psycopg
from typer.testing import CliRunner

from vecshift.cli import app


def test_bench_samples_from_pgvector(db_dsn: str, setup: psycopg.Connection) -> None:
    setup.execute(
        "CREATE TABLE public.chunks (id bigint PRIMARY KEY, content text, "
        "embedding extensions.vector(2))"
    )
    # Each row gets its own vocabulary, reused across its sentences, so a sentence taken
    # out as a query still has words in common with its own row and no other.
    setup.execute(
        """
        INSERT INTO public.chunks
        SELECT g, format(
            'Note %s %s %s %s %s %s. Then %s %s %s %s %s %s. '
            'Also %s %s %s %s %s %s.',
            w[1], w[2], w[3], w[4], w[5], w[6], w[1], w[7], w[3], w[8], w[5], w[9],
            w[2], w[10], w[4], w[11], w[6], w[12]), '[1,0]'
        FROM generate_series(1, 300) g,
        LATERAL (SELECT array_agg(substr(md5(g || '-' || i), 1, 8)) AS w
                 FROM generate_series(1, 12) i) words
        """
    )
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--table",
            "chunks",
            "--sample",
            "100",
            "--num-queries",
            "20",
            "-m",
            "hash/256",
            "--json",
            "--no-cache",
        ],
        env={"VECSHIFT_DSN": db_dsn},
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["documents"] == 100 and data["source"] == "public.chunks"
    assert data["models"][0]["scores"]["recall_at_10"] > 0
