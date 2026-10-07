# Prior art

VecShift builds on ideas from other tools. This page notes what each does and how VecShift
relates to it, so you can pick the right tool for your job.

## Native reindex paths

Some stores can re-embed in place on the server.

- **Elasticsearch** can reindex into a new index through an ingest pipeline with an
  inference processor, run asynchronously as a task. On recent versions a `semantic_text`
  field's inference endpoint can be changed when the new model is compatible with the old one.
- **OpenSearch** supports the same pattern with a `text_embedding` ingest processor and
  `_reindex`.

If your model runs inside the cluster, that path avoids moving data over the network.
VecShift's value there is in what surrounds the reindex: diagnosing the index, evaluating
the new model, and cutting over safely.

## Copy tools

- **[Qdrant Migration Tool](https://github.com/qdrant/migration)** copies vectors and
  payloads into Qdrant from many sources, with resumable batched transfers. It's a strong
  reference for how to scan each source database. It copies vectors as they are, without
  re-embedding. If you only need a copy into Qdrant, use it.
- **Zilliz Vector Transport Service**, built on Apache SeaTunnel, moves vector data into
  Milvus and Zilliz Cloud.

## Model migration tools

- **[vecbee](https://pypi.org/project/vecbee/)** describes itself as "Alembic for vector
  databases": re-embed, re-chunk, or copy with a blue-green cutover.
- **[EmbedFlow](https://pypi.org/project/embedflow/)** migrates progressively: the new model
  fills in vectors over time as traffic arrives.
- **[isotrieve](https://pypi.org/project/isotrieve/)** learns a linear map from the old
  embedding space to the new one from a small calibration set, so the corpus doesn't need
  re-embedding.

## What VecShift focuses on

- Diagnosing an existing index before anything changes
- Benchmarking models on your own data
- Tracking which vector space every vector belongs to
- Store-agnostic migration with safe cutover and rollback
