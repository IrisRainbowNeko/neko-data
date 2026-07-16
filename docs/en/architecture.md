# Architecture and the 500 GB Disk Plan

## Recommended flow

```text
HF streaming / Hub tar -> NormalizedSample -> tar + Parquet -> R2
    -> manifest -> epoch shuffle -> node/rank/worker split
    -> shared node DiskShardCache -> RainbowNeko/HCP handlers
```

The build side reads Hub WebDataset tars with a non-seekable tar stream or
uses `load_dataset(streaming=True)`. Only the current sample is retained in
memory. The writer closes a tar and its Parquet sidecar atomically, then a
publisher can upload both and delete local copies.

For a 500 GB training machine, use a cache limit around 420 GiB and an eviction
target around 350 GiB. The remaining space is reserved for the OS, model
weights, logs, temporary files, and checkpoints. Cache files are named by the
manifest SHA-256, protected by a cross-process lock, and leased while a tar is
being read.

Every epoch shuffles shards deterministically and then assigns them by node,
process, and DataLoader worker. `required` cache mode is the stable training
mode; `preferred` falls back to direct object-store streaming; `disabled` is
for debugging. Do not use an `rclone mount` as the production training
filesystem.

## Shard size

Use 512 MiB to 2 GiB for R2 training, with 1 GiB as the default. Smaller shards
increase GET and lock overhead; larger shards increase first-batch latency and
reduce scheduling granularity. Keep substantially more shards than ranks so
distributed placement remains balanced.

## Why JSON plus Parquet

The JSON travels with the image and avoids a metadata request per sample.
Parquet is a separate per-shard analysis copy, so label, ID, dimension, and
source analysis can run without downloading images. DuckDB rows are joined at
build time and are never queried from R2 during training.

