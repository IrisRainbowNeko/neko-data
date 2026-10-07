# neko-data Documentation

`neko-data` converts HuggingFace/KHub sources into verifiable, resumable
WebDataset data for R2-backed diffusion training and exposes a unified
RainbowNeko reader.

## Documents

- [Architecture and the 500 GB disk plan](architecture.md)
- [Unified format and field contract](format.md)
- [Build, publish, and train operations](operations.md)
- [Labeled image datasets and grouped shards](labeled.md)

## Principles

1. Build from Hub sources as a stream; never require a full local download.
2. Use a shard as the network and cache unit, not an individual image.
3. Store one image copy in tar and put captions/metadata in tar JSON plus Parquet.
4. Join DuckDB only during build; training reads finalized shards only.
5. Treat the manifest as the source of truth for epoch/node/rank/worker planning.

