# Changelog

## 0.2.0 - 2026-07-27

- Add resumable, checksum-verified mirroring for prebuilt Hub WebDataset shards.
- Add R2 object metadata validation and atomic final manifest publication.
- Add automatic torchrun/SLURM runtime discovery and a Rainbow image-only source.

## 0.1.1 - 2026-07-16

- Add public HTTP manifest and shard storage.
- Add resumable build error policy and HF relative image path support.
- Protect active cache leases during high-water eviction and keep prefetching
  after completed futures.
- Expand storage, planner, cache, and build error regression coverage.

## 0.1.0 - 2026-07-16

- Add normalized image-text sample and manifest contracts.
- Add streaming HuggingFace WebDataset and `datasets` sources.
- Add DuckDB build-time metadata join.
- Add WebDataset tar and Parquet sidecar output.
- Add local/S3-compatible publishing and node-local shard cache.
- Add deterministic shard planning, prefetch, and RainbowNeko adapter.

