# Changelog

## 0.3.0 - 2026-10-07

- Add `group_key` grouped builds: shards roll over only at group boundaries.
- Support several splits in one dataset directory (per-split journals, summaries,
  and indexes; the manifest keeps other splits).
- Add group-level runtime shuffle (`group_key`, `group_shuffle`, `group_chunk_size`),
  `output_mode="image_label"`, and `RainbowLabeledImageSource` for CCIP-style training.
- Add the `python` YAML source type (`factory: module:function`).

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

