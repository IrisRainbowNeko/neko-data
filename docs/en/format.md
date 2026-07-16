# Unified Format and Field Contract

## Layout

```text
datasets/<dataset-id>/<version>/
  manifest.json
  indexes/train/shards.jsonl
  wds/train/train-000000.tar
  metadata/train/train-000000.parquet
  reports/build-summary.json
  reports/errors.jsonl
```

Each manifest shard contains at least `path`, `split`, `num_samples`,
`size_bytes`, and `sha256`; the Parquet path and checksum are included when
metadata output is enabled.

Each tar sample contains one image, one complete JSON metadata member, and an
optional `.txt` member containing the default caption. Stable JSON fields are
`sample_key`, `caption`, `captions`, `width`, `height`, `source_id`, `split`,
and `image_sha256`. Caption variants such as `tags`, `regular_summary`,
`prompt`, and `annotation` remain selectable by name.

The Parquet sidecar has one row per sample with stable fields including
`sample_key`, `source_id`, `split`, `shard_path`, dimensions, image hash,
caption, and `metadata_json`. Named caption variants are expanded as
`caption_<name>` columns. No image bytes are duplicated in Parquet.

