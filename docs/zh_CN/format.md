# 统一格式与字段契约

## 目录布局

```text
datasets/<dataset-id>/<version>/
  manifest.json
  indexes/train/shards.jsonl
  wds/train/train-000000.tar
  metadata/train/train-000000.parquet
  reports/build-summary.json
  reports/errors.jsonl
```

manifest 中的 shard 至少包含：

```json
{
  "path": "wds/train/train-000000.tar",
  "metadata_path": "metadata/train/train-000000.parquet",
  "split": "train",
  "num_samples": 100000,
  "size_bytes": 1073741824,
  "sha256": "..."
}
```

## Tar sample

```text
<sample-key>.<jpg|png|webp|...>
<sample-key>.json
<sample-key>.txt       # 有默认 caption 时生成
```

JSON 的稳定字段包括：

| 字段 | 含义 |
| --- | --- |
| `sample_key` | 统一样本 ID，不能依赖 `int(id)` |
| `caption` | 默认训练 caption |
| `captions` | 所有命名 caption，如 `tags`、`regular_summary`、`prompt`、`annotation` |
| `width`, `height` | 构建阶段记录的图片尺寸，用于 RatioBucket |
| `source_id`, `split` | 来源和数据集 split |
| `image_sha256` | 图片内容 hash |
| 其他字段 | DuckDB/HF metadata，保留在同一 JSON |

训练用 `caption_key="tags"`、`caption_key="regular_summary"` 等显式选择字段，
不会出现旧 reader “`.txt` 优先导致 JSON caption key 失效”的问题。

## Parquet schema

每行对应一个 tar sample，固定字段有 `sample_key`、`source_id`、`split`、
`shard_path`、`width`、`height`、`image_extension`、`image_sha256`、`caption`、
`metadata_json`。caption 变体展开为 `caption_<name>` 列，简单标量 metadata 也会保留
为独立列；复杂 metadata 保留在 `metadata_json`。

