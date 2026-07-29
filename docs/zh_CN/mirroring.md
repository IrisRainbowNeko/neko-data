# 原样镜像已有 WebDataset

当 Hub 仓库已经提供大小合适的 tar shard 时，使用 `neko-data mirror` 原样发布，避免逐样本
解码和重打包。镜像任务固定源 revision，并使用同名 JSON sidecar 提供样本数、tar 大小和
SHA-256。最终 manifest 只有在全部 shard、sidecar 和辅助 metadata 完成校验后才会发布。

## 配置

```yaml
dataset:
  dataset_id: example-prebuilt
  version: v1
  split: train

source:
  type: hf_prebuilt_webdataset
  repo_id: owner/dataset
  repo_type: dataset
  endpoint: https://huggingface.co
  revision: main
  include: ["images/**/*.tar"]
  sidecar_suffix: .json
  auxiliary_files:
    README.md: source/README.md
    records.parquet: metadata/records.parquet

publish:
  destination: s3://bucket/datasets/example-prebuilt/v1
  endpoint: https://<account>.r2.cloudflarestorage.com
  region: auto
  enable_multipart: false
  upload_concurrency: 1

mirror:
  output_dir: /data/mirror/example-prebuilt/v1
  staging_dir: /data/mirror/example-prebuilt/v1/staging
  workers: 8
  request_timeout: 120
  max_retries: 8
  retry_backoff: 5
```

凭据仅通过 `KHUB_TOKEN`/`HF_TOKEN`、`AWS_ACCESS_KEY_ID`、
`AWS_SECRET_ACCESS_KEY` 和 `R2_ENDPOINT` 提供。先执行只读检查：

```bash
neko-data mirror --config mirror.yaml
neko-data mirror --config mirror.yaml --status
```

确认后启动或恢复：

```bash
neko-data mirror --config mirror.yaml --workers 8 --execute --resume
```

每个 worker 最多暂存一个源文件，`.part` 文件支持 HTTP Range 续传。每次上传后通过 R2
HEAD 校验大小和对象 metadata 中的 SHA-256；已完成对象不会重复下载。变更源 revision 或
筛选条件时使用新的数据集版本，不要把不同输入恢复到同一个 manifest。

## RainbowNeko 图像训练

`RainbowWebDatasetImageSource` 与原 `WebDatasetImageSource` 一样返回 PIL Image，但使用
manifest 做 epoch、node、rank 和 DataLoader worker 分片，并通过节点共享缓存读取 R2：

```python
from neko_data.integrations.rainbow import RainbowWebDatasetImageSource
from neko_data.runtime import RuntimeContext

source = RainbowWebDatasetImageSource.from_manifest(
    "s3://bucket/datasets/example-prebuilt/v1/manifest.json",
    runtime=RuntimeContext.from_env(),
    cache_root="/local/nvme/neko-data-cache/example-prebuilt",
    cache_max_size_bytes=420 * 1024**3,
    cache_evict_size_bytes=350 * 1024**3,
    cache_strategy="required",
    prefetch_workers=4,
    prefetch_shards=8,
    sample_shuffle=300,
)
```
