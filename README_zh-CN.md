# neko-data

`neko-data` 是一个独立的数据层，用于把大规模图文数据稳定地接入扩散模型训练：

```text
HF streaming / Hub WebDataset tar / DuckDB
    -> 统一样本
    -> WebDataset tar + 独立 Parquet metadata
    -> 本地或 S3 兼容对象存储（Cloudflare R2）
    -> manifest 分片规划 + 节点共享缓存
    -> RainbowNeko / HCP-Diffusion
```

核心约束是本地磁盘只有约 500G。构建时 Hub 上的 tar 使用非 seekable tar 流读取，普通
HuggingFace 数据集使用 `streaming=True`，不会把原始数据集全量下载到本地。训练时只缓存
当前节点和当前 rank 需要的 shard。

## 安装

```bash
pip install -e '.[hf,r2,duckdb,rainbow]'
```

`hf`、`r2`、`rainbow` 都是可选依赖。本地 tar 读取和 Parquet metadata 不需要 boto3，也不
要求训练环境安装 `webdataset` 包。
公开 HTTP manifest 和 shard 会自动使用 `HTTPStorage`；安装 `hf` extra 即可提供它所需的
`requests` 依赖。

## 构建

```bash
neko-data build --config examples/build/hf_webdataset.yaml
```

把 `publish.destination` 设置为 `s3://bucket/prefix` 即可发布到 R2。R2 endpoint 和凭证
使用 boto3 的环境变量或 profile 配置，不把凭证写进代码和 manifest。`delete_after_upload: true`
会在每个 shard 上传成功后删除构建机上的 tar，适合 500G 磁盘。

## 原样镜像已有 WebDataset

已有合适 tar shard 的 Hub 数据集使用 `neko-data mirror --config mirror.yaml`，无需逐样本
解码或重打包。命令默认只做只读规划，添加 `--execute --resume` 后执行可恢复上传。详细配置
和 DINO 图像训练接入见 [`docs/zh_CN/mirroring.md`](docs/zh_CN/mirroring.md)。

## 统一数据格式

每条样本只保存一份图片：

```text
sample-key.webp
sample-key.json       # 完整字段：caption、tags、regular_summary、尺寸、来源、id 等
sample-key.txt        # 可选，默认 caption，兼容旧 WebDataset reader
```

每个 tar 同时生成一个 Parquet sidecar。Parquet 只存一行样本 metadata，不重复存图片，适合
在不下载图片的情况下做去重、标签统计、ID 检查和质量分析。DuckDB 只在构建阶段按 sample
key join；训练时不会逐样本访问外部 DuckDB 或 R2 metadata object。

## RainbowNeko 接入

```python
from neko_data.integrations.rainbow import RainbowTextImageSource
from neko_data.runtime import RuntimeContext

source = RainbowTextImageSource.from_manifest(
    "s3://bucket/datasets/example/v1/manifest.json",
    runtime=RuntimeContext(global_rank=rank, world_size=world_size),
    cache_root="/local/nvme/neko-data-cache",
    cache_strategy="required",
    sample_shuffle=512,
    caption_key="caption",
    prompt_template="{caption}",
)
```

返回值保持现有训练 handler 需要的结构：

```python
{
    "id": sample_key,
    "image": image_bytes,
    "prompt": {"template": "{caption}", "caption": caption},
    "metadata": {...},
}
```

详细的 500G 磁盘规划、R2 布局、缓存策略和失败恢复见
[`docs/zh_CN/architecture.md`](docs/zh_CN/architecture.md)。

带标签（类别/角色）数据集、分组分片、多 split 构建和 `RainbowLabeledImageSource`，
见 [docs/zh_CN/labeled.md](docs/zh_CN/labeled.md)。

