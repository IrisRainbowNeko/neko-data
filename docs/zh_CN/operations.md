# 构建、发布和训练运行手册

## 构建前检查

```bash
pip install -e '.[hf,r2,duckdb]'
export HF_TOKEN='...'
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'
export R2_ENDPOINT='https://<account>.r2.cloudflarestorage.com'
```

不要把 token、secret 或完整带签名 URL 放入 YAML 和 Git。先用小范围 tar 和较小
`max_shard_size` 做 smoke test，再开始大规模构建。

## HF tar 流式构建

`HFWebDatasetSource` 会先列出符合 `include`/`exclude` 的 tar，再对每个 tar 发一个
streaming HTTP 请求，使用 `tarfile.open(..., mode="r|*")` 处理。Hub 输入 tar 不会存到
本地 cache；本地输出只保留正在写的目标 shard。

## 普通 HF 数据集

将 source 类型设置为 `hf_dataset`，并提供 `path`、`image_column`、`key_column` 和
`caption_columns`。实现依赖 `datasets.load_dataset(..., streaming=True)`；如果数据集
返回图片路径而非 bytes，只有当前样本的路径会被读取。

## DuckDB join

```yaml
metadata_provider:
  type: duckdb
  path: /data/labels/tags+captions.duckdb
  table: data
  index_key: id
```

字段通过 sample key 查询，`id` 不要求是整数；字段会写进 tar JSON，并将 caption 字段
写入 Parquet。构建完成后可以只处理 Parquet：

```python
import pyarrow.parquet as pq
table = pq.read_table("metadata/train/train-000000.parquet")
print(table.column_names)
```

## 训练缓存

缓存目录必须是节点上所有训练进程可访问的本地目录，例如 NVMe 上的
`/local/nvme/neko-data-cache`。每个节点只需要一份缓存；不要为每个 rank 配一个独立
目录，否则会重复占用磁盘和下载带宽。

训练恢复时可以从新 epoch 开始；manifest checksum 和 shard checksum 保证坏文件会被
重新下载。v0.1 的 cache lease 是文件级的，后续可替换为节点 daemon 而不改变 reader API。

## 构建恢复与错误样本

`build.resume: true` 只适用于输入流稳定且可以重放的情况。journal 记录已经完成的
shard；恢复时会重新遍历输入，并跳过 journal 已经包含的样本。输入顺序、过滤条件或
join 结果改变后不要继续 resume。

`skip_invalid_samples: true` 会把样本规范化阶段的 `TypeError`、`ValueError` 和
`UnicodeError` 写入 `reports/errors.jsonl` 并继续构建。发布器失败、存储错误和网络
I/O 错误仍然直接失败。要有意重建时使用 `overwrite: true`，不要将新旧输入混在同一
个 manifest 中。

## HTTP 读取

公开的 `http://` 和 `https://` manifest、shard 会自动使用 `HTTPStorage`；读取过程是
流式的。训练时默认 cache 仍会把当前分配的完整 tar 下载到节点共享目录，再从本地读取。
这适合公开镜像；私有 R2 仍应使用 `s3://`、`r2` extra 和环境变量中的凭证。

Parquet sidecar 与 tar 一起发布，可以只下载 metadata 做标签、ID、尺寸和重复项分析，
无需下载图片；训练阶段也不会逐样本访问 DuckDB。
