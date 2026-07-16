# 架构与 500G 磁盘规划

## 推荐闭环

```text
HF datasets streaming / Hub tar
        |
        v
  统一 NormalizedSample
        |                DuckDB lookup（构建阶段）
        v                         |
  shard writer <------------------+
        |
        +--> wds/train/*.tar       图片 + JSON + 可选 TXT
        +--> metadata/train/*.parquet
        +--> manifest.json + indexes/train/shards.jsonl
        |
        v
  R2 dataset prefix
        |
        v
  manifest -> epoch shuffle -> node/rank/worker split
        |
        v
  节点共享 DiskShardCache -> RainbowNeko/HCP handler
```

## 500G 机器的实际策略

构建机和训练机需要区分考虑：

- 构建端使用 `HFWebDatasetSource` 的 HTTP tar 流，或 `HFDatasetsSource` 的
  `load_dataset(streaming=True)`。一个输入 tar 内存中只保留当前样本的成员。
- `WebDatasetShardWriter` 默认以 1 GiB 为目标分片大小。上传回调在 tar 和 Parquet
  都完成后触发，上传成功即可删除本地 tar。因此构建阶段不需要保留整个数据集。
- 训练端建议把缓存上限设置为 420 GiB、淘汰水位设置为 350 GiB，给系统、模型、日志和
  临时文件留下余量。缓存对象按照 manifest 的 SHA-256 命名，多个进程用同一个 lock。
- 同一节点的 rank 可以共享缓存目录；每个使用中的 shard 有 lease，淘汰器不会删除它。
- 每个 epoch 重新打乱 shard 顺序，再进行 node/rank/worker 切分。不同节点不会因为各自
  生成随机顺序而重复读取同一个 shard。
- 正式训练默认 `cache_strategy="required"`。网络短暂不稳定但允许降级时使用
  `preferred`；`disabled` 只适合调试或能接受每次直接 GET 的场景。

## 分片大小和请求模型

1 GiB 不是硬编码要求。100 MiB 适合低延迟实验，512 MiB 到 2 GiB 适合 R2 训练。分片
太小会增加 GET、锁和 manifest 开销；分片太大则首个 batch 等待时间长，且并发度不足。
分片数应至少明显多于一个训练节点的 rank 数，这样才能均匀分配。

不要使用 `rclone mount` 作为正式训练文件系统：挂载层把对象存储的延迟和一致性问题
暴露给每个小文件操作。直接从 R2 读 tar 可以作为 `preferred` fallback，本地 shard
cache 才是默认快路径。

## 为什么同时有 JSON 和 Parquet

- tar JSON 跟随图片传输，训练 sample 不需要额外 metadata GET。
- Parquet 是独立分析副本，按 shard 与 tar 一一对应；分析 id、标签、尺寸、来源时只拉
  Parquet，不下载图片。
- 构建阶段外部 DuckDB 的一行会合并到 tar JSON，并将 caption 字段展开成 Parquet 列。
  训练阶段不打开 DuckDB，避免每条样本一次远程或共享文件查询。

## 后续扩展

latent、text embedding 可以作为新的 tar member（例如 `.latent.pt`、`.text_emb.npz`）
或新的 dataset version；v0.1 保留原图和完整 JSON，避免在尚未固定 VAE/文本编码器时
丢失训练灵活性。

