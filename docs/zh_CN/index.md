# neko-data 文档

`neko-data` 把 HuggingFace/KHub 数据源转换为可验证、可恢复、适合 R2 训练的 WebDataset
数据集，并提供 RainbowNeko Engine 的统一读取接口。

## 文档

- [架构与 500G 磁盘规划](architecture.md)
- [统一格式与字段契约](format.md)
- [构建、发布和训练运行手册](operations.md)
- [带标签图像数据集与分组分片](labeled.md)

## 设计原则

1. 构建端流式处理，绝不要求下载完整 Hub 数据集。
2. 训练端以 shard 为网络和缓存单位，不以单张图片为对象。
3. 图片在 tar 中只保存一份；caption 和 metadata 同时进入 tar JSON 和独立 Parquet。
4. DuckDB 只在构建阶段 join，训练时只读已完成 shard。
5. manifest 是唯一的 shard 真相源，epoch、node、rank、worker 分配都从它派生。

