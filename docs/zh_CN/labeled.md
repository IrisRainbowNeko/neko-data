# 带标签图像数据集与分组分片

CCIP 这类角色/身份数据集类别很多、每类样本很少。样本级 shuffle buffer 几乎不会让同一类的
两张图靠在一起，RainbowNeko 流式模式下的 `PosNegBucket`、`CategoryBucket` 因此凑不出正样本。
neko-data 用"分组分片 + 组级打乱"解决这个问题。

## 构建

数据源按组连续输出样本（组的顺序由你自己随机打乱），并把 `group_key` 设为存放组号的 metadata 字段：

```python
build_dataset(BuildJob(
    source=samples("train"), output_dir=root, dataset_id="ccip", version="v1",
    split="train", group_key="label", max_shard_size="1GiB",
    metadata={"num_classes": 33000, "label_key": "label"},
))
```

设置 `group_key` 后，只在组与组之间切换分片，同一组绝不会跨两个分片；单个组超过上限时，
会生成一个偏大的分片。YAML 任务支持 `build.group_key`，以及 `python` 类型的数据源：

```yaml
source:
  type: python
  factory: my_package.convert:iter_samples
  kwargs: {split: train}
build:
  group_key: label
```

## 多 split

每个 split 跑一次任务，输出到同一个 `output_dir`。每个 split 各自有：

- `reports/build-journal-<split>.jsonl`
- `reports/build-summary-<split>.json`
- `indexes/<split>/shards.jsonl`

`manifest.json` 会保留其他 split 的分片，并在 `metadata.split_stats` 中记录每个 split 的统计。
用 `overwrite=True` 重建某个 split 时，只替换这个 split。`reports/errors.jsonl` 的每一行带 `split` 字段。

## 读取

```python
from neko_data.integrations.rainbow import RainbowLabeledImageSource

source = RainbowLabeledImageSource.from_manifest(
    "/data/ccip/manifest.json", split="train",
    cache_root="/local/nvme/neko-data-cache",
    group_key="label", group_shuffle=2048, group_chunk_size=16,
)
# 输出 {"id": ..., "image": PIL.Image, "label": int}
```

参数说明：

- `DatasetView(output_mode="image_label", label_key="label")` 输出原始字节，而不是 PIL 图像。
- `group_shuffle`：打乱缓冲区中容纳的组数。
- `group_chunk_size`：把大组拆成若干块，各块独立参与打乱。
- `shuffle_within_group`（默认开启）：打乱组内样本。
- `sample_shuffle` 不能和 `group_key` 同时使用。
- `source.num_classes`：读取 manifest 里的 `metadata.num_classes`。
