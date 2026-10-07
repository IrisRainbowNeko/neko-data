# Labeled Image Datasets and Grouped Shards

Identity/classification datasets (for example CCIP character datasets) have
many small classes. A sample-level shuffle buffer almost never yields two images
of the same class close together, which starves contrastive buckets such as
RainbowNeko's streaming `PosNegBucket` and `CategoryBucket`. neko-data solves
this with grouped shards and group-level shuffle.

## Build

Emit samples so that all members of a group are consecutive (shuffle the order
of groups yourself) and set `group_key` to the metadata field holding the group:

```python
build_dataset(BuildJob(
    source=samples("train"), output_dir=root, dataset_id="ccip", version="v1",
    split="train", group_key="label", max_shard_size="1GiB",
    metadata={"num_classes": 33000, "label_key": "label"},
))
```

With `group_key`, a shard only rolls over between groups, so a group never
straddles two shards (a single group larger than the limit produces one larger
shard). YAML jobs accept `build.group_key` and a `python` source type:

```yaml
source:
  type: python
  factory: my_package.convert:iter_samples
  kwargs: {split: train}
build:
  group_key: label
```

## Multiple splits

Run one job per split into the same `output_dir`. Each split has its own
`reports/build-journal-<split>.jsonl`, `reports/build-summary-<split>.json`
and `indexes/<split>/shards.jsonl`; `manifest.json` keeps the shards of the
other splits and records per-split counts in `metadata.split_stats`. Rebuilding
a split with `overwrite=True` replaces only that split. Lines in
`reports/errors.jsonl` carry a `split` field.

## Read

```python
from neko_data.integrations.rainbow import RainbowLabeledImageSource

source = RainbowLabeledImageSource.from_manifest(
    "/data/ccip/manifest.json", split="train",
    cache_root="/local/nvme/neko-data-cache",
    group_key="label", group_shuffle=2048, group_chunk_size=16,
)
# yields {"id": ..., "image": PIL.Image, "label": int}
```

`DatasetView(output_mode="image_label", label_key="label")` yields raw bytes
instead of PIL images. `group_shuffle` is the number of groups held in the
shuffle buffer; `group_chunk_size` splits large groups into independently
shuffled chunks; `shuffle_within_group` (default true) shuffles members of a
group. `sample_shuffle` cannot be combined with `group_key`.
`source.num_classes` reads `metadata.num_classes` from the manifest.
