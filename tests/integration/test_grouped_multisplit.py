import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import load_manifest
from neko_data.contract.records import NormalizedSample
from neko_data.integrations.rainbow import RainbowLabeledImageSource
from neko_data.runtime import RuntimeContext, open_dataset

GROUP_SIZES = [3, 1, 5, 2, 4, 3]


def _samples(image: bytes, split: str = "train"):
    for label, size in enumerate(GROUP_SIZES):
        for index in range(size):
            yield NormalizedSample(
                sample_key=f"{label:03d}-{index:03d}", image=image, split=split, source_id="fixture",
                metadata={"label": label, "name": f"char{label}"}, width=32, height=16,
            )


def _build(root: Path, image: bytes, split: str = "train", **kwargs):
    options = {"max_shard_size": "1MiB", "max_samples_per_shard": 4, "group_key": "label",
               "metadata": {"num_classes": len(GROUP_SIZES)}}
    options.update(kwargs)
    return build_dataset(BuildJob(source=_samples(image, split), output_dir=root, dataset_id="grouped",
                                  version="v1", split=split, **options))


def _shard_labels(root: Path, manifest) -> list[set[int]]:
    result = []
    for shard in manifest.shards:
        with tarfile.open(root / shard.path) as archive:
            labels = {json.load(archive.extractfile(member))["label"]
                      for member in archive.getmembers() if member.name.endswith(".json")}
        result.append(labels)
    return result


def test_group_key_never_splits_a_group_across_shards(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    manifest = _build(root, sample_image_bytes)
    shard_labels = _shard_labels(root, manifest)
    seen: set[int] = set()
    for labels in shard_labels:
        assert not labels & seen
        seen |= labels
    assert seen == set(range(len(GROUP_SIZES)))
    # The 5-sample group exceeds max_samples_per_shard but stays in one shard.
    assert max(shard.num_samples for shard in manifest.shards) > 4
    assert manifest.splits == {"train": sum(GROUP_SIZES)}


def test_multiple_splits_share_one_manifest(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    _build(root, sample_image_bytes, "train")
    _build(root, sample_image_bytes, "val")
    manifest = _build(root, sample_image_bytes, "test")
    assert manifest.splits == {split: sum(GROUP_SIZES) for split in ("train", "val", "test")}
    assert load_manifest(root / "manifest.json").splits == manifest.splits
    for split in ("train", "val", "test"):
        assert (root / "reports" / f"build-journal-{split}.jsonl").exists()
        index = (root / "indexes" / split / "shards.jsonl").read_text(encoding="utf-8").splitlines()
        assert {json.loads(line)["split"] for line in index} == {split}
    assert set(manifest.metadata["split_stats"]) == {"train", "val", "test"}
    assert manifest.metadata["num_classes"] == len(GROUP_SIZES)

    # Rebuilding one split replaces only that split.
    manifest = _build(root, sample_image_bytes, "val", overwrite=True)
    assert manifest.splits == {split: sum(GROUP_SIZES) for split in ("train", "val", "test")}


def test_group_shuffle_emits_groups_contiguously(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    _build(root, sample_image_bytes)
    orders = []
    for epoch in range(3):
        dataset = open_dataset(root / "manifest.json", RuntimeContext(), cache_root=tmp_path / "cache",
                               prefetch_workers=0, output_mode="image_label", group_key="label",
                               group_shuffle=4)
        dataset.set_epoch(epoch)
        items = list(dataset)
        labels = [item["label"] for item in items]
        assert sorted(labels) == sorted(label for label, size in enumerate(GROUP_SIZES) for _ in range(size))
        runs = [label for index, label in enumerate(labels) if index == 0 or labels[index - 1] != label]
        assert len(runs) == len(GROUP_SIZES)
        assert all(isinstance(item["image"], bytes) for item in items)
        orders.append(runs)
    assert len({tuple(order) for order in orders}) > 1


def test_group_chunk_size_splits_large_groups(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    _build(root, sample_image_bytes)
    dataset = open_dataset(root / "manifest.json", RuntimeContext(), cache_root=tmp_path / "cache",
                           prefetch_workers=0, output_mode="image_label", group_key="label",
                           group_shuffle=0, group_chunk_size=2)
    labels = [item["label"] for item in dataset]
    runs = [label for index, label in enumerate(labels) if index == 0 or labels[index - 1] != label]
    assert len(runs) == len(GROUP_SIZES)  # unshuffled chunks remain adjacent
    assert len(labels) == sum(GROUP_SIZES)


def test_group_key_rejects_sample_shuffle(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    _build(root, sample_image_bytes)
    with pytest.raises(ValueError):
        open_dataset(root / "manifest.json", RuntimeContext(), cache_root=tmp_path / "cache",
                     group_key="label", sample_shuffle=16)


def test_rainbow_labeled_image_source(tmp_path, sample_image_bytes):
    root = tmp_path / "dataset"
    _build(root, sample_image_bytes)
    _build(root, sample_image_bytes, "val")
    source = RainbowLabeledImageSource.from_manifest(
        root / "manifest.json", runtime=RuntimeContext(), split="val",
        cache_root=tmp_path / "cache", prefetch_workers=0,
    )
    items = list(source)
    assert len(source) == len(items) == sum(GROUP_SIZES)
    assert set(items[0]) == {"id", "image", "label"}
    assert isinstance(items[0]["image"], Image.Image)
    assert source.get_image_size(items[0]) == (32, 16)
    assert source.num_classes == len(GROUP_SIZES)


def test_python_factory_source(tmp_path, sample_image_bytes, monkeypatch):
    import sys
    import types

    module = types.ModuleType("fixture_factory")
    module.make = lambda count: [
        NormalizedSample(sample_key=f"k{i}", image=sample_image_bytes, metadata={"label": i})
        for i in range(count)
    ]
    monkeypatch.setitem(sys.modules, "fixture_factory", module)
    config = tmp_path / "job.yaml"
    config.write_text(
        "dataset:\n  dataset_id: py\n  version: v1\n  output_dir: " + str(tmp_path / "out") + "\n"
        "source:\n  type: python\n  factory: fixture_factory:make\n  kwargs:\n    count: 3\n"
        "build:\n  group_key: label\n",
        encoding="utf-8",
    )
    job = BuildJob.from_yaml(config)
    assert job.group_key == "label"
    assert build_dataset(job).splits == {"train": 3}
