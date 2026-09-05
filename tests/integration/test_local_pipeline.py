import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract.records import NormalizedSample
from neko_data.integrations.rainbow import RainbowTextImageSource
from neko_data.runtime import RuntimeContext, open_dataset
from neko_data.runtime.manifest_loader import ManifestLoader


def test_build_and_read_local_dataset(tmp_path: Path, sample_image_bytes: bytes):
    samples = [
        NormalizedSample(
            sample_key=f"sample-{index}",
            image=sample_image_bytes,
            source_id="fixture",
            caption=f"caption {index}",
            captions={"tags": f"tag_{index}"},
            metadata={"row_id": index},
            width=32,
            height=16,
        )
        for index in range(5)
    ]
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob(
        source=samples,
        output_dir=root,
        dataset_id="fixture",
        version="v1",
        max_shard_size="1MiB",
        max_samples_per_shard=2,
    ))
    assert manifest.splits == {"train": 5}
    assert len(manifest.shards) == 3
    view = open_dataset(root / "manifest.json", RuntimeContext(), cache_root=tmp_path / "cache",
                        prompt_template="{caption}", sample_shuffle=0)
    rows = list(view)
    assert {row["id"] for row in rows} == {f"sample-{index}" for index in range(5)}
    assert {row["prompt"]["caption"] for row in rows} == {f"caption {index}" for index in range(5)}
    assert rows[0]["metadata"]["width"] == 32
    assert view.get_image_size(rows[0]) == (32, 16)
    assert pq.read_table(root / "metadata/train/train-000000.parquet").num_rows == 2


def test_rainbow_adapter_returns_training_source(
    tmp_path: Path, sample_image_bytes: bytes
):
    root = tmp_path / "dataset"
    build_dataset(
        BuildJob(
            source=[
                NormalizedSample(
                    "sample",
                    sample_image_bytes,
                    caption="a caption",
                    width=32,
                    height=16,
                )
            ],
            output_dir=root,
            dataset_id="fixture",
            version="v1",
        )
    )
    source = RainbowTextImageSource.from_manifest(
        root / "manifest.json",
        cache_root=tmp_path / "cache",
        sample_shuffle=0,
    )
    row = next(iter(source))
    assert row["id"] == "sample"
    assert row["image"] == sample_image_bytes
    assert row["prompt"]["caption"] == "a caption"


def test_manifest_loader_can_drop_ambiguous_legacy_sidecars(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "dataset_id": "legacy",
            "version": "v1",
            "shards": [
                {
                    "path": "wds/a.tar",
                    "metadata_path": "metadata/train/train-000000.parquet",
                    "metadata_sha256": "a",
                    "split": "train",
                    "num_samples": 1,
                    "size_bytes": 1,
                    "sha256": "a",
                },
                {
                    "path": "wds/b.tar",
                    "metadata_path": "metadata/train/train-000000.parquet",
                    "metadata_sha256": "b",
                    "split": "train",
                    "num_samples": 1,
                    "size_bytes": 1,
                    "sha256": "b",
                },
            ],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate metadata_path"):
        ManifestLoader().load(manifest_path)
    loaded = ManifestLoader().load(manifest_path, allow_duplicate_metadata_paths=True)
    assert all(shard.metadata_path is None for shard in loaded.manifest.shards)
