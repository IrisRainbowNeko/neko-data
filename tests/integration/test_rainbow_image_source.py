import hashlib
import io
import tarfile

import numpy as np
from PIL import Image

from neko_data.contract import DatasetManifest, ShardRecord, write_manifest
from neko_data.integrations.rainbow import RainbowWebDatasetImageSource
from neko_data.runtime import RuntimeContext


def _add_member(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def test_rainbow_image_source_reads_image_and_ignores_npy(sample_image_bytes, tmp_path):
    root = tmp_path / "dataset"
    shard = root / "wds" / "train" / "images" / "0" / "001.tar"
    shard.parent.mkdir(parents=True)
    vector = io.BytesIO()
    np.save(vector, np.arange(8, dtype=np.float32), allow_pickle=False)
    with tarfile.open(shard, "w") as archive:
        _add_member(archive, "fixture.jpg", sample_image_bytes)
        _add_member(archive, "fixture.npy", vector.getvalue())
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = DatasetManifest(
        dataset_id="image-fixture",
        version="v1",
        shards=[ShardRecord(
            path="wds/train/images/0/001.tar",
            split="train",
            num_samples=1,
            size_bytes=shard.stat().st_size,
            sha256=digest,
        )],
    )
    manifest_path = write_manifest(root / "manifest.json", manifest)

    source = RainbowWebDatasetImageSource.from_manifest(
        manifest_path,
        runtime=RuntimeContext(),
        cache_root=tmp_path / "cache",
        prefetch_workers=0,
        sample_shuffle=0,
    )
    source.set_epoch(3)
    item = next(iter(source))

    assert set(item) == {"id", "image"}
    assert item["id"] == "fixture"
    assert isinstance(item["image"], Image.Image)
    assert item["image"].size == (32, 16)
    assert len(source) == 1
