import json
import tarfile
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from PIL import Image

from neko_data.build import BuildJob, LocalImagesSource, build_dataset
from neko_data.runtime.reader import iter_shard_samples


@pytest.fixture
def sample_png_bytes(sample_image_bytes: bytes) -> bytes:
    output = BytesIO()
    with Image.open(BytesIO(sample_image_bytes)) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def test_local_images_preserve_bytes_paths_and_order(tmp_path: Path, sample_png_bytes: bytes):
    root = tmp_path / "images"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "b/001.PNG").write_bytes(sample_png_bytes)
    (root / "a/001.png").write_bytes(sample_png_bytes)
    (root / "notes.txt").write_text("not an image")
    source = LocalImagesSource(root, split="validation", source_id="fixture")
    samples = list(source)
    assert [sample.sample_key for sample in samples] == ["a/001", "b/001"]
    assert all(sample.image == sample_png_bytes for sample in samples)
    assert all(sample.image_size == (32, 16) for sample in samples)
    assert all(sample.split == "validation" and sample.caption is None for sample in samples)
    assert samples[1].metadata["relative_path"] == "b/001.PNG"
    assert [sample.sample_key for sample in source] == ["a/001", "b/001"]
    output = tmp_path / "output"
    manifest = build_dataset(BuildJob(source=source, output_dir=output, dataset_id="fixture", version="v1",
                                     split="validation", skip_invalid_samples=False))
    restored = list(iter_shard_samples(output / manifest.shards[0].path,
                                      split="validation", source_id="fixture"))
    assert [sample.sample_key for sample in restored] == ["a/001", "b/001"]
    assert all(sample.image == sample_png_bytes for sample in restored)


def test_local_images_build_yaml_and_resume(tmp_path: Path, sample_png_bytes: bytes):
    root = tmp_path / "images"
    root.mkdir()
    for index in range(3):
        (root / f"{index}.png").write_bytes(sample_png_bytes)
    output = tmp_path / "output"
    config = tmp_path / "build.yaml"
    config.write_text(
        f"dataset:\n  dataset_id: fixture\n  version: v1\n  split: validation\n"
        f"  output_dir: {output}\nsource:\n  type: local_images\n  root: {root}\n"
        "build:\n  max_samples_per_shard: 2\n  resume: true\n  skip_invalid_samples: false\n"
    )
    manifest = build_dataset(BuildJob.from_yaml(config))
    assert manifest.splits == {"validation": 3}
    assert len(manifest.shards) == 2
    shard = manifest.shards[0]
    with tarfile.open(output / shard.path) as archive:
        assert archive.getnames() == ["0.png", "0.json", "1.png", "1.json"]
        assert archive.extractfile("0.png").read() == sample_png_bytes
        metadata = json.load(archive.extractfile("0.json"))
        assert metadata["relative_path"] == "0.png"
        assert metadata["width"] == 32
    assert pq.read_table(output / shard.metadata_path).num_rows == 2
    resumed = build_dataset(BuildJob.from_yaml(config))
    assert resumed.shards == manifest.shards
    assert resumed.metadata["resumed_samples"] == 3


def test_local_images_reject_duplicate_keys_before_yield(tmp_path: Path, sample_image_bytes: bytes):
    (tmp_path / "one.png").write_bytes(sample_image_bytes)
    (tmp_path / "one.jpg").write_bytes(sample_image_bytes)
    with pytest.raises(ValueError, match="Duplicate local image sample key"):
        next(iter(LocalImagesSource(tmp_path)))


def test_local_images_reject_empty_missing_and_invalid(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        LocalImagesSource(tmp_path / "missing")
    with pytest.raises(ValueError, match="No images found"):
        list(LocalImagesSource(tmp_path))
    (tmp_path / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read image dimensions"):
        list(LocalImagesSource(tmp_path))
