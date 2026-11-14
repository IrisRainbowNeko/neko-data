from pathlib import Path

import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import BlobSample, TypedSample
from neko_data.runtime import open_dataset


@pytest.mark.parametrize("key", ["", ".", "..", ".snapshot", "./snapshot", "../snapshot", "/snapshot",
                                 "a\\b", "C:/snapshot", "a/../b", "a/./b", "a//b", "a/", "a\x00b"])
def test_versioned_samples_reject_keys_changed_or_unsafe_in_tar(tmp_path, key):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    with pytest.raises(ValueError, match="canonical"):
        TypedSample(key, {"x": 1})
    with pytest.raises(ValueError, match="canonical"):
        BlobSample(key, {"weights": source})


@pytest.mark.parametrize("key", ["episode-17-segment-0", "checkpoint-760", "map.alpha/episode-001",
                                 "group/.hidden", "录像/第1局"])
@pytest.mark.parametrize("mode", ["typed", "blob"])
def test_versioned_keys_roundtrip_without_rewriting(tmp_path: Path, key, mode):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    sample = TypedSample(key, {"x": 1}) if mode == "typed" else BlobSample(key, {"weights": source})
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob([sample], root, "review", "v1", skip_invalid_samples=False))
    view = open_dataset(root / "manifest.json", output_mode=mode, prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    assert next(iter(view))["id"] == key
    if mode == "blob":
        row = view.locate_blob(key, manifest.shards[0])
        with view.acquire_blob(row, "weights") as path:
            assert path.read_bytes() == b"payload"
