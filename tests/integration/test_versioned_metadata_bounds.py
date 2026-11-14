import io
import json
import tarfile

import pytest

from neko_data.build import sample_io, source_hf_webdataset
from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import BlobSample, TypedSample, sample_metadata
from neko_data.runtime import blobs, open_dataset
from neko_data.runtime.reader import iter_shard_samples


@pytest.fixture
def metadata_cap(monkeypatch):
    cap = 1024
    for module in (sample_metadata, source_hf_webdataset, blobs):
        monkeypatch.setattr(module, "MAX_SAMPLE_METADATA_BYTES", cap)
    return cap


@pytest.mark.parametrize("mode", ["typed", "blob"])
def test_metadata_cap_matches_build_iteration_and_locate(tmp_path, metadata_cap, mode):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")

    def make_sample(note):
        metadata = {"note": note}
        return (TypedSample("safe", {"x": 1}, metadata=metadata) if mode == "typed"
                else BlobSample("safe", {"weights": source}, metadata=metadata))

    _, descriptor = sample_io.sample_members(make_sample(""))
    overhead = len(sample_metadata.encode_sample_metadata(descriptor))
    note = "x" * (metadata_cap - overhead)
    assert len(sample_metadata.encode_sample_metadata(sample_io.sample_members(make_sample(note))[1])) == metadata_cap
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob([make_sample(note)], root, "review", "v1", skip_invalid_samples=False))
    view = open_dataset(root / "manifest.json", output_mode=mode, prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    assert next(iter(view))["id"] == "safe"
    if mode == "blob":
        assert view.locate_blob("safe", manifest.shards[0])["id"] == "safe"
    with pytest.raises(ValueError, match="descriptor size"):
        build_dataset(BuildJob([make_sample(note + "x")], tmp_path / "too-large", "review", "v1",
                               skip_invalid_samples=False))


def _tar_with_json_first(note, *, blob=False):
    metadata = {"sample_format": "blob-files-v1", "sample_key": "safe", "note": note} if blob else {"note": note}
    payloads = [("safe.json", json.dumps(metadata).encode()),
                ("safe.nblob00000000" if blob else "safe.jpg", b"content")]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in payloads:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    stream.seek(0)
    return stream


def test_oversized_blob_json_is_bounded_before_reading(metadata_cap, monkeypatch):
    original = tarfile.ExFileObject.read
    reads = []

    def bounded_read(self, size=-1):
        reads.append(size)
        assert 0 <= size <= 8 * 1024**2, "versioned JSON must not use an unbounded read"
        return original(self, size)

    monkeypatch.setattr(tarfile.ExFileObject, "read", bounded_read)
    with pytest.raises(ValueError, match="descriptor size"):
        list(iter_shard_samples(_tar_with_json_first("x" * (metadata_cap + 1), blob=True),
                                split="train", source_id="fixture"))
    assert reads


def test_oversized_json_first_legacy_image_remains_compatible(metadata_cap):
    note = "x" * (metadata_cap + 1)
    samples = list(iter_shard_samples(_tar_with_json_first(note), split="train", source_id="fixture"))
    assert len(samples) == 1
    assert samples[0].metadata["note"] == note
