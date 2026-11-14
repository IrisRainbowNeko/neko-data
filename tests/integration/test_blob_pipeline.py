import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import BlobSample
from neko_data.contract.blobs import file_sha256
from neko_data.runtime import open_dataset
from neko_data.runtime.cache import DiskShardCache


def make_files(tmp_path, size=20 * 1024**2):
    model = tmp_path / "model.bin"
    with model.open("wb") as output:
        for _ in range(size // 1024):
            output.write(b"checkpoint!" * 93 + b"x")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"architecture": "synthetic"}), encoding="utf-8")
    return model, config


def test_blob_stream_write_read_pins_and_hf_directory(tmp_path, monkeypatch):
    model, config = make_files(tmp_path)
    expected = file_sha256(model)
    # Neither builder nor reader may turn an artifact into a whole bytes object.
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("Whole artifact read_bytes is forbidden"))
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob([BlobSample("checkpoint-17", {"model.bin": model, "config.json": config},
                                                  metadata={"policy_version": 17})],
                                     root, "models", "v1", skip_invalid_samples=False))
    assert len(manifest.shards) == 1
    view = open_dataset(root / "manifest.json", output_mode="blob", prefetch_workers=0,
                        cache_root=tmp_path / "cache", blob_cache_max_size_bytes=32 * 1024**2)
    row = next(iter(view))
    assert row["id"] == "checkpoint-17"
    assert row["files"]["model.bin"]["sha256"] == expected
    with view.acquire_blob(row, "model.bin") as path:
        assert file_sha256(path) == expected
        assert list(path.parent.glob(f".{path.name}.lease.*"))
    assert not list(path.parent.glob(f".{path.name}.lease.*"))
    with view.acquire_blobs(row) as directory:
        assert file_sha256(directory / "model.bin") == expected
        assert json.loads((directory / "config.json").read_text()) == {"architecture": "synthetic"}
    assert not directory.exists()


def test_remote_blob_uses_existing_shard_cache_and_hashes(tmp_path):
    model, _ = make_files(tmp_path, size=1024**2)
    root = tmp_path / "dataset"
    build_dataset(BuildJob([BlobSample("snapshot", {"weights": model})], root, "models", "v1"))
    calls = []

    class Remote:
        def download(self, uri, destination):
            calls.append(uri)
            shutil.copyfile(root / uri.removeprefix("s3://fixture/"), destination)

    cache = DiskShardCache(tmp_path / "shards", Remote(), max_size_bytes=2 * 1024**2)
    view = open_dataset(root / "manifest.json", cache=cache, output_mode="blob", prefetch_workers=0)
    view.loaded = replace(view.loaded, base_uri="s3://fixture")
    row = next(iter(view))
    with view.acquire_blob(row, "weights") as path:
        assert file_sha256(path) == file_sha256(model)
    with view.acquire_blob(row, "weights") as path:
        assert path.is_file()
    assert len(calls) == 1
    assert not list(cache.objects.glob("*.lease.*"))
    assert not list((view.blob_cache_root / "objects").glob("*.lease.*"))


def test_locate_blob_does_not_download_other_snapshots(tmp_path):
    model, _ = make_files(tmp_path, size=1024)
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob([BlobSample(f"snapshot-{i}", {"weights": model}) for i in range(3)],
                                     root, "models", "v1", max_samples_per_shard=1))
    calls = []

    class Remote:
        def download(self, uri, destination):
            calls.append(uri)
            shutil.copyfile(root / uri.removeprefix("s3://fixture/"), destination)

    cache = DiskShardCache(tmp_path / "cache", Remote())
    view = open_dataset(root / "manifest.json", output_mode="blob", cache=cache, prefetch_workers=0)
    view.loaded = replace(view.loaded, base_uri="s3://fixture")
    row = view.locate_blob("snapshot-2", manifest.shards[2].path)
    with view.acquire_blob(row, "weights") as path:
        assert file_sha256(path) == file_sha256(model)
    assert calls == ["s3://fixture/" + manifest.shards[2].path]
    with pytest.raises(ValueError, match="assigned"):
        view.locate_blob("snapshot-2", "s3://arbitrary/file.tar")


def test_blob_corruption_and_unsafe_fields_rejected(tmp_path):
    model, _ = make_files(tmp_path, size=1024)
    with pytest.raises(ValueError, match="Unsafe"):
        BlobSample("x", {"../model": model})
    root = tmp_path / "dataset"
    build_dataset(BuildJob([BlobSample("snapshot", {"weights": model})], root, "models", "v1"))
    view = open_dataset(root / "manifest.json", output_mode="blob", prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    row = next(iter(view))
    row["_blob_reference"].sample.files["weights"]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(IOError, match="checksum"):
        with view.acquire_blob(row, "weights"):
            pytest.fail("Corrupt artifact must never be handed to the caller")
    assert not list((view.blob_cache_root / "objects").glob("*.partial.*"))


def test_optional_torch_checkpoint_mmap(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"weights": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)}, checkpoint)
    root = tmp_path / "dataset"
    build_dataset(BuildJob([BlobSample("snapshot", {"checkpoint.pt": checkpoint})], root, "models", "v1"))
    view = open_dataset(root / "manifest.json", output_mode="blob", prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    row = next(iter(view))
    with view.acquire_blob(row, "checkpoint.pt") as path:
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        assert loaded["weights"].dtype == torch.bfloat16
        assert torch.equal(loaded["weights"], torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
        del loaded  # release mmap before the lease can be evicted (required on Windows)


def test_publisher_uploads_finalized_shards_and_deletes_staging(tmp_path):
    from neko_data.build.publisher import DatasetPublisher

    model, _ = make_files(tmp_path, size=1024)
    staging, published = tmp_path / "staging", tmp_path / "published"
    uploaded = []

    class Remote:
        def upload_file(self, source, target, **_kwargs):
            uploaded.append(target)
            destination = published / target.removeprefix("s3://fixture/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    manifest = build_dataset(BuildJob(
        [BlobSample(f"version-{i}", {"weights": model}) for i in range(3)],
        staging, "models", "v1", max_samples_per_shard=1, max_shard_size=512,
        skip_invalid_samples=False,
        publisher=DatasetPublisher("s3://fixture", storage=Remote(), delete_after_upload=True),
    ))
    assert len(manifest.shards) == 3  # each single artifact is allowed to exceed the 512-byte target
    assert not list((staging / "wds").rglob("*.tar"))
    assert not list((staging / "metadata").rglob("*.parquet"))
    assert len(uploaded) == 8  # three tar/Parquet pairs, index, then final manifest
    assert uploaded[-1] == "s3://fixture/manifest.json"
    view = open_dataset(published / "manifest.json", output_mode="blob", prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    row = view.locate_blob("version-2", manifest.shards[2])
    with view.acquire_blob(row, "weights") as path:
        assert file_sha256(path) == file_sha256(model)


def test_extracted_blob_cache_never_evicts_active_pin(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"a" * 1024**2)
    second.write_bytes(b"b" * 1024**2)
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob([BlobSample("first", {"weights": first}),
                                      BlobSample("second", {"weights": second})],
                                     root, "models", "v1", max_samples_per_shard=1))
    view = open_dataset(root / "manifest.json", output_mode="blob", cache_root=tmp_path / "cache",
                        prefetch_workers=0, blob_cache_max_size_bytes=1536 * 1024,
                        blob_cache_evict_size_bytes=1024**2)
    row_a = view.locate_blob("first", manifest.shards[0])
    row_b = view.locate_blob("second", manifest.shards[1])
    with view.acquire_blob(row_a, "weights") as pinned:
        with view.acquire_blob(row_b, "weights") as other:
            assert pinned.is_file() and other.is_file()
        assert pinned.is_file()
        assert not other.exists()
    assert not list(pinned.parent.glob("*.lease.*"))


def test_hardlink_mutation_cannot_poison_future_verified_acquire(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"GOOD-CHECKPOINT")
    root = tmp_path / "dataset"
    build_dataset(BuildJob([BlobSample("snapshot", {"model.bin": source})],
                           root, "models", "v1", skip_invalid_samples=False))
    view = open_dataset(root / "manifest.json", output_mode="blob", prefetch_workers=0,
                        cache_root=tmp_path / "cache")
    row = next(iter(view))
    with view.acquire_blobs(row) as directory:
        linked = directory / "model.bin"
        before = linked.stat()
        # API callers must treat links as read-only, but an accidental mutation
        # must not silently persist as a valid cache hit after the lease ends.
        linked.write_bytes(b"EVIL-CHECKPOINT")
        os.utime(linked, ns=(before.st_atime_ns, before.st_mtime_ns))
    with view.acquire_blob(row, "model.bin") as repaired:
        assert repaired.read_bytes() == b"GOOD-CHECKPOINT"
        assert file_sha256(repaired) == row["files"]["model.bin"]["sha256"]
