import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import neko_data.build.mirror as mirror_module
from neko_data.build.mirror import HubFile, MirrorJob, MirrorPlan


class MissingObject(Exception):
    def __init__(self):
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.upload_order = []

    def head(self, uri):
        if uri not in self.objects:
            raise MissingObject()
        return {
            "size_bytes": len(self.objects[uri]),
            "metadata": self.metadata[uri],
            "etag": "",
        }

    def upload_bytes(self, data, uri, *, metadata=None):
        self.objects[uri] = bytes(data)
        self.metadata[uri] = dict(metadata or {})
        self.upload_order.append(uri)

    def upload_file(self, source, uri, max_concurrency=8, *, metadata=None, **kwargs):
        self.objects[uri] = Path(source).read_bytes()
        self.metadata[uri] = dict(metadata or {})
        self.upload_order.append(uri)


def _job(tmp_path, *, resume=False):
    root = tmp_path / "mirror"
    return MirrorJob(
        dataset_id="fixture",
        version="v1",
        repo_id="owner/repo",
        destination="s3://bucket/datasets/fixture/v1",
        output_dir=root,
        staging_dir=root / "staging",
        workers=1,
        resume=resume,
        expected_shards=1,
        expected_samples=1,
        expected_tar_bytes=11,
    )


def _sidecar(tar_data):
    return json.dumps({
        "filesize": len(tar_data),
        "hash_lfs": hashlib.sha256(tar_data).hexdigest(),
        "files": {
            "sample.webp": {"size": 7},
            "sample.npy": {"size": 4},
        },
    }).encode()


def test_sidecar_record_counts_images_and_preserves_source_paths(tmp_path):
    tar_data = b"tar payload"
    source = HubFile("images/0/001.tar", len(tar_data), hashlib.sha256(tar_data).hexdigest())
    record = mirror_module._sidecar_record(
        _job(tmp_path),
        source,
        "images/0/001.json",
        _sidecar(tar_data),
        3,
    )

    assert record.path == "wds/train/images/0/001.tar"
    assert record.metadata_path == "metadata/shards/images/0/001.json"
    assert record.num_samples == 1
    assert record.index == 3


def test_sidecar_record_rejects_size_mismatch(tmp_path):
    tar_data = b"tar payload"
    source = HubFile("images/0/001.tar", len(tar_data) + 1)

    with pytest.raises(ValueError, match="Sidecar size mismatch"):
        mirror_module._sidecar_record(
            _job(tmp_path),
            source,
            "images/0/001.json",
            _sidecar(tar_data),
            0,
        )


def test_mirror_publishes_exact_bytes_last_manifest_and_resumes(monkeypatch, tmp_path):
    tar_data = b"tar payload"
    digest = hashlib.sha256(tar_data).hexdigest()
    sidecar = _sidecar(tar_data)
    plan = MirrorPlan("commit-sha", (HubFile("images/0/001.tar", len(tar_data), digest),), ())
    storage = MemoryStorage()

    monkeypatch.setattr(mirror_module, "inspect_mirror", lambda *args, **kwargs: plan)
    monkeypatch.setattr(mirror_module, "_storage", lambda job: storage)
    monkeypatch.setattr(mirror_module, "_request_bytes", lambda *args: sidecar)

    def download(job, revision, source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(tar_data)
        return destination, digest

    monkeypatch.setattr(mirror_module, "_download_hub_file", download)
    job = _job(tmp_path)
    manifest = mirror_module.mirror_dataset(job, copy_auxiliary=False)

    tar_uri = "s3://bucket/datasets/fixture/v1/wds/train/images/0/001.tar"
    assert storage.objects[tar_uri] == tar_data
    assert storage.metadata[tar_uri]["sha256"] == digest
    assert manifest.splits == {"train": 1}
    assert storage.upload_order[-1].endswith("/manifest.json")

    resumed = replace(job, resume=True)

    def should_not_download(*args, **kwargs):
        raise AssertionError("a verified shard must not be downloaded again")

    monkeypatch.setattr(mirror_module, "_download_hub_file", should_not_download)
    second = mirror_module.mirror_dataset(resumed, copy_auxiliary=False)
    assert second.splits == {"train": 1}

    for journal in (job.output_dir / "reports" / "workers").glob("worker-*.jsonl"):
        journal.unlink()
    recovered = mirror_module.mirror_dataset(resumed, copy_auxiliary=False)
    assert recovered.splits == {"train": 1}


def test_failed_shard_does_not_publish_manifest(monkeypatch, tmp_path):
    tar_data = b"tar payload"
    digest = hashlib.sha256(tar_data).hexdigest()
    plan = MirrorPlan("commit-sha", (HubFile("images/0/001.tar", len(tar_data), digest),), ())
    storage = MemoryStorage()

    monkeypatch.setattr(mirror_module, "inspect_mirror", lambda *args, **kwargs: plan)
    monkeypatch.setattr(mirror_module, "_storage", lambda job: storage)
    monkeypatch.setattr(mirror_module, "_request_bytes", lambda *args: _sidecar(tar_data))

    def fail_download(*args, **kwargs):
        raise OSError("source unavailable")

    monkeypatch.setattr(mirror_module, "_download_hub_file", fail_download)
    with pytest.raises(OSError, match="source unavailable"):
        mirror_module.mirror_dataset(_job(tmp_path), copy_auxiliary=False)

    assert not any(uri.endswith("/manifest.json") for uri in storage.objects)


def test_destination_revision_mismatch_is_rejected(monkeypatch, tmp_path):
    storage = MemoryStorage()
    uri = "s3://bucket/datasets/fixture/v1/manifest.json"
    storage.objects[uri] = b"{}"
    storage.metadata[uri] = {"source-revision": "old-revision"}
    monkeypatch.setattr(mirror_module, "_storage", lambda job: storage)

    with pytest.raises(RuntimeError, match="belongs to source revision"):
        mirror_module._ensure_destination_revision(_job(tmp_path, resume=True), "new-revision")
