import hashlib
import json
import os
import shutil

from neko_data.contract.schema import ShardRecord
from neko_data.runtime import cache as cache_module
from neko_data.runtime.cache import DiskShardCache


class Remote:
    def __init__(self, source):
        self.source = source

    def download(self, uri, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, destination)


def test_verified_marker_avoids_rehashing_cache_hits(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"content")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = ShardRecord("train.tar", "train", 1, source.stat().st_size, digest)
    calls = 0
    original = cache_module._file_sha256

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(cache_module, "_file_sha256", counted)
    cache = DiskShardCache(tmp_path / "cache", Remote(source))
    with cache.open("s3://bucket/train.tar", record):
        pass
    with cache.open("s3://bucket/train.tar", record):
        pass
    assert calls == 1


def test_same_size_rewrite_with_restored_mtime_is_repaired(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"trusted")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = ShardRecord("train.tar", "train", 1, source.stat().st_size, digest)
    cache = DiskShardCache(tmp_path / "cache", Remote(source))
    with cache.acquire("s3://bucket/train.tar", record) as path:
        before = path.stat()
    path.write_bytes(b"changed")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with cache.acquire("s3://bucket/train.tar", record) as repaired:
        assert repaired.read_bytes() == b"trusted"
    assert cache.stats.downloads == 2


def test_old_and_corrupt_markers_are_rehashed(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"content")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = ShardRecord("train.tar", "train", 1, source.stat().st_size, digest)
    cache = DiskShardCache(tmp_path / "cache", Remote(source))
    with cache.acquire("s3://bucket/train.tar", record) as path:
        pass
    calls = []
    original = cache_module._file_sha256
    monkeypatch.setattr(cache_module, "_file_sha256", lambda p: calls.append(p) or original(p))
    for old_marker in (digest + "\n", "{invalid-json", "[]"):
        cache._verified_marker(path).write_text(old_marker, encoding="ascii")
        with cache.acquire("s3://bucket/train.tar", record):
            pass
    assert len(calls) == 3
    marker = json.loads(cache._verified_marker(path).read_text(encoding="ascii"))
    assert marker["version"] == 2
    assert set(marker["fingerprint"]) >= {"size", "mtime_ns", "ctime_ns", "inode", "device"}


def test_change_during_verification_is_not_blessed(tmp_path, monkeypatch):
    source = tmp_path / "source.tar"
    source.write_bytes(b"content")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    record = ShardRecord("train.tar", "train", 1, source.stat().st_size, digest)
    cache = DiskShardCache(tmp_path / "cache", Remote(source))
    actual = cache._fingerprint(source)
    changed = {**actual, "inode": actual["inode"] + 1}
    fingerprints = iter((actual, changed))
    monkeypatch.setattr(cache, "_fingerprint", lambda _path: next(fingerprints))
    assert not cache._valid(source, record)
    assert not cache._verified_marker(source).exists()

