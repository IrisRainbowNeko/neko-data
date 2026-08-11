import os
import shutil
import threading
from pathlib import Path

from neko_data.contract.schema import ShardRecord
from neko_data.runtime.cache import DiskShardCache


class FakeRemoteStorage:
    def __init__(self, source: Path):
        self.source = source
        self.downloads = 0

    def download(self, uri, destination):
        self.downloads += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, destination)

    def open(self, uri):
        return self.source.open("rb")


def test_cache_downloads_once_and_holds_lease(tmp_path):
    source = tmp_path / "remote.tar"
    source.write_bytes(b"tar bytes")
    import hashlib
    record = ShardRecord(
        "wds/train.tar", "train", 1, source.stat().st_size,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    storage = FakeRemoteStorage(source)
    cache = DiskShardCache(tmp_path / "cache", storage, max_size_bytes=1024, evict_size_bytes=512)
    with cache.open("s3://bucket/train.tar", record) as path:
        assert path.read_bytes() == b"tar bytes"
        leases = list((tmp_path / "cache" / "objects").glob("*.lease.*"))
        assert leases
    with cache.open("s3://bucket/train.tar", record) as path:
        assert path.exists()
    assert storage.downloads == 1
    assert cache.stats.hits == 1



def test_cache_eviction_respects_active_lease(tmp_path):
    source_one = tmp_path / "one.tar"
    source_one.write_bytes(b"firsttar")
    source_two = tmp_path / "two.tar"
    source_two.write_bytes(b"secondtr")

    class Remote:
        def __init__(self, sources):
            self.sources = sources

        def download(self, uri, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.sources[uri], destination)

    storage = Remote({"s3://bucket/one": source_one, "s3://bucket/two": source_two})
    import hashlib

    def record(path, name):
        return ShardRecord(
            f"wds/{name}.tar", "train", 1, path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    record_one = record(source_one, "one")
    record_two = record(source_two, "two")
    cache = DiskShardCache(tmp_path / "cache", storage, max_size_bytes=12, evict_size_bytes=0)
    with cache.open("s3://bucket/one", record_one) as first_path:
        with cache.open("s3://bucket/two", record_two) as second_path:
            assert second_path.exists()
        assert first_path.exists()
    assert first_path.exists()


def test_cache_download_does_not_leave_partial_verification_marker(tmp_path):
    source = tmp_path / "remote.tar"
    source.write_bytes(b"tar bytes")
    import hashlib

    record = ShardRecord(
        "wds/train.tar", "train", 1, source.stat().st_size,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    cache = DiskShardCache(tmp_path / "cache", FakeRemoteStorage(source))
    with cache.open("s3://bucket/train.tar", record):
        pass
    assert not list(cache.objects.glob("*.partial.*"))


def test_cache_eviction_is_single_writer(tmp_path, monkeypatch):
    source = tmp_path / "remote.tar"
    source.write_bytes(b"tar bytes")
    cache_one = DiskShardCache(tmp_path / "cache", FakeRemoteStorage(source))
    cache_two = DiskShardCache(tmp_path / "cache", FakeRemoteStorage(source))
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def slow_evict():
        calls.append("one")
        entered.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(cache_one, "_evict_locked", slow_evict)
    monkeypatch.setattr(cache_two, "_evict_locked", lambda: calls.append("two"))
    thread = threading.Thread(target=cache_one._evict)
    thread.start()
    assert entered.wait(timeout=2)
    cache_two._evict()
    release.set()
    thread.join(timeout=2)
    assert calls == ["one"]


def test_cache_eviction_census_finds_leased_object(tmp_path, monkeypatch):
    cache = DiskShardCache(
        tmp_path / "cache", FakeRemoteStorage(tmp_path / "unused"),
        max_size_bytes=1, evict_size_bytes=0,
    )
    leased = cache.objects / "leased.tar"
    evictable = cache.objects / "evictable.tar"
    leased.write_bytes(b"leased")
    evictable.write_bytes(b"evictable")
    lease = cache._lease(leased)
    monkeypatch.setattr(cache, "_has_lease", lambda path: False)
    try:
        cache._evict_locked()
    finally:
        lease.unlink(missing_ok=True)
    assert leased.exists()
    assert not evictable.exists()


def test_cache_initialization_removes_dead_process_files(tmp_path):
    objects = tmp_path / "cache" / "objects"
    objects.mkdir(parents=True)
    hostname = __import__("socket").gethostname().replace(".", "_")
    dead_pid = 2**31 - 1
    stale_partial = objects / f".item.tar.partial.{dead_pid}.token"
    stale_lease = objects / f".item.tar.lease.{hostname}.{dead_pid}.token"
    orphan_marker = objects / ".missing.tar.verified"
    active_partial = objects / f".live.tar.partial.{os.getpid()}.token"
    active_lease = objects / f".live.tar.lease.{hostname}.{os.getpid()}.token"
    for path in (
        stale_partial, stale_lease, orphan_marker, active_partial, active_lease
    ):
        path.write_text("state")

    DiskShardCache(tmp_path / "cache", FakeRemoteStorage(tmp_path / "unused"))

    assert not stale_partial.exists()
    assert not stale_lease.exists()
    assert not orphan_marker.exists()
    assert active_partial.exists()
    assert active_lease.exists()
