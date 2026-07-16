import shutil
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
