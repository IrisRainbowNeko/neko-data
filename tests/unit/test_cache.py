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

