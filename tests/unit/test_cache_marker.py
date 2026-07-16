import hashlib
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

