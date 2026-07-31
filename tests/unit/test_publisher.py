from pathlib import Path

from neko_data.build.publisher import DatasetPublisher
from neko_data.contract import ShardRecord


class MemoryStorage:
    def __init__(self):
        self.calls = []

    def upload_file(self, source, uri, *, max_concurrency, metadata=None):
        self.calls.append((Path(source), uri, max_concurrency, metadata))


def test_publisher_attaches_shard_checksums(tmp_path):
    storage = MemoryStorage()
    tar_path = tmp_path / "shard.tar"
    metadata_path = tmp_path / "shard.parquet"
    tar_path.write_bytes(b"tar")
    metadata_path.write_bytes(b"metadata")
    record = ShardRecord(
        path="wds/train/shard.tar",
        split="train",
        num_samples=1,
        size_bytes=3,
        sha256="tar-sha",
        metadata_path="metadata/train/shard.parquet",
        metadata_sha256="metadata-sha",
    )

    DatasetPublisher("s3://bucket/dataset", storage=storage).publish_shard(
        record,
        tar_path,
        metadata_path,
    )

    assert storage.calls == [
        (
            tar_path,
            "s3://bucket/dataset/wds/train/shard.tar",
            8,
            {"sha256": "tar-sha"},
        ),
        (
            metadata_path,
            "s3://bucket/dataset/metadata/train/shard.parquet",
            8,
            {"sha256": "metadata-sha"},
        ),
    ]
