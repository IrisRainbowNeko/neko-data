"""Publish completed shards to a local prefix or S3-compatible R2."""

from __future__ import annotations

import os
from pathlib import Path

from ..contract.schema import ShardRecord
from ..storage import LocalStorage, S3Storage, is_local_path, local_path_from_uri


def join_uri(base: str | os.PathLike[str], relative: str) -> str:
    base_value = os.fspath(base)
    if is_local_path(base_value):
        return str(local_path_from_uri(base_value) / Path(relative))
    return base_value.rstrip("/") + "/" + relative.lstrip("/")


class DatasetPublisher:
    """Upload one finalized shard at a time and optionally release its disk."""

    def __init__(
        self,
        destination: str | os.PathLike[str],
        *,
        storage=None,
        delete_after_upload: bool = False,
        upload_concurrency: int = 8,
    ) -> None:
        self.destination = os.fspath(destination)
        self.delete_after_upload = delete_after_upload
        self.upload_concurrency = upload_concurrency
        self.storage = storage
        if self.storage is None:
            self.storage = LocalStorage() if is_local_path(self.destination) else S3Storage()

    def publish_file(self, local_path: Path, relative_path: str) -> str:
        target = join_uri(self.destination, relative_path)
        if isinstance(self.storage, LocalStorage):
            self.storage.upload_file(local_path, target)
        else:
            self.storage.upload_file(local_path, target, max_concurrency=self.upload_concurrency)
        return target

    def publish_shard(self, record: ShardRecord, tar_path: Path, metadata_path: Path | None) -> None:
        self.publish_file(tar_path, record.path)
        if metadata_path is not None and record.metadata_path is not None:
            self.publish_file(metadata_path, record.metadata_path)
        if self.delete_after_upload:
            tar_path.unlink(missing_ok=True)
            if metadata_path is not None:
                metadata_path.unlink(missing_ok=True)

    def publish_manifest(self, local_path: Path) -> str:
        return self.publish_file(local_path, "manifest.json")

    def publish_index(self, local_path: Path, split: str) -> str:
        return self.publish_file(local_path, f"indexes/{split}/shards.jsonl")

