"""Load a manifest from a local path, HTTP endpoint, or S3/R2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..contract.schema import DatasetManifest, ShardRecord
from ..storage import HTTPStorage, S3Storage, is_local_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedManifest:
    manifest: DatasetManifest
    base_uri: str

    def shard_uri(self, shard: ShardRecord) -> str:
        return self.manifest.resolve_path(self.base_uri, shard.path)


class ManifestLoader:
    def __init__(self, storage=None) -> None:
        self.storage = storage

    @staticmethod
    def _allow_ambiguous_metadata_paths(data: dict) -> dict:
        """Drop sidecar references that collide in an upstream manifest.

        Some legacy datasets reused ``metadata/train/train-000000.parquet``
        for several independent tar workers.  Keeping any one reference would
        associate the wrong metadata with most shards.  The tar paths remain
        valid, so an explicitly opted-in training reader can safely ignore the
        ambiguous sidecars while the default contract stays strict.
        """
        raw_shards = data.get("shards", [])
        if not isinstance(raw_shards, list):
            return data
        counts: dict[str, int] = {}
        for shard in raw_shards:
            if isinstance(shard, dict):
                path = shard.get("metadata_path")
                if isinstance(path, str) and path:
                    counts[path] = counts.get(path, 0) + 1
        duplicates = {path for path, count in counts.items() if count > 1}
        if not duplicates:
            return data
        sanitized = dict(data)
        sanitized["shards"] = []
        for shard in raw_shards:
            if not isinstance(shard, dict):
                sanitized["shards"].append(shard)
                continue
            current = dict(shard)
            if current.get("metadata_path") in duplicates:
                current.pop("metadata_path", None)
                current.pop("metadata_sha256", None)
            sanitized["shards"].append(current)
        logger.warning(
            "Ignoring %d ambiguous metadata_path values in legacy manifest; "
            "tar samples remain available but sidecars are disabled",
            len(duplicates),
        )
        return sanitized

    def load(
        self,
        uri: str | Path,
        *,
        allow_duplicate_metadata_paths: bool = False,
    ) -> LoadedManifest:
        value = str(uri)
        if is_local_path(value):
            path = Path(value[7:] if value.startswith("file://") else value)
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if allow_duplicate_metadata_paths:
                data = self._allow_ambiguous_metadata_paths(data)
            return LoadedManifest(DatasetManifest.from_dict(data), str(path.parent))
        parsed = urlparse(value)
        if parsed.scheme in {"s3", "r2"}:
            storage = self.storage or S3Storage()
            with storage.open(value) as file:
                data = json.load(file)
        elif parsed.scheme in {"http", "https"}:
            storage = self.storage or HTTPStorage()
            with storage.open(value) as file:
                data = json.load(file)
        else:
            raise ValueError(f"Unsupported manifest URI: {value!r}")
        if allow_duplicate_metadata_paths:
            data = self._allow_ambiguous_metadata_paths(data)
        base = value.rsplit("/", 1)[0]
        return LoadedManifest(DatasetManifest.from_dict(data), base)
