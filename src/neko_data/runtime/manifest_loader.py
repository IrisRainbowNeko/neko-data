"""Load a manifest from a local path, HTTP endpoint, or S3/R2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..contract.schema import DatasetManifest, ShardRecord
from ..storage import S3Storage, is_local_path


@dataclass(frozen=True)
class LoadedManifest:
    manifest: DatasetManifest
    base_uri: str

    def shard_uri(self, shard: ShardRecord) -> str:
        return self.manifest.resolve_path(self.base_uri, shard.path)


class ManifestLoader:
    def __init__(self, storage=None) -> None:
        self.storage = storage

    def load(self, uri: str | Path) -> LoadedManifest:
        value = str(uri)
        if is_local_path(value):
            path = Path(value[7:] if value.startswith("file://") else value)
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return LoadedManifest(DatasetManifest.from_dict(data), str(path.parent))
        parsed = urlparse(value)
        if parsed.scheme in {"s3", "r2"}:
            storage = self.storage or S3Storage()
            with storage.open(value) as file:
                data = json.load(file)
        elif parsed.scheme in {"http", "https"}:
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError("HTTP manifest loading requires requests") from exc
            response = requests.get(value, timeout=60)
            response.raise_for_status()
            data = response.json()
        else:
            raise ValueError(f"Unsupported manifest URI: {value!r}")
        base = value.rsplit("/", 1)[0]
        return LoadedManifest(DatasetManifest.from_dict(data), base)

