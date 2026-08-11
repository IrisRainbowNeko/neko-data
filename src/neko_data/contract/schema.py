"""Versioned manifest schema for a published dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping

DATASET_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ShardRecord:
    """A content-addressed WebDataset tar and its sidecar metadata."""

    path: str
    split: str
    num_samples: int
    size_bytes: int
    sha256: str
    metadata_path: str | None = None
    metadata_sha256: str | None = None
    index: int | None = None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.path:
            raise ValueError("shard path cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "split": self.split,
            "num_samples": self.num_samples,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.metadata_path is not None:
            data["metadata_path"] = self.metadata_path
        if self.metadata_sha256 is not None:
            data["metadata_sha256"] = self.metadata_sha256
        if self.index is not None:
            data["index"] = self.index
        if self.source_ids:
            data["source_ids"] = list(self.source_ids)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShardRecord":
        return cls(
            path=str(data["path"]),
            split=str(data.get("split", "train")),
            num_samples=int(data.get("num_samples", 0)),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=str(data.get("sha256", "")),
            metadata_path=data.get("metadata_path"),
            metadata_sha256=data.get("metadata_sha256"),
            index=None if data.get("index") is None else int(data["index"]),
            source_ids=tuple(str(x) for x in data.get("source_ids", [])),
        )


@dataclass
class DatasetManifest:
    """Top-level dataset contract.

    All paths are relative to the directory containing ``manifest.json`` or
    the URI prefix containing it.  This keeps the same manifest usable from a
    local build directory and from ``s3://``/R2.
    """

    dataset_id: str
    version: str
    shards: list[ShardRecord] = field(default_factory=list)
    schema_version: str = DATASET_SCHEMA_VERSION
    format: str = "webdataset"
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")
        if self.format != "webdataset":
            raise ValueError("v0.1 supports WebDataset tar shards only")
        self._validate_unique_paths()

    def _validate_unique_paths(self) -> None:
        for field_name in ("path", "metadata_path"):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for shard in self.shards:
                value = getattr(shard, field_name)
                if not value:
                    continue
                if value in seen:
                    duplicates.add(value)
                seen.add(value)
            if duplicates:
                examples = ", ".join(sorted(duplicates)[:3])
                raise ValueError(
                    f"manifest contains duplicate {field_name} values: {examples}"
                )

    @property
    def splits(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for shard in self.shards:
            result[shard.split] = result.get(shard.split, 0) + shard.num_samples
        return result

    def for_split(self, split: str) -> list[ShardRecord]:
        return [shard for shard in self.shards if shard.split == split]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "format": self.format,
            "created_at": self.created_at,
            "shards": [shard.to_dict() for shard in self.shards],
            "splits": self.splits,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        schema_version = str(data.get("schema_version", DATASET_SCHEMA_VERSION))
        if schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema version: {schema_version}")
        return cls(
            dataset_id=str(data["dataset_id"]),
            version=str(data["version"]),
            shards=[ShardRecord.from_dict(item) for item in data.get("shards", [])],
            schema_version=schema_version,
            format=str(data.get("format", "webdataset")),
            created_at=str(data.get("created_at", _utc_now())),
            metadata=dict(data.get("metadata", {})),
        )

    def resolve_path(self, base: str, relative_path: str) -> str:
        """Resolve a shard path against a local directory or object URI."""
        if "://" in relative_path or relative_path.startswith("/"):
            return relative_path
        if base.endswith("/"):
            return base + str(PurePosixPath(relative_path))
        return base + "/" + str(PurePosixPath(relative_path))


def manifest_json(manifest: DatasetManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
