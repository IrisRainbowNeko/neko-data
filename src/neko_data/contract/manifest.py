"""Atomic manifest and shard index serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .schema import DatasetManifest, ShardRecord, manifest_json


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def write_manifest(path: str | Path, manifest: DatasetManifest) -> Path:
    destination = Path(path)
    _atomic_write(destination, manifest_json(manifest))
    return destination


def load_manifest(path: str | Path) -> DatasetManifest:
    with Path(path).open("r", encoding="utf-8") as file:
        return DatasetManifest.from_dict(json.load(file))


def write_shard_index(path: str | Path, shards: Iterable[ShardRecord]) -> Path:
    lines = "".join(json.dumps(shard.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for shard in shards)
    destination = Path(path)
    _atomic_write(destination, lines)
    return destination

