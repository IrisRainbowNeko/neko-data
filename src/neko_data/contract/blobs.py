"""Streamed file artifacts: no model deserialization or in-memory file payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .sample_keys import validate_sample_key

BLOB_SAMPLE_FORMAT = "blob-files-v1"


def safe_blob_name(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\x00" in name or path.is_absolute()
            or any(part in {".", ".."} for part in name.split("/")) or "" in name.split("/")):
        raise ValueError(f"Unsafe blob field name: {name!r}")
    return path.as_posix()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BlobSample:
    """Named local files written directly into tar; paths are never serialized."""

    sample_key: str
    files: Mapping[str, Path]
    split: str = "train"
    source_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_sample_key(self.sample_key)
        if not self.files:
            raise ValueError("Blob samples require a key and at least one file")
        for name, path in self.files.items():
            safe_blob_name(name)
            if not Path(path).is_file():
                raise ValueError(f"Blob source is not a file: {path}")
        json.dumps(dict(self.metadata), allow_nan=False)

    def prepare(self) -> tuple[list[tuple[str, Path, int, str]], dict]:
        members = []
        descriptors = {}
        for index, (name, source) in enumerate(self.files.items()):
            path = Path(source)
            before = path.stat()
            digest = file_sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Blob source changed during hashing: {path}")
            extension = f"nblob{index:08d}"
            descriptors[name] = {"extension": extension, "size_bytes": after.st_size, "sha256": digest}
            members.append((extension, path, after.st_size, digest))
        metadata = {**self.metadata, "schema_version": "1.0", "sample_format": BLOB_SAMPLE_FORMAT,
                    "sample_key": self.sample_key, "source_id": self.source_id, "split": self.split,
                    "blobs": descriptors}
        return members, metadata


@dataclass(frozen=True)
class BlobSampleReference:
    """File descriptors read from tar JSON; actual bytes remain in the shard."""

    sample_key: str
    files: Mapping[str, Mapping[str, Any]]
    split: str
    source_id: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_sample_key(self.sample_key)
        if not self.files:
            raise ValueError("Blob sample has no files")
        extensions = set()
        for name, entry in self.files.items():
            safe_blob_name(name)
            extension = entry.get("extension", "")
            if (len(extension) != 13 or not extension.startswith("nblob") or not extension[5:].isdigit()
                    or extension in extensions):
                raise ValueError("Invalid or duplicate blob member extension")
            extensions.add(extension)
            if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
                raise ValueError("Invalid blob size")
            digest = entry.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid blob SHA-256")
