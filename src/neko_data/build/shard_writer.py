"""Bounded WebDataset tar and Parquet sidecar writer."""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Callable

from ..contract.records import NormalizedSample
from ..contract.schema import ShardRecord
from .metadata import MetadataWriter, metadata_row

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|b)?\s*$", re.IGNORECASE)


def parse_size(value: int | str) -> int:
    if isinstance(value, int):
        return value
    match = _SIZE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "": 1, "b": 1, "k": 1000, "kb": 1000, "m": 1000**2,
        "mb": 1000**2, "g": 1000**3, "gb": 1000**3, "t": 1000**4,
        "tb": 1000**4, "ki": 1024, "kib": 1024, "mi": 1024**2,
        "mib": 1024**2, "gi": 1024**3, "gib": 1024**3, "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Invalid size unit: {unit!r}")
    return int(number * multipliers[unit])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_name(sample_key: str, extension: str) -> str:
    key = sample_key.replace("\\", "/")
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or "\x00" in key:
        raise ValueError(f"Unsafe sample key for tar member: {sample_key!r}")
    return f"{key}.{extension.lstrip('.') or 'jpg'}"


class WebDatasetShardWriter:
    """Write sample-aligned shards with atomic finalization."""

    def __init__(self, output_dir: str | os.PathLike[str], metadata_dir: str | os.PathLike[str], *,
                 split: str = "train", max_shard_size: int | str = "1GiB",
                 max_samples_per_shard: int = 0, pattern: str = "{split}-{index:06d}.tar",
                 start_index: int = 0, overwrite: bool = False,
                 on_shard: Callable[[ShardRecord, Path, Path | None], None] | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.max_shard_size = parse_size(max_shard_size)
        self.max_samples_per_shard = max_samples_per_shard
        self.pattern = pattern
        self.next_index = start_index
        self.overwrite = overwrite
        self.on_shard = on_shard
        self.metadata_writer = MetadataWriter(self.metadata_dir)
        self._tar: tarfile.TarFile | None = None
        self._temporary: Path | None = None
        self._final: Path | None = None
        self._current_name: str | None = None
        self._estimated_size = 0
        self._sample_count = 0
        self._rows: list[dict] = []
        self._source_ids: set[str] = set()
        self.records: list[ShardRecord] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _open_next(self) -> None:
        while True:
            name = self.pattern.format(split=self.split, index=self.next_index)
            self.next_index += 1
            final = self.output_dir / name
            temporary = final.with_name(f".{final.name}.tmp")
            if self.overwrite or (not final.exists() and not temporary.exists()):
                break
            raise FileExistsError(f"Output shard exists: {final}; use overwrite=True or another start_index")
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        self._tar = tarfile.open(temporary, mode="w")
        self._temporary, self._final, self._current_name = temporary, final, name
        self._estimated_size, self._sample_count = 0, 0
        self._rows, self._source_ids = [], set()

    @staticmethod
    def _entry(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
        info = tarfile.TarInfo(name=name)
        info.size, info.mode = len(data), 0o644
        return info, io.BytesIO(data)

    def add(self, sample: NormalizedSample) -> None:
        json_data = sample.metadata_json()
        text_data = (sample.caption + "\n").encode("utf-8") if sample.caption else None
        estimated = sum(len(data) + 1024 for data in (sample.image, json_data, text_data or b""))
        if self._tar is not None and self._sample_count > 0 and (
            self._estimated_size + estimated > self.max_shard_size
            or (self.max_samples_per_shard > 0 and self._sample_count >= self.max_samples_per_shard)
        ):
            self.close_current()
        if self._tar is None:
            self._open_next()
        assert self._tar is not None and self._current_name is not None
        key = sample.sample_key
        entries = [self._entry(_member_name(key, sample.image_extension), sample.image),
                   self._entry(_member_name(key, "json"), json_data)]
        if text_data is not None:
            entries.append(self._entry(_member_name(key, "txt"), text_data))
        for info, data in entries:
            self._tar.addfile(info, data)
        self._estimated_size += estimated
        self._sample_count += 1
        self._rows.append(metadata_row(sample, self._current_name))
        self._source_ids.add(sample.source_id)

    def close_current(self) -> ShardRecord | None:
        if self._tar is None:
            return None
        assert self._temporary is not None and self._final is not None and self._current_name is not None
        self._tar.close()
        self._tar = None
        if self._sample_count == 0:
            self._temporary.unlink(missing_ok=True)
            self._reset()
            return None
        os.replace(self._temporary, self._final)
        relative_metadata = Path(self._current_name).with_suffix(".parquet")
        metadata_path = self.metadata_writer.write_rows(relative_metadata, self._rows)
        relative_tar = Path("wds") / self.split / self._current_name
        relative_metadata_path = Path("metadata") / self.split / relative_metadata.name
        record = ShardRecord(path=relative_tar.as_posix(), split=self.split,
                             num_samples=self._sample_count, size_bytes=self._final.stat().st_size,
                             sha256=_sha256(self._final),
                             metadata_path=relative_metadata_path.as_posix() if metadata_path else None,
                             metadata_sha256=_sha256(metadata_path) if metadata_path else None,
                             index=self.next_index - 1, source_ids=tuple(sorted(self._source_ids)))
        self.records.append(record)
        if self.on_shard is not None:
            self.on_shard(record, self._final, metadata_path)
        self._reset()
        return record

    def _reset(self) -> None:
        self._temporary = self._final = self._current_name = None
        self._estimated_size, self._sample_count = 0, 0
        self._rows, self._source_ids = [], set()

    def close(self) -> list[ShardRecord]:
        self.close_current()
        return list(self.records)

