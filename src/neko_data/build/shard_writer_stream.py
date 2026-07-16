"""Streaming WebDataset writer used by BuildJob."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Callable

from ..contract.records import NormalizedSample
from ..contract.schema import ShardRecord
from .metadata import metadata_row
from .metadata_rows import write_parquet_rows
from .shard_writer import parse_size


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


class StreamingWebDatasetShardWriter:
    """Write image, complete JSON metadata, optional txt, and Parquet sidecars."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        metadata_dir: str | os.PathLike[str],
        *,
        split: str = "train",
        max_shard_size: int | str = "1GiB",
        max_samples_per_shard: int = 0,
        pattern: str = "{split}-{index:06d}.tar",
        start_index: int = 0,
        overwrite: bool = False,
        on_shard: Callable[[ShardRecord, Path, Path | None], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.max_shard_size = parse_size(max_shard_size)
        self.max_samples_per_shard = max_samples_per_shard
        self.pattern = pattern
        self.next_index = start_index
        self.overwrite = overwrite
        self.on_shard = on_shard
        self.records: list[ShardRecord] = []
        self._tar: tarfile.TarFile | None = None
        self._temporary: Path | None = None
        self._final: Path | None = None
        self._name: str | None = None
        self._size = 0
        self._count = 0
        self._rows: list[dict] = []
        self._source_ids: set[str] = set()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        while True:
            name = self.pattern.format(split=self.split, index=self.next_index)
            self.next_index += 1
            final = self.output_dir / name
            temporary = final.with_name(f".{final.name}.tmp")
            if self.overwrite or (not final.exists() and not temporary.exists()):
                break
            raise FileExistsError(f"Output shard exists: {final}; set overwrite=True to replace it")
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.unlink(missing_ok=True)
        self._tar = tarfile.open(temporary, mode="w")
        self._temporary, self._final, self._name = temporary, final, name
        self._size, self._count = 0, 0
        self._rows, self._source_ids = [], set()

    @staticmethod
    def _entry(name: str, data: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o644
        return info, io.BytesIO(data)

    def add(self, sample: NormalizedSample) -> None:
        json_data = sample.metadata_json()
        text_data = (sample.caption + "\n").encode("utf-8") if sample.caption else None
        estimated = sum(len(data) + 1024 for data in (sample.image, json_data, text_data or b""))
        if self._tar is not None and self._count and (
            self._size + estimated > self.max_shard_size
            or (self.max_samples_per_shard > 0 and self._count >= self.max_samples_per_shard)
        ):
            self.close_current()
        if self._tar is None:
            self._open()
        assert self._tar is not None and self._name is not None
        entries = [
            self._entry(_member_name(sample.sample_key, sample.image_extension), sample.image),
            self._entry(_member_name(sample.sample_key, "json"), json_data),
        ]
        if text_data is not None:
            entries.append(self._entry(_member_name(sample.sample_key, "txt"), text_data))
        for info, content in entries:
            self._tar.addfile(info, content)
        self._size += estimated
        self._count += 1
        self._rows.append(metadata_row(sample, self._name))
        self._source_ids.add(sample.source_id)

    def close_current(self) -> ShardRecord | None:
        if self._tar is None:
            return None
        assert self._temporary is not None and self._final is not None and self._name is not None
        self._tar.close()
        self._tar = None
        if not self._count:
            self._temporary.unlink(missing_ok=True)
            self._reset()
            return None
        os.replace(self._temporary, self._final)
        metadata_name = Path(self._name).with_suffix(".parquet")
        metadata_path = write_parquet_rows(self.metadata_dir, metadata_name, self._rows)
        record = ShardRecord(
            path=(Path("wds") / self.split / self._name).as_posix(),
            split=self.split,
            num_samples=self._count,
            size_bytes=self._final.stat().st_size,
            sha256=_sha256(self._final),
            metadata_path=(Path("metadata") / self.split / metadata_name.name).as_posix() if metadata_path else None,
            metadata_sha256=_sha256(metadata_path) if metadata_path else None,
            index=self.next_index - 1,
            source_ids=tuple(sorted(self._source_ids)),
        )
        self.records.append(record)
        if self.on_shard is not None:
            self.on_shard(record, self._final, metadata_path)
        self._reset()
        return record

    def _reset(self) -> None:
        self._temporary = self._final = self._name = None
        self._size, self._count = 0, 0
        self._rows, self._source_ids = [], set()

    def close(self) -> list[ShardRecord]:
        self.close_current()
        return list(self.records)

