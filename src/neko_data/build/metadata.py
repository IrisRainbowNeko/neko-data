"""DuckDB joins and Parquet sidecar metadata."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..contract.records import NormalizedSample, json_dumps

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MetadataProvider(Protocol):
    def lookup(self, sample_key: str) -> Mapping[str, Any] | None:
        ...


class DuckDBMetadataProvider:
    """Per-process read-only DuckDB lookup used only during dataset building."""

    def __init__(self, db_path: str | os.PathLike[str], table: str = "data", index_key: str = "id") -> None:
        self.db_path = str(db_path)
        self.table = self._identifier(table)
        self.index_key = self._identifier(index_key)
        self._connection = None
        self._connection_pid: int | None = None

    @staticmethod
    def _identifier(value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"Unsafe SQL identifier: {value!r}")
        return value

    def _conn(self):
        if self._connection is not None and self._connection_pid == os.getpid():
            return self._connection
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB joins require 'duckdb': pip install duckdb") from exc
        self._connection = duckdb.connect(database=self.db_path, read_only=True)
        self._connection_pid = os.getpid()
        return self._connection

    def lookup(self, sample_key: str) -> Mapping[str, Any] | None:
        connection = self._conn()
        result = connection.execute(
            f"SELECT * FROM {self.table} WHERE {self.index_key} = ? LIMIT 1",
            [sample_key],
        )
        row = result.fetchone()
        if row is None and str(sample_key).isdigit():
            row = connection.execute(
                f"SELECT * FROM {self.table} WHERE {self.index_key} = ? LIMIT 1",
                [int(sample_key)],
            ).fetchone()
        if row is None:
            return None
        names = [description[0] for description in result.description]
        return dict(zip(names, row))

    def __len__(self) -> int:
        connection = self._conn()
        return int(connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])

    def close(self) -> None:
        if self._connection is not None and self._connection_pid == os.getpid():
            self._connection.close()
        self._connection = None
        self._connection_pid = None


def metadata_row(sample: NormalizedSample, shard_path: str) -> dict[str, Any]:
    """Flatten stable fields while retaining arbitrary metadata as JSON."""
    row: dict[str, Any] = {
        "sample_key": sample.sample_key,
        "source_id": sample.source_id,
        "split": sample.split,
        "shard_path": shard_path,
        "width": sample.width,
        "height": sample.height,
        "image_extension": sample.image_extension,
        "image_sha256": sample.image_sha256,
        "caption": sample.caption,
        "metadata_json": json_dumps(dict(sample.metadata)),
    }
    for key, value in sample.captions.items():
        row[f"caption_{key}"] = value
    for key, value in sample.metadata.items():
        if key in {"sample_key", "source_id", "split", "width", "height", "image_extension", "image_sha256"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            row.setdefault(key, value)
    return row


class MetadataWriter:
    """Write one Parquet sidecar per tar shard."""

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)

    def write(self, path: str | os.PathLike[str], samples: list[NormalizedSample], shard_path: str) -> Path | None:
        rows = [metadata_row(sample, shard_path) for sample in samples]
        return self.write_rows(path, rows)

    def write_rows(self, path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> Path | None:
        if not rows:
            return None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet metadata requires pyarrow") from exc

        destination = self.output_dir / Path(path) if not Path(path).is_absolute() else Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row})
        table = pa.Table.from_pylist([{key: row.get(key) for key in keys} for row in rows])
        temporary = destination.with_name(f".{destination.name}.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, destination)
        return destination
