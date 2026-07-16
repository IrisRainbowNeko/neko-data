"""Small, bounded Parquet writer used by the streaming shard writer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def write_parquet_rows(output_dir: Path, path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet metadata requires pyarrow") from exc

    destination = output_dir / Path(path) if not Path(path).is_absolute() else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    table_rows = [{key: row.get(key) for key in keys} for row in rows]
    table = pa.Table.from_pylist(table_rows)
    temporary = destination.with_name(f".{destination.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, destination)
    return destination

