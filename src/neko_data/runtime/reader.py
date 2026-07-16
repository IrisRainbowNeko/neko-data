"""Read normalized samples from completed WebDataset shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..build.source_hf_webdataset import normalized_samples_from_tar


def iter_shard_samples(file_or_path, *, split: str, source_id: str) -> Iterator:
    if isinstance(file_or_path, (str, Path)):
        with Path(file_or_path).open("rb") as file:
            yield from normalized_samples_from_tar(file, split=split, source_id=source_id)
    else:
        yield from normalized_samples_from_tar(file_or_path, split=split, source_id=source_id)

