"""Streaming adapter for non-WebDataset HuggingFace datasets."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Iterable, Iterator

from ..contract.records import NormalizedSample
from .normalize import sample_from_mapping


class HFDatasetsSource:
    """Use ``datasets.load_dataset(..., streaming=True)`` without full download."""

    def __init__(
        self,
        path: str,
        *,
        name: str | None = None,
        split: str = "train",
        revision: str | None = None,
        token: str | None = None,
        streaming: bool = True,
        image_column: str = "image",
        key_column: str | None = None,
        caption_columns: Iterable[str] | None = None,
        source_id: str | None = None,
        expected_samples: int | None = None,
        metadata_provider=None,
        image_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if not streaming:
            raise ValueError("HFDatasetsSource requires streaming=True")
        self.path = path
        self.name = name
        self.split = split
        self.revision = revision
        self.token = token
        self.image_column = image_column
        self.key_column = key_column
        self.caption_columns = tuple(caption_columns) if caption_columns is not None else None
        self.source_id = source_id or path
        self.expected_samples = expected_samples
        self.metadata_provider = metadata_provider
        self.image_root = Path(image_root) if image_root is not None else None

    def _dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("HF datasets support requires the 'hf' extra: pip install neko-data[hf]") from exc
        kwargs = {
            "path": self.path,
            "split": self.split,
            "streaming": True,
        }
        if self.name is not None:
            kwargs["name"] = self.name
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.token is not None and "token" in inspect.signature(load_dataset).parameters:
            kwargs["token"] = self.token
        return load_dataset(**kwargs)

    def __iter__(self) -> Iterator[NormalizedSample]:
        for row in self._dataset():
            if self.key_column:
                key = row.get(self.key_column)
            elif "id" in row:
                key = row["id"]
            else:
                key = row.get("key")
            provider_data = None
            if self.metadata_provider is not None and key is not None:
                provider_data = self.metadata_provider.lookup(str(key))
            yield sample_from_mapping(
                row,
                image_column=self.image_column,
                key_column=self.key_column,
                image_root=self.image_root,
                caption_columns=self.caption_columns,
                split=self.split,
                source_id=self.source_id,
                metadata_provider=provider_data,
            )

    def __len__(self) -> int:
        if self.expected_samples is None:
            raise TypeError("streaming HF dataset length is unknown; set expected_samples to expose one")
        return self.expected_samples

