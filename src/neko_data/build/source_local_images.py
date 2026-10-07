"""Deterministic, byte-preserving input for local image directory trees."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ..contract.records import NormalizedSample
from .normalize import IMAGE_EXTENSIONS, sample_from_mapping


class LocalImagesSource:
    """Read one image at a time, retaining relative paths as sample identities.

    Files are sorted by relative path so an unchanged tree supports BuildJob
    resume. Images are never decoded and re-encoded, and no captions or labels
    are inferred from directory names.
    """

    def __init__(self, root: str | os.PathLike[str], *, split: str = "train",
                 source_id: str = "local-images") -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.split = split
        self.source_id = source_id

    def __iter__(self) -> Iterator[NormalizedSample]:
        paths = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No images found in {self.root}")
        keys: set[str] = set()
        # Validate identities before emitting anything, including on resume.
        for path in paths:
            key = path.relative_to(self.root).with_suffix("").as_posix()
            if key in keys:
                raise ValueError(f"Duplicate local image sample key: {key!r}")
            keys.add(key)
        for path in paths:
            relative = path.relative_to(self.root)
            sample = sample_from_mapping(
                {"id": relative.with_suffix("").as_posix(), "image": path,
                 "relative_path": relative.as_posix()},
                split=self.split,
                source_id=self.source_id,
            )
            if sample.width is None or sample.height is None:
                raise ValueError(f"Cannot read image dimensions: {relative.as_posix()}")
            yield sample
