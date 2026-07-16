"""Local filesystem implementation of the small storage interface."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import BinaryIO


def is_local_path(uri: str | os.PathLike[str]) -> bool:
    value = os.fspath(uri)
    return "://" not in value or value.startswith("file://")


def local_path_from_uri(uri: str | os.PathLike[str]) -> Path:
    value = os.fspath(uri)
    if value.startswith("file://"):
        value = value[7:]
    return Path(value)


class LocalStorage:
    """Storage adapter used by local builds and tests."""

    def open(self, uri: str | os.PathLike[str]) -> BinaryIO:
        return local_path_from_uri(uri).open("rb")

    def download(self, uri: str | os.PathLike[str], destination: Path) -> None:
        source = local_path_from_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def upload_file(self, source: Path, uri: str) -> None:
        destination = local_path_from_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def head(self, uri: str | os.PathLike[str]) -> dict[str, int]:
        path = local_path_from_uri(uri)
        return {"size_bytes": path.stat().st_size}

