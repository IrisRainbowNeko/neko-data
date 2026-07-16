"""Read-only HTTP object storage adapter for public manifests and shards."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path


class HTTPStorage:
    """Stream HTTP objects without loading them into memory."""

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout

    @contextmanager
    def open(self, uri: str):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("HTTP storage requires requests; install neko-data[hf]") from exc
        with requests.get(uri, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            yield response.raw

    def download(self, uri: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.open(uri) as stream, destination.open("wb") as output:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                output.write(block)
