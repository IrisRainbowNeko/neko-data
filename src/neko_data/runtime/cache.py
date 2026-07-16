"""Node-local content-addressed shard cache with leases and high-water eviction."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..contract.schema import ShardRecord
from ..storage import is_local_path, local_path_from_uri


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    downloads: int = 0
    waits: int = 0
    errors: int = 0
    bytes_downloaded: int = 0


class DiskShardCache:
    """A process-safe cache intended to be shared by all ranks on one node."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        storage,
        *,
        max_size_bytes: int = 420 * 1024**3,
        evict_size_bytes: int = 350 * 1024**3,
        strategy: str = "required",
    ) -> None:
        if strategy not in {"required", "preferred", "disabled"}:
            raise ValueError("strategy must be required, preferred, or disabled")
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.storage = storage
        self.max_size_bytes = max_size_bytes
        self.evict_size_bytes = min(evict_size_bytes, max_size_bytes)
        self.strategy = strategy
        self.stats = CacheStats()
        self.objects.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _lock(path: Path):
        try:
            from filelock import FileLock
        except ImportError as exc:
            raise RuntimeError("Disk cache requires filelock") from exc
        return FileLock(str(path) + ".lock")

    def _key(self, uri: str, shard: ShardRecord | None) -> str:
        if shard is not None and shard.sha256:
            return shard.sha256
        return hashlib.sha256(uri.encode("utf-8")).hexdigest()

    def _verified_marker(self, path: Path) -> Path:
        return path.with_name(f".{path.name}.verified")

    def _valid(self, path: Path, shard: ShardRecord | None) -> bool:
        if not path.exists() or not path.is_file():
            return False
        stat = path.stat()
        if shard is not None and shard.size_bytes and stat.st_size != shard.size_bytes:
            return False
        if shard is None or not shard.sha256:
            return True
        marker = self._verified_marker(path)
        if (marker.exists() and marker.read_text(encoding="ascii").strip() == shard.sha256
                and marker.stat().st_mtime_ns >= stat.st_mtime_ns):
            return True
        if _file_sha256(path) != shard.sha256:
            return False
        marker.write_text(shard.sha256 + "\n", encoding="ascii")
        return True

    def _lease(self, path: Path) -> Path:
        lease = path.with_name(f".{path.name}.lease.{os.getpid()}.{uuid.uuid4().hex}")
        lease.touch()
        return lease

    def _evict(self) -> None:
        entries = [
            path for path in self.objects.iterdir()
            if (path.is_file() and ".lease." not in path.name
                    and ".partial." not in path.name and ".verified" not in path.name
                    and not path.name.endswith(".lock"))
        ]
        total = sum(path.stat().st_size for path in entries)
        if total <= self.max_size_bytes:
            return
        for path in sorted(entries, key=lambda item: item.stat().st_atime):
            if total <= self.evict_size_bytes:
                break
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
            except FileNotFoundError:
                continue

    def _download(self, uri: str, path: Path, shard: ShardRecord | None) -> None:
        partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            self.storage.download(uri, partial)
            if not self._valid(partial, shard):
                raise IOError(f"Downloaded shard failed size/checksum validation: {uri}")
            os.replace(partial, path)
            if shard is not None and shard.sha256:
                self._verified_marker(path).write_text(shard.sha256 + "\n", encoding="ascii")
            self.stats.downloads += 1
            self.stats.bytes_downloaded += path.stat().st_size
        finally:
            partial.unlink(missing_ok=True)

    @contextmanager
    def acquire(self, uri: str, shard: ShardRecord | None = None) -> Iterator[Path]:
        if is_local_path(uri):
            yield local_path_from_uri(uri)
            return
        key = self._key(uri, shard)
        path = self.objects / f"{key}.tar"
        lock = self._lock(path)
        if getattr(lock, "is_locked", False):
            self.stats.waits += 1
        with lock:
            if self._valid(path, shard):
                self.stats.hits += 1
            else:
                try:
                    self._download(uri, path, shard)
                except Exception:
                    self.stats.errors += 1
                    raise
            stat = path.stat()
            os.utime(path, (time.time(), stat.st_mtime))
            lease = self._lease(path)
        try:
            yield path
        finally:
            lease.unlink(missing_ok=True)
            self._evict()

    @contextmanager
    def open(self, uri: str, shard: ShardRecord | None = None):
        """Open a shard, falling back to direct streaming in preferred mode."""
        if self.strategy == "disabled" or is_local_path(uri):
            if is_local_path(uri):
                yield local_path_from_uri(uri)
            else:
                with self.storage.open(uri) as stream:
                    yield stream
            return
        try:
            with self.acquire(uri, shard) as path:
                yield path
        except Exception:
            if self.strategy != "preferred":
                raise
            with self.storage.open(uri) as stream:
                yield stream

