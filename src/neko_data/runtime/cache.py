"""Node-local content-addressed shard cache with leases and high-water eviction."""

from __future__ import annotations

import hashlib
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

from ..contract.schema import ShardRecord
from ..storage import is_local_path, local_path_from_uri

_STALE_PROCESS_FILE_SECONDS = 24 * 60 * 60
_STALE_CLEANUP_INTERVAL_SECONDS = 60 * 60


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
        self.eviction_lock = self.root / ".eviction.lock"
        self.cleanup_stamp = self.root / ".stale-cleanup.stamp"
        self.hostname = socket.gethostname().replace(".", "_")
        self.storage = storage
        self.max_size_bytes = max_size_bytes
        self.evict_size_bytes = min(evict_size_bytes, max_size_bytes)
        self.strategy = strategy
        self.stats = CacheStats()
        self.objects.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale()

    @staticmethod
    def _lock(path: Path):
        return FileLock(str(path) + ".lock")

    def _key(self, uri: str, shard: ShardRecord | None) -> str:
        if shard is not None and shard.sha256:
            return shard.sha256
        return hashlib.sha256(uri.encode("utf-8")).hexdigest()

    def _verified_marker(self, path: Path) -> Path:
        return path.with_name(f".{path.name}.verified")

    def _write_verified_marker(self, path: Path, digest: str) -> None:
        """Write a marker whose mtime is never older than the cache object."""
        marker = self._verified_marker(path)
        marker.write_text(digest + "\n", encoding="ascii")
        # Some distributed filesystems can assign the marker an mtime a few
        # nanoseconds before the object despite the write ordering. Aligning
        # both timestamps keeps the marker check deterministic.
        mtime_ns = path.stat().st_mtime_ns
        os.utime(marker, ns=(mtime_ns, mtime_ns))

    def _valid(self, path: Path, shard: ShardRecord | None) -> bool:
        if not path.exists() or not path.is_file():
            return False
        stat = path.stat()
        if shard is not None and shard.size_bytes and stat.st_size != shard.size_bytes:
            return False
        if shard is None or not shard.sha256:
            return True
        marker = self._verified_marker(path)
        if (
            marker.exists()
            and marker.read_text(encoding="ascii").strip() == shard.sha256
            # Allow the small timestamp skew seen on the shared filesystem;
            # the marker is still keyed by the expected content digest.
            and marker.stat().st_mtime_ns + 1_000_000 >= stat.st_mtime_ns
        ):
            return True
        if _file_sha256(path) != shard.sha256:
            return False
        self._write_verified_marker(path, shard.sha256)
        return True

    def _lease(self, path: Path) -> Path:
        lease = path.with_name(
            f".{path.name}.lease.{self.hostname}.{os.getpid()}.{uuid.uuid4().hex}"
        )
        lease.touch()
        return lease

    def _has_lease(self, path: Path) -> bool:
        return any(self.objects.glob(f".{path.name}.lease.*"))

    @staticmethod
    def _is_cache_object(path: Path) -> bool:
        return (
            path.is_file()
            and not path.name.startswith(".")
            and not path.name.endswith(".lock")
        )

    @staticmethod
    def _leased_object_names(paths: list[Path]) -> set[str]:
        leased = set()
        for path in paths:
            if not path.name.startswith(".") or ".lease." not in path.name:
                continue
            object_name, _, _ = path.name[1:].partition(".lease.")
            if object_name:
                leased.add(object_name)
        return leased

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _process_file_is_stale(self, path: Path, now: float) -> bool:
        name = path.name
        age = max(0.0, now - path.stat().st_mtime)
        if ".partial." in name:
            owner = name.partition(".partial.")[2].split(".", 1)[0]
            try:
                pid = int(owner)
            except ValueError:
                return age >= _STALE_PROCESS_FILE_SECONDS
            return not self._pid_is_alive(pid)
        if ".lease." in name:
            owner = name.partition(".lease.")[2].split(".")
            if len(owner) < 2 or owner[0] != self.hostname:
                return age >= _STALE_PROCESS_FILE_SECONDS
            try:
                pid = int(owner[1])
            except ValueError:
                return age >= _STALE_PROCESS_FILE_SECONDS
            return not self._pid_is_alive(pid)
        return False

    def _cleanup_stale_locked(self, paths: list[Path] | None = None) -> int:
        paths = list(self.objects.iterdir()) if paths is None else paths
        now = time.time()
        removed = 0
        for path in paths:
            try:
                if self._process_file_is_stale(path, now):
                    path.unlink(missing_ok=True)
                    removed += 1
                    continue
                if path.name.startswith(".") and path.name.endswith(".verified"):
                    object_name = path.name[1:-len(".verified")]
                    if not (self.objects / object_name).is_file():
                        path.unlink(missing_ok=True)
                        removed += 1
            except FileNotFoundError:
                continue
        return removed

    def cleanup_stale_process_files(self, *, timeout: float = 60.0) -> int:
        """Remove files owned by dead local processes under the cache lock.

        Unlike the periodic initialization cleanup, this method deliberately
        bypasses the one-hour rate limit. Launchers call it after DataLoader
        processes have exited, when their generator finalizers may not have had
        an opportunity to release every prefetched-shard lease.
        """
        lock = self._lock(self.eviction_lock)
        with lock.acquire(timeout=timeout):
            removed = self._cleanup_stale_locked()
            self.cleanup_stamp.touch()
        return removed

    def _cleanup_stale(self) -> None:
        lock = self._lock(self.eviction_lock)
        try:
            with lock.acquire(timeout=0):
                now = time.time()
                if (
                    self.cleanup_stamp.exists()
                    and now - self.cleanup_stamp.stat().st_mtime
                    < _STALE_CLEANUP_INTERVAL_SECONDS
                ):
                    return
                self._cleanup_stale_locked()
                self.cleanup_stamp.touch()
        except Timeout:
            return

    def _evict_locked(self) -> None:
        # A single directory census replaces the former per-object glob. The old
        # implementation became quadratic and all ranks repeated it concurrently.
        paths = list(self.objects.iterdir())
        self._cleanup_stale_locked(paths)
        paths = [path for path in paths if path.exists()]
        entries = [path for path in paths if self._is_cache_object(path)]
        total = sum(path.stat().st_size for path in entries)
        if total <= self.max_size_bytes:
            return
        leased = self._leased_object_names(paths)
        for path in sorted(entries, key=lambda item: item.stat().st_atime):
            if total <= self.evict_size_bytes:
                break
            if path.name in leased:
                continue
            with self._lock(path):
                # A lease may have appeared after the census while this process
                # waited for the object lock, so recheck before unlinking.
                if self._has_lease(path):
                    continue
                try:
                    size = path.stat().st_size
                    path.unlink()
                    self._verified_marker(path).unlink(missing_ok=True)
                    total -= size
                except FileNotFoundError:
                    continue

    def _evict(self) -> None:
        lock = self._lock(self.eviction_lock)
        try:
            with lock.acquire(timeout=0):
                self._evict_locked()
        except Timeout:
            # Another process on the node is already bringing the shared cache
            # below its low-water mark. A later download will retry if needed.
            return

    def _download(self, uri: str, path: Path, shard: ShardRecord | None) -> None:
        partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
        try:
            self.storage.download(uri, partial)
            if not self._valid(partial, shard):
                raise IOError(f"Downloaded shard failed size/checksum validation: {uri}")
            os.replace(partial, path)
            if shard is not None and shard.sha256:
                self._write_verified_marker(path, shard.sha256)
            self.stats.downloads += 1
            self.stats.bytes_downloaded += path.stat().st_size
        finally:
            partial.unlink(missing_ok=True)
            self._verified_marker(partial).unlink(missing_ok=True)

    @contextmanager
    def acquire(self, uri: str, shard: ShardRecord | None = None) -> Iterator[Path]:
        if is_local_path(uri):
            yield local_path_from_uri(uri)
            return
        key = self._key(uri, shard)
        path = self.objects / f"{key}.tar"
        lock = self._lock(path)
        downloaded = False
        if getattr(lock, "is_locked", False):
            self.stats.waits += 1
        with lock:
            if self._valid(path, shard):
                self.stats.hits += 1
            else:
                try:
                    self._download(uri, path, shard)
                    downloaded = True
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
            if downloaded:
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
