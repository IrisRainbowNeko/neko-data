"""Bounded asynchronous shard prefetch."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor


class ShardPrefetcher:
    def __init__(self, cache, max_workers: int = 4, max_pending: int = 8) -> None:
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="neko-shard")
        self.max_pending = max_pending
        self.pending: dict[str, Future] = {}

    def schedule(self, uri: str, shard) -> None:
        if len(self.pending) >= self.max_pending or uri in self.pending:
            return
        self.pending[uri] = self.executor.submit(self._ensure, uri, shard)

    def _ensure(self, uri: str, shard) -> None:
        with self.cache.acquire(uri, shard):
            return

    def close(self) -> None:
        for future in self.pending.values():
            try:
                future.result()
            except Exception:
                pass
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.pending.clear()

