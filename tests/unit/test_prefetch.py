from contextlib import contextmanager

from neko_data.runtime.prefetch import ShardPrefetcher


class Cache:
    def __init__(self):
        self.calls = []

    @contextmanager
    def acquire(self, uri, shard):
        self.calls.append(uri)
        yield


def test_prefetcher_reuses_capacity_after_completed_future():
    cache = Cache()
    prefetcher = ShardPrefetcher(cache, max_workers=1, max_pending=1)
    try:
        prefetcher.schedule("one", None)
        prefetcher.pending["one"].result(timeout=2)
        prefetcher.schedule("two", None)
        prefetcher.pending["two"].result(timeout=2)
        assert cache.calls == ["one", "two"]
    finally:
        prefetcher.close()
