"""Iterable training view over manifest-listed WebDataset shards."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError:
    class IterableDataset:
        pass

    def get_worker_info():
        return None

from ..contract.schema import ShardRecord
from .cache import DiskShardCache
from .manifest_loader import LoadedManifest, ManifestLoader
from .planner import RuntimeContext, ShardPlanner
from .prefetch import ShardPrefetcher
from .reader import iter_shard_samples


def _shuffle_stream(items: Iterable, buffer_size: int, rng: random.Random) -> Iterator:
    iterator = iter(items)
    try:
        if buffer_size <= 1:
            yield from iterator
            return
        buffer = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) < buffer_size:
                continue
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def _group_runs(samples: Iterable, group_key: str, chunk_size: int = 0) -> Iterator[list]:
    """Collect consecutive samples sharing ``metadata[group_key]`` into lists.

    Builds written with the same ``group_key`` never split a group across
    shards, so consecutive runs are complete groups.  ``chunk_size`` caps a
    run so that very large groups become several independently shuffled units.
    """
    iterator = iter(samples)
    marker = object()
    current: list = []
    current_group = marker
    try:
        for sample in iterator:
            group = sample.metadata.get(group_key, marker)
            if current and (group != current_group or group is marker
                            or (chunk_size > 0 and len(current) >= chunk_size)):
                yield current
                current = []
            current.append(sample)
            current_group = group
        if current:
            yield current
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def _flatten_groups(groups: Iterable[list], rng: random.Random | None) -> Iterator:
    iterator = iter(groups)
    try:
        for group in iterator:
            if rng is not None:
                rng.shuffle(group)
            yield from group
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


class DatasetView(IterableDataset):
    """A deterministic, cache-backed training dataset.

    It yields the structure expected by RainbowNeko and HCP-Diffusion:
    ``id``, raw image bytes, a prompt mapping, and complete metadata.

    ``output_mode="image_label"`` yields ``id``, raw image bytes and
    ``metadata[label_key]`` as ``label``.  When ``group_key`` is set, samples
    are shuffled per group (for example one character class) instead of per
    sample, so members of a group are emitted consecutively.  This keeps
    class-balanced buckets fed in streaming mode.
    """

    def __init__(
        self,
        loaded: LoadedManifest,
        runtime: RuntimeContext | None = None,
        *,
        split: str = "train",
        cache: DiskShardCache | None = None,
        cache_root: str | Path | None = None,
        storage=None,
        cache_max_size_bytes: int = 420 * 1024**3,
        cache_evict_size_bytes: int = 350 * 1024**3,
        cache_strategy: str = "required",
        seed: int = 42,
        sample_shuffle: int = 0,
        prefetch_workers: int = 4,
        prefetch_shards: int = 4,
        caption_key: str = "caption",
        prompt_template: str | Sequence[str] | None = None,
        include_metadata: bool = True,
        output_mode: str = "text_image",
        label_key: str = "label",
        group_key: str | None = None,
        group_shuffle: int = 0,
        group_chunk_size: int = 0,
        shuffle_within_group: bool = True,
    ) -> None:
        self.loaded = loaded
        self.manifest = loaded.manifest
        self.runtime = runtime or RuntimeContext()
        self.split = split
        self.storage = storage
        self.cache = cache
        if self.cache is None:
            if cache_root is None:
                cache_root = Path.home() / ".cache" / "neko-data" / self.manifest.dataset_id
            if storage is None:
                from ..storage import HTTPStorage, LocalStorage, S3Storage, is_local_path
                if is_local_path(loaded.base_uri):
                    storage = LocalStorage()
                elif loaded.base_uri.startswith(("http://", "https://")):
                    storage = HTTPStorage()
                else:
                    storage = S3Storage()
            self.cache = DiskShardCache(
                cache_root,
                storage,
                max_size_bytes=cache_max_size_bytes,
                evict_size_bytes=cache_evict_size_bytes,
                strategy=cache_strategy,
            )
        self.seed = seed
        self.sample_shuffle = sample_shuffle
        self.prefetch_workers = prefetch_workers
        self.prefetch_shards = prefetch_shards
        self.caption_key = caption_key
        if isinstance(prompt_template, str) and Path(prompt_template).is_file():
            prompt_template = [
                line.strip()
                for line in Path(prompt_template).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.prompt_template = prompt_template
        self.include_metadata = include_metadata
        if output_mode not in {"text_image", "image", "image_label"}:
            raise ValueError("output_mode must be text_image, image or image_label")
        if group_key is not None and sample_shuffle > 1:
            raise ValueError("sample_shuffle breaks group order; use group_shuffle with group_key")
        self.output_mode = output_mode
        self.label_key = label_key
        self.group_key = group_key
        self.group_shuffle = group_shuffle
        self.group_chunk_size = group_chunk_size
        self.shuffle_within_group = shuffle_within_group
        self.epoch = 0
        self.planner = ShardPlanner(self.manifest.for_split(split), seed=seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(shard.num_samples for shard in self.manifest.for_split(self.split))

    @staticmethod
    def _worker_context(runtime: RuntimeContext) -> RuntimeContext:
        info = get_worker_info()
        if info is None:
            return runtime
        return runtime.for_worker(info.id, info.num_workers)

    def _raw_samples(self, shards: list[ShardRecord], prefetcher: ShardPrefetcher | None) -> Iterator:
        for index, shard in enumerate(shards):
            uri = self.loaded.shard_uri(shard)
            for future_shard in shards[index + 1:index + 1 + self.prefetch_shards]:
                if prefetcher is not None:
                    prefetcher.schedule(self.loaded.shard_uri(future_shard), future_shard)
            with self.cache.open(uri, shard) as file_or_path:
                yield from iter_shard_samples(file_or_path, split=self.split, source_id=uri)

    def __iter__(self) -> Iterator[dict]:
        context = self._worker_context(self.runtime)
        shards = self.planner.plan(self.epoch, context)
        prefetcher = None
        if self.prefetch_workers > 0 and self.cache.strategy != "disabled":
            prefetcher = ShardPrefetcher(self.cache, max_workers=self.prefetch_workers,
                                         max_pending=max(1, self.prefetch_shards * 2))
        samples = None
        try:
            samples = self._raw_samples(shards, prefetcher)
            shuffle_rng = random.Random(self.seed + self.epoch)
            if self.group_key is not None:
                groups = _group_runs(samples, self.group_key, self.group_chunk_size)
                groups = _shuffle_stream(groups, self.group_shuffle, shuffle_rng)
                samples = _flatten_groups(groups, shuffle_rng if self.shuffle_within_group else None)
            else:
                samples = _shuffle_stream(samples, self.sample_shuffle, shuffle_rng)
            rng = random.Random(self.seed + self.epoch)
            for sample in samples:
                if self.output_mode == "image":
                    yield {"id": sample.sample_key, "image": sample.image}
                    continue
                if self.output_mode == "image_label":
                    if self.label_key not in sample.metadata:
                        raise KeyError(f"Sample {sample.sample_key!r} has no label field {self.label_key!r}")
                    yield {"id": sample.sample_key, "image": sample.image,
                           "label": sample.metadata[self.label_key]}
                    continue
                yield sample.to_training_dict(
                    caption_key=self.caption_key,
                    prompt_template=self.prompt_template,
                    include_metadata=self.include_metadata,
                    rng=rng,
                )
        finally:
            close = getattr(samples, "close", None)
            if close is not None:
                close()
            if prefetcher is not None:
                prefetcher.close()

    def get_image_size(self, data: dict) -> tuple[int, int]:
        metadata = data.get("metadata", {})
        width, height = metadata.get("width"), metadata.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return width, height
        import io

        from PIL import Image
        with Image.open(io.BytesIO(data["image"])) as image:
            return image.size


def open_dataset(
    manifest_uri: str | Path,
    runtime: RuntimeContext | None = None,
    **kwargs,
) -> DatasetView:
    loaded = ManifestLoader(storage=kwargs.pop("storage", None)).load(
        manifest_uri,
        allow_duplicate_metadata_paths=kwargs.pop("allow_duplicate_metadata_paths", False),
    )
    return DatasetView(loaded, runtime=runtime, **kwargs)
