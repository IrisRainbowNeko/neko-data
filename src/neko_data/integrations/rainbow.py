"""Optional RainbowNeko Engine adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime import DatasetView, RuntimeContext, open_dataset

try:
    from rainbowneko.data.source.base import DataSource as _RainbowDataSource
except ImportError:
    class _RainbowDataSource:
        def __init__(self, repeat=1, **kwargs):
            self.repeat = repeat


class RainbowWebDatasetSource(_RainbowDataSource):
    """Expose a :class:`DatasetView` as RainbowNeko's iterable DataSource."""

    def __init__(self, dataset: DatasetView, repeat: int = 1, **kwargs) -> None:
        super().__init__(repeat=repeat, **kwargs)
        self.dataset = dataset
        self.size = len(dataset)

    def __getitem__(self, index) -> dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} is not indexable")

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return self.size * self.repeat

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def get_image_size(self, data: dict[str, Any]) -> tuple[int, int]:
        return self.dataset.get_image_size(data)


class RainbowTextImageSource(RainbowWebDatasetSource):
    """Named adapter matching the existing HCP text-image source terminology."""

    @classmethod
    def from_manifest(
        cls,
        manifest_uri: str | Path,
        runtime: RuntimeContext | None = None,
        **kwargs,
    ) -> "RainbowTextImageSource":
        return cls(open_dataset(manifest_uri, runtime=runtime, **kwargs))

