"""Optional RainbowNeko Engine adapters."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from ..runtime import DatasetView, RuntimeContext, open_dataset
from ..storage import S3Storage

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


class RainbowWebDatasetImageSource(RainbowWebDatasetSource):
    """Manifest-backed equivalent of RainbowNeko's WebDatasetImageSource."""

    def __iter__(self):
        for data in self.dataset:
            image = Image.open(BytesIO(data["image"]))
            yield {"id": data["id"], "image": image}

    def get_image_size(self, data: dict[str, Any]) -> tuple[int, int]:
        return data["image"].size

    @classmethod
    def from_manifest(
        cls,
        manifest_uri: str | Path,
        runtime: RuntimeContext | None = None,
        *,
        repeat: int = 1,
        endpoint_url: str | None = None,
        region_name: str = "auto",
        **kwargs,
    ) -> "RainbowWebDatasetImageSource":
        kwargs.update({"output_mode": "image", "include_metadata": False})
        if endpoint_url is not None and "storage" not in kwargs:
            kwargs["storage"] = S3Storage(endpoint_url=endpoint_url, region_name=region_name)
        context = runtime or RuntimeContext.from_env()
        return cls(
            open_dataset(manifest_uri, runtime=context, **kwargs),
            repeat=repeat,
        )


class RainbowLabeledImageSource(RainbowWebDatasetSource):
    """Image + integer label source for classification/contrastive training.

    Yields ``{"id", "image": PIL.Image, "label"}``.  Combine with a grouped
    build (``group_key``) and ``group_key``/``group_shuffle`` here so that
    RainbowNeko's streaming ``PosNegBucket``/``CategoryBucket`` receive
    several samples of the same class within their buffers.
    """

    def __iter__(self):
        for data in self.dataset:
            yield {"id": data["id"], "image": Image.open(BytesIO(data["image"])), "label": data["label"]}

    def get_image_size(self, data: dict[str, Any]) -> tuple[int, int]:
        return data["image"].size

    @property
    def num_classes(self) -> int | None:
        value = self.dataset.manifest.metadata.get("num_classes")
        return int(value) if value is not None else None

    @classmethod
    def from_manifest(
        cls,
        manifest_uri: str | Path,
        runtime: RuntimeContext | None = None,
        *,
        repeat: int = 1,
        label_key: str = "label",
        group_key: str | None = "label",
        group_shuffle: int = 2048,
        endpoint_url: str | None = None,
        region_name: str = "auto",
        **kwargs,
    ) -> "RainbowLabeledImageSource":
        kwargs.update({
            "output_mode": "image_label",
            "include_metadata": False,
            "label_key": label_key,
            "group_key": group_key,
            "group_shuffle": group_shuffle,
        })
        if endpoint_url is not None and "storage" not in kwargs:
            kwargs["storage"] = S3Storage(endpoint_url=endpoint_url, region_name=region_name)
        context = runtime or RuntimeContext.from_env()
        return cls(open_dataset(manifest_uri, runtime=context, **kwargs), repeat=repeat)
