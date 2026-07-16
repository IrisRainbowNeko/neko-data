"""Neko data contracts, streaming builders, and training readers."""

from .build.job import BuildJob, build_dataset
from .build.metadata import DuckDBMetadataProvider
from .build.source_hf_dataset import HFDatasetsSource
from .build.source_hf_webdataset import HFWebDatasetSource
from .contract import DatasetManifest, NormalizedSample, ShardRecord
from .integrations.rainbow import RainbowTextImageSource, RainbowWebDatasetSource
from .runtime import DatasetView, RuntimeContext, open_dataset

__all__ = [
    "DatasetManifest",
    "BuildJob",
    "DuckDBMetadataProvider",
    "HFDatasetsSource",
    "HFWebDatasetSource",
    "RainbowTextImageSource",
    "RainbowWebDatasetSource",
    "build_dataset",
    "DatasetView",
    "NormalizedSample",
    "RuntimeContext",
    "ShardRecord",
    "open_dataset",
]

