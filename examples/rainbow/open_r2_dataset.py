"""Minimal RainbowNeko adapter example.

Run this from an environment where both neko-data and RainbowNekoEngine are
installed.  Pass the resulting source to RainbowNeko's existing WebDataset
or NekoDiffWebDataset configuration.
"""

from neko_data.integrations.rainbow import RainbowTextImageSource
from neko_data.runtime import RuntimeContext


def make_source(manifest_uri: str, rank: int, world_size: int, cache_root: str):
    return RainbowTextImageSource.from_manifest(
        manifest_uri,
        runtime=RuntimeContext(global_rank=rank, world_size=world_size),
        cache_root=cache_root,
        cache_strategy="required",
        cache_max_size_bytes=420 * 1024**3,
        cache_evict_size_bytes=350 * 1024**3,
        prefetch_workers=4,
        prefetch_shards=4,
        sample_shuffle=512,
        caption_key="caption",
        prompt_template="{}",
    )

