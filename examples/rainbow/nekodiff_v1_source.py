"""Replace the local glob/DuckDB sources in NekoDiffusion with one manifest.

The manifest is built once with all caption variants joined into each sample.
Choose the variant explicitly for each logical training source.
"""

from neko_data.integrations.rainbow import RainbowTextImageSource
from neko_data.runtime import RuntimeContext


def full_source(manifest_uri: str, rank: int, world_size: int, cache_root: str):
    common = dict(
        runtime=RuntimeContext(global_rank=rank, world_size=world_size),
        cache_root=cache_root,
        cache_strategy="required",
        cache_max_size_bytes=420 * 1024**3,
        cache_evict_size_bytes=350 * 1024**3,
        prefetch_workers=4,
        prefetch_shards=4,
        sample_shuffle=512,
        prompt_template="{caption}",
    )
    return {
        "source_danbooru_tags": RainbowTextImageSource.from_manifest(
            manifest_uri, caption_key="tags", **common,
        ),
        "source_danbooru_nl": RainbowTextImageSource.from_manifest(
            manifest_uri, caption_key="regular_summary", **common,
        ),
    }

