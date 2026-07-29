# neko-data

`neko-data` is a small, framework-neutral data plane for diffusion training:

```text
HuggingFace streaming / Hub WebDataset tar / DuckDB
    -> normalized samples
    -> WebDataset tar + Parquet metadata
    -> local filesystem or S3-compatible storage (Cloudflare R2)
    -> manifest planner + node-local shard cache
    -> RainbowNeko / HCP-Diffusion training
```

The primary design constraint is a training machine with limited local disk.
Input WebDataset tars are read with a non-seekable tar stream, and regular
HuggingFace datasets use `streaming=True`; neither path downloads the whole
source dataset. During training, only assigned shards are cached locally.

See [README_zh-CN.md](README_zh-CN.md) for the Chinese guide and
[docs/zh_CN/index.md](docs/zh_CN/index.md) for the detailed data layout.

## Install

```bash
pip install -e '.[hf,r2,duckdb,rainbow]'
```

`hf`, `r2`, and `rainbow` are optional. Local WebDataset reading and Parquet
metadata do not require boto3 or the `webdataset` Python package.
Public HTTP manifests and shards use `HTTPStorage` automatically; the `hf`
extra provides its `requests` dependency.

## Build

```bash
neko-data build --config examples/build/hf_webdataset.yaml
```

Use `publish.destination: s3://bucket/prefix` with an R2 endpoint configured
through the normal boto3 environment variables. `delete_after_upload: true`
lets a build host upload each finalized shard and release its local copy.

## Mirror prebuilt shards

Use `neko-data mirror --config mirror.yaml` to copy existing Hub WebDataset tar
files without repacking them. The command is a read-only plan unless `--execute`
is supplied; completed objects are resumable and checksum-verified. See
[docs/en/mirroring.md](docs/en/mirroring.md).

## Train

```python
from neko_data.integrations.rainbow import RainbowTextImageSource
from neko_data.runtime import RuntimeContext

source = RainbowTextImageSource.from_manifest(
    "s3://bucket/datasets/example/v1/manifest.json",
    runtime=RuntimeContext(global_rank=rank, world_size=world_size),
    cache_root="/local/nvme/neko-data-cache",
    cache_strategy="required",
    sample_shuffle=512,
    caption_key="caption",
    prompt_template="{caption}",
)
```

The adapter returns the existing RainbowNeko shape:

```python
{
    "id": sample_key,
    "image": image_bytes,
    "prompt": {"template": "{caption}", "caption": caption},
    "metadata": {...},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).

