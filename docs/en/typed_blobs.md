# Tensor trees and streamed model files

The existing manifest, tar/Parquet publisher, assigned-shard planner and
node-local cache also support non-image samples. No framework is required for
the core codecs. Existing `NormalizedSample` and image output modes are unchanged.

## Tensor trees

```python
from pathlib import Path
import numpy as np
from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import TypedSample
from neko_data.runtime import RuntimeContext, open_dataset

samples = [TypedSample(
    sample_key="episode-17-segment-0",
    data={
        "observations": [np.zeros((3, 8), np.float32), np.zeros((7, 8), np.float32)],
        "actions": (np.array([1, 4], np.int64), None),
        "legal": np.array([True, False]),
        "behavior_logp": np.array([-0.4, -1.2], np.float32),
    },
    metadata={"episode_id": 17, "policy_version": 23, "outcome": "win"},
)]
build_dataset(BuildJob(samples, Path("dataset"), "trajectory-example", "v1",
                       max_samples_per_shard=128, skip_invalid_samples=False))
view = open_dataset("dataset/manifest.json", RuntimeContext(), output_mode="typed")
for row in view:
    tree = row["data"]
```

Supported leaves are JSON scalars, bytes, numeric/bool NumPy arrays and scalars,
and `TensorValue(dtype, shape, data)`. Dicts, lists, tuples and `None` retain their
types. Ragged data uses lists/tuples of dense arrays; object arrays are rejected.
NumPy dtype (including endianness) and shape are preserved. Device, strides,
autograd graphs and storage alias relationships are not serialized.

For optional PyTorch conversion use
`neko_data.integrations.torch.from_torch_tree` before building and
`to_torch_tree(row["data"])` after reading. Tensors are detached/copied to CPU.
`bfloat16` is preserved as its original two-byte words, rather than converted to
float32. No torch import occurs in the tensor codec itself.

The `typed-npz-v1` payload is an NPZ with a tagged JSON tree and numeric NPY
arrays. It remains **uncompressed by default**, with exactly the original
`np.savez` byte path. Optional `TypedSample(..., compression_level=1)` or
`encode_typed_tree(tree, compression_level=1)` applies lossless Deflate to each
unchanged NPY member; integer levels 1 through 9 are supported. The default is 0
(stored). No new sample format or metadata fields are needed: the ZIP member
headers describe the codec, and existing readers support it. A reader returns
decoded data, without preserving the original encoding-level preference.

Loading always uses `allow_pickle=False`. The reader checks
archive names, duplicate entries, expanded size, NPY shape/byte counts, dtypes,
tree depth and references before returning data. Default payload limit is 1 GiB;
split large trajectories into independent segments. Use `BlobSample` for model
files instead of putting multi-GB checkpoints inside a tensor tree.
Compression does not relax the expanded-data limit. A sample whose original
arrays and tree exceed 1 GiB still requires splitting into complete segments.
Blob/model files remain streamed without compression. Compression is opt-in;
measure CPU time, encoded bytes and sustained publication speed on actual data
before choosing it for a training pipeline.

Versioned tensor/blob keys must be canonical relative names: no leading dot or
slash, backslashes, drive prefixes, NUL, empty path components, `.` or `..`.
Keys such as `map.alpha/episode-001` and Unicode names retain their exact spelling.
Their JSON sidecars are limited to 4 MiB at both build and read time. Oversized
JSON of an unknown sample type is temporarily spooled with bounded buffers so
legacy JSON-first image samples can retain their previous behavior.

## Model files without whole-file RAM copies

```python
from neko_data.contract import BlobSample
from neko_data.build.publisher import DatasetPublisher

artifacts = [BlobSample(
    "checkpoint-760",
    files={"checkpoint.pt": Path("ppo-00760.pt"),
           "config.json": Path("config.json")},
    metadata={"policy_version": 760, "lineage": "example"},
)]
build_dataset(BuildJob(
    artifacts, Path("model-staging"), "model-example", "v1",
    max_shard_size="512MiB", max_samples_per_shard=1,
    skip_invalid_samples=False,
    publisher=DatasetPublisher("s3://bucket/models/v1", delete_after_upload=True),
))

view = open_dataset(
    "s3://bucket/models/v1/manifest.json", output_mode="blob",
    cache_root="/local/cache/shards", cache_max_size_bytes=8 * 1024**3,
    cache_evict_size_bytes=6 * 1024**3,
    blob_cache_root="/local/cache/model-files",
    blob_cache_max_size_bytes=56 * 1024**3,
    blob_cache_evict_size_bytes=48 * 1024**3,
    prefetch_workers=0,
)
row = view.locate_blob("checkpoint-760", shard="wds/train/train-000000.tar")
with view.acquire_blob(row, "checkpoint.pt") as local_path:
    # Optional; the library never deserializes the checkpoint itself.
    import torch
    state = torch.load(local_path, map_location="cpu", mmap=True, weights_only=True)
    # Keep this context alive while mmap-backed tensors use the file.
    del state

# HF-style directories: field names can include safe relative subpaths.
with view.acquire_blobs(row) as directory:
    pass  # AutoModel.from_pretrained(directory), when the appropriate files exist
```

Build hashes each file incrementally and streams it into tar while independently
checking that the file has not changed. No source path is published. Tar members
are copied to cache with bounded buffers and checked by SHA-256 and byte size.
Every finalized shard is published immediately; `delete_after_upload` releases
that local tar and Parquet after successful uploads. A single sample may exceed
`max_shard_size`: rollover happens **between** samples. For large model files use
one artifact/sample per shard. `resume=True` reuses the existing completed-shard
journal; the source iterator must replay the same immutable prefix and order.

`locate_blob` opens only the explicitly named, rank/worker-assigned manifest
shard, so looking up the latest model does not enumerate/download old snapshots.
`list(view)` intentionally enumerates all assigned snapshots. The caller can use
manifest/Parquet metadata as its model catalog. Standard boto3 credentials,
endpoint and proxy configuration work unchanged with S3/R2.

The shard cache and extracted-file cache have **separate** high/low watermarks;
budget their sum (8 + 56 GiB above). The same lock/checksum/lease/LRU machinery
protects both. Active leases are never evicted, so simultaneous pins can exceed
the high watermark until released. Include the largest live pin set and one
incoming shard in disk capacity planning. `acquire_blobs` uses temporary hard
links on the file-cache filesystem; it does not duplicate large model files.
Treat returned cache files/directories as read-only. Do not modify them in place.
Verification markers bind the checksum to the exact size, modification/change
times, inode and device (including native Windows ChangeTime). Changed files and
old marker formats are rehashed; an accidental same-size rewrite with restored
mtime cannot silently remain a verified hit. A lease prevents eviction, not
concurrent modification by the file owner while it is in use.

## Wire compatibility and training boundaries

The top-level manifest remains `schema_version="1.0"`, format `webdataset`:
assignment, checksum and relative-path semantics have not changed. New samples
declare an independent versioned **sample format** in their JSON member:

* `typed-npz-v1`: `<key>.ntensor` plus `<key>.json` containing its SHA-256.
* `blob-files-v1`: `<key>.json` plus `<key>.nblob00000000`, etc. JSON maps each
  safe field name to the extension, size and SHA-256. Parquet retains this map in
  `blobs_json`; typed Parquet rows expose `payload_sha256` and `sample_format`.

Old image/text members are unchanged. New data requires a reader supporting its
sample format and an explicit `output_mode="typed"`/`"blob"`; mismatched modes
fail instead of pretending a tensor/model artifact is an image. Prefer separate
dataset manifests for image, typed and model samples.

Runtime epoch/shard/worker planning is deterministic and uses only assigned
shards. This is not a trainer checkpoint or a transparent mid-epoch DDP cursor:
the trainer must save its own epoch/sample progress and account for uneven rank
sample counts. For RL, these archives suit offline/AWR, diagnostics and replay.
They do **not** turn old trajectories into fresh on-policy PPO data. Preserve
behavior probabilities, masks, boundaries, policy versions and other algorithm
contracts in the tree; this library does not validate their RL meaning.
