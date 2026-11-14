import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract import TypedSample
from neko_data.runtime import RuntimeContext, open_dataset
from neko_data.runtime.cache import DiskShardCache


def samples(count):
    return [TypedSample(f"trajectory-{i}", {"observations": [np.zeros((n, 3), dtype=np.float32) for n in (0, i)],
                                            "actions": (np.array([i], dtype=np.int64), None)},
                        metadata={"episode_id": i, "policy_version": 17, "outcome": "win"}) for i in range(count)]


def test_build_resume_and_distributed_assignment(tmp_path):
    root = tmp_path / "dataset"
    build_dataset(BuildJob(samples(2), root, "rl", "v1", max_samples_per_shard=1, skip_invalid_samples=False))
    manifest = build_dataset(BuildJob(samples(8), root, "rl", "v1", max_samples_per_shard=1,
                                     skip_invalid_samples=False, resume=True))
    assert len(manifest.shards) == 8
    assert manifest.metadata["resumed_samples"] == 2
    rows = pq.read_table(root / manifest.shards[0].metadata_path).to_pylist()
    assert rows[0]["sample_format"] == "typed-npz-v1"
    partitions = []
    for rank in range(2):
        for worker in range(2):
            view = open_dataset(root / "manifest.json", RuntimeContext(world_size=2, global_rank=rank,
                                worker_id=worker, num_workers=2), output_mode="typed", prefetch_workers=0)
            actual = list(view)
            partitions.append({row["id"] for row in actual})
            assert all(isinstance(row["data"]["actions"], tuple) for row in actual)
    assert sum(map(len, partitions)) == 8
    assert len(set.union(*partitions)) == 8


def test_remote_cache_reads_only_assigned_shards_and_releases_lease(tmp_path):
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob(samples(4), root, "rl", "v1", max_samples_per_shard=1))
    calls = []

    class FakeRemote:
        def download(self, uri, destination):
            calls.append(uri)
            source = root / uri.removeprefix("s3://fixture/")
            Path(destination).write_bytes(source.read_bytes())

    cache = DiskShardCache(tmp_path / "cache", FakeRemote(), max_size_bytes=1024**2)
    view = open_dataset(root / "manifest.json", RuntimeContext(world_size=2, global_rank=1),
                        output_mode="typed", cache=cache, prefetch_workers=0)
    view.loaded = replace(view.loaded, base_uri="s3://fixture")
    rows = list(view)
    assert len(rows) == 2
    assert len(calls) == 2
    assert not list(cache.objects.glob("*.lease.*"))
    for shard in manifest.shards:
        assert hashlib.sha256((root / shard.path).read_bytes()).hexdigest() == shard.sha256


def test_typed_payload_corruption_fails_closed(tmp_path):
    import io
    import tarfile

    from neko_data.runtime.reader import iter_shard_samples

    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob(samples(1), root, "rl", "v1"))
    content = io.BytesIO()
    with tarfile.open(root / manifest.shards[0].path) as source, tarfile.open(fileobj=content, mode="w") as dest:
        for member in source:
            data = source.extractfile(member).read()
            if member.name.endswith(".ntensor"):
                data = data[:-1] + bytes([data[-1] ^ 1])
            dest.addfile(member, io.BytesIO(data))
    content.seek(0)
    with pytest.raises(ValueError, match="checksum"):
        list(iter_shard_samples(content, split="train", source_id="fixture"))
