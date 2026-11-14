"""Pinned, streamed file extraction through the existing node-local cache."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..contract.blobs import BLOB_SAMPLE_FORMAT, BlobSampleReference, safe_blob_name
from ..contract.sample_metadata import MAX_SAMPLE_METADATA_BYTES
from ..contract.schema import ShardRecord
from .cache import DiskShardCache


@dataclass(frozen=True)
class LocatedBlobSample:
    sample: BlobSampleReference
    uri: str
    shard: ShardRecord

    @property
    def metadata(self):
        return self.sample.metadata


class _TarMemberStorage:
    def __init__(self, cache, located, field):
        self.cache, self.located, self.field = cache, located, field

    def download(self, _uri, destination):
        descriptor = self.located.sample.files[self.field]
        target = f"{self.located.sample.sample_key}.{descriptor['extension']}"
        # Keep the shard pinned until the member copy has completed. The blob
        # cache then checks both size and SHA before publishing the local file.
        with self.cache.open(self.located.uri, self.located.shard) as file_or_path:
            with ExitStack() as stack:
                stream = (stack.enter_context(Path(file_or_path).open("rb"))
                          if isinstance(file_or_path, (str, Path)) else file_or_path)
                archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r|*"))
                found = False
                for member in archive:
                    if member.name != target:
                        continue
                    if found or not member.isfile() or member.size != descriptor["size_bytes"]:
                        raise ValueError("Invalid or duplicate blob member")
                    found = True
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError("Blob member is not readable")
                    with extracted, Path(destination).open("wb") as output:
                        shutil.copyfileobj(extracted, output, length=8 * 1024**2)
                if not found:
                    raise KeyError(f"Missing blob member {target!r}")


def locate_blob(view, sample_key: str, shard: ShardRecord | str) -> dict:
    """Read one sample descriptor from an explicitly assigned shard.

    This avoids enumerating/downloading all historical model snapshots. A
    shard path comes from the manifest/Parquet index, never an arbitrary URI.
    """
    context = view._worker_context(view.runtime)
    assigned = view.planner.plan(view.epoch, context)
    matches = [item for item in assigned if item == shard or item.path == shard]
    if len(matches) != 1:
        raise ValueError("Blob shard must identify one shard assigned to this rank/worker")
    record = matches[0]
    uri = view.loaded.shard_uri(record)
    target = f"{sample_key}.json"
    with view.cache.open(uri, record) as file_or_path:
        with ExitStack() as stack:
            stream = (stack.enter_context(Path(file_or_path).open("rb"))
                      if isinstance(file_or_path, (str, Path)) else file_or_path)
            archive = stack.enter_context(tarfile.open(fileobj=stream, mode="r|*"))
            for member in archive:
                if member.name != target:
                    continue
                if not member.isfile() or member.size > MAX_SAMPLE_METADATA_BYTES:
                    raise ValueError("Invalid blob descriptor member")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("Blob descriptor is not readable")
                with extracted:
                    metadata = json.load(extracted)
                if metadata.get("sample_format") != BLOB_SAMPLE_FORMAT or metadata.get("sample_key") != sample_key:
                    raise ValueError("Invalid blob descriptor")
                reference = BlobSampleReference(sample_key, metadata["blobs"], metadata.get("split", view.split),
                                                metadata.get("source_id", uri), metadata)
                located = LocatedBlobSample(reference, uri, record)
                row = {"id": sample_key, "files": dict(reference.files), "_blob_reference": located}
                if view.include_metadata:
                    row["metadata"] = dict(metadata)
                return row
    raise KeyError(f"Blob sample {sample_key!r} is absent from shard {record.path!r}")


@contextmanager
def acquire_blob(view, row: Mapping, field: str):
    """Yield a checksum-verified local file while its cache lease is held.

    Pass a row produced by ``output_mode='blob'``. No torch/pickle loader runs.
    The path can be used with ``torch.load(..., mmap=True, weights_only=True)``.
    """
    located = row.get("_blob_reference")
    if not isinstance(located, LocatedBlobSample) or located.shard not in view.manifest.shards:
        raise ValueError("Blob row does not belong to this dataset manifest")
    if located.uri != view.loaded.shard_uri(located.shard):
        raise ValueError("Blob row source does not match its manifest shard")
    descriptor = located.sample.files[field]
    digest, size = descriptor["sha256"], descriptor["size_bytes"]
    record = ShardRecord(path=digest, split=located.shard.split, num_samples=1, size_bytes=size, sha256=digest)
    cache = DiskShardCache(view.blob_cache_root, _TarMemberStorage(view.cache, located, field),
                           max_size_bytes=view.blob_cache_max_size_bytes,
                           evict_size_bytes=view.blob_cache_evict_size_bytes, strategy="required")
    with cache.acquire(f"blob://{digest}", record) as path:
        yield path


@contextmanager
def acquire_blobs(view, row: Mapping, fields=None):
    """Yield a temporary HF-style directory, with named files pinned by leases.

    Hard links avoid duplicating multi-GB files; all links and the temporary
    directory are removed on exit. Never pass this directory to a mutating tool.
    """
    names = list(row["files"] if fields is None else fields)
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one distinct blob field")
    names = [safe_blob_name(name) for name in names]
    root = Path(view.blob_cache_root) / "views"
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pinned-", dir=root)))
        for name in names:
            source = stack.enter_context(acquire_blob(view, row, name))
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        yield directory
