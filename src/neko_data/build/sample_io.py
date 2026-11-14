"""Tar member serialization shared by both shard writers."""

from __future__ import annotations

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

from ..contract.blobs import BlobSample
from ..contract.records import NormalizedSample
from ..contract.sample_metadata import encode_sample_metadata
from ..contract.typed import TYPED_EXTENSION, TypedSample


@dataclass(frozen=True)
class FileMember:
    path: Path
    size: int
    sha256: str

    def __len__(self):
        return self.size


def sample_members(sample: NormalizedSample | TypedSample | BlobSample) -> tuple[list, dict | None]:
    if isinstance(sample, BlobSample):
        files, metadata = sample.prepare()
        encoded = encode_sample_metadata(metadata)
        members = [("json", encoded)] + [(ext, FileMember(path, size, digest)) for ext, path, size, digest in files]
        return members, metadata
    if isinstance(sample, TypedSample):
        payload = sample.encode()
        metadata = sample.metadata_payload(payload)
        return [(TYPED_EXTENSION, payload), ("json", encode_sample_metadata(metadata))], metadata
    if not isinstance(sample, NormalizedSample):
        raise TypeError("Shard writers require NormalizedSample, TypedSample or BlobSample")
    members = [(sample.image_extension, sample.image), ("json", sample.metadata_json())]
    if sample.caption:
        members.append(("txt", (sample.caption + "\n").encode("utf-8")))
    return members, None


def write_member(archive: tarfile.TarFile, name: str, data: bytes | FileMember) -> None:
    info = tarfile.TarInfo(name=name)
    info.size, info.mode = len(data), 0o644
    if isinstance(data, bytes):
        archive.addfile(info, io.BytesIO(data))
        return
    digest = hashlib.sha256()

    class HashingReader:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size):
            block = self.stream.read(size)
            digest.update(block)
            return block

    with data.path.open("rb") as stream:
        if data.path.stat().st_size != data.size:
            raise OSError("Blob source size changed before tar write")
        archive.addfile(info, HashingReader(stream))
        if stream.read(1) or digest.hexdigest() != data.sha256:
            # This must abort the shard rather than skip a partly written sample.
            raise OSError("Blob source changed during tar write")
