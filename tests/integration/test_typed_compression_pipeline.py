import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.build.shard_writer import WebDatasetShardWriter
from neko_data.contract import BlobSample, TypedSample
from neko_data.runtime import open_dataset


def assert_deflated_and_metadata(path):
    with tarfile.open(path) as archive:
        typed = next(member for member in archive if member.name.endswith('.ntensor'))
        payload = archive.extractfile(typed).read()
        metadata = json.load(archive.extractfile('episode.json'))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist())
    assert metadata['sample_format'] == 'typed-npz-v1'
    assert metadata['payload_extension'] == 'ntensor'
    return payload


def test_streaming_writer_typed_descriptor_roundtrips_and_blobs_stay_uncompressed(tmp_path):
    values = np.zeros((64, 128), np.int64)
    root = tmp_path / 'typed'
    manifest = build_dataset(BuildJob([TypedSample('episode', {'data': values}, compression_level=1)],
                                     root, 'compressed', 'v1', skip_invalid_samples=False))
    assert_deflated_and_metadata(root / manifest.shards[0].path)
    row = next(iter(open_dataset(root / 'manifest.json', output_mode='typed', prefetch_workers=0,
                                 cache_root=tmp_path / 'cache')))
    assert row['data']['data'].tobytes() == values.tobytes()
    assert pq.read_table(root / manifest.shards[0].metadata_path).to_pylist()[0]['sample_format'] == 'typed-npz-v1'
    blob = tmp_path / 'weights.bin'
    blob.write_bytes(b'unchanged model bytes' * 100)
    artifacts = tmp_path / 'models'
    models = build_dataset(BuildJob([BlobSample('model', {'weights': blob})], artifacts, 'models', 'v1'))
    with tarfile.open(artifacts / models.shards[0].path) as archive:
        member = archive.getmember('model.nblob00000000')
        assert archive.extractfile(member).read() == blob.read_bytes()


def test_nonstreaming_writer_honors_same_typed_descriptor(tmp_path):
    writer = WebDatasetShardWriter(tmp_path / 'wds', tmp_path / 'metadata')
    writer.add(TypedSample('episode', (np.zeros(2048, np.bool_), None), compression_level=1))
    records = writer.close()
    assert len(records) == 1
    assert_deflated_and_metadata(tmp_path / 'wds' / Path(records[0].path).name)


def test_default_descriptor_and_wire_metadata_remain_unchanged():
    value = {'data': np.zeros(2048, np.uint8)}
    plain = TypedSample('episode', value)
    compressed = TypedSample('episode', value, compression_level=1)
    assert plain.compression_level == 0
    assert set(plain.metadata_payload(plain.encode())) == set(compressed.metadata_payload(compressed.encode()))
    assert plain.metadata_payload(plain.encode())['sample_format'] == \
        compressed.metadata_payload(compressed.encode())['sample_format']
    with pytest.raises(TypeError):
        BlobSample('model', {}, compression_level=1)
