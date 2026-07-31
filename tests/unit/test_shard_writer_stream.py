from neko_data.build import StreamingWebDatasetShardWriter
from neko_data.contract import NormalizedSample


def test_streaming_writer_preserves_pattern_subdirectory_in_metadata_path(sample_image_bytes, tmp_path):
    writer = StreamingWebDatasetShardWriter(
        tmp_path / "wds" / "train",
        tmp_path / "metadata" / "train",
        pattern="worker-00/{split}-{index:06d}.tar",
    )
    writer.add(NormalizedSample(sample_key="sample", image=sample_image_bytes))

    records = writer.close()

    assert records[0].path == "wds/train/worker-00/train-000000.tar"
    assert records[0].metadata_path == "metadata/train/worker-00/train-000000.parquet"
    assert (tmp_path / records[0].metadata_path).is_file()
