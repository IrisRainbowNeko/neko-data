import io
import json
import tarfile

import pyarrow.parquet as pq
import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.build.metadata import DuckDBMetadataProvider
from neko_data.build.source_hf_webdataset import HFWebDatasetSource

duckdb = pytest.importorskip("duckdb")


def _write_tar(path, image):
    with tarfile.open(path, "w") as archive:
        for name, data in {
            "42.jpg": image,
            "42.json": json.dumps({"tags": "input tags"}).encode(),
            "42.txt": b"input caption\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_duckdb_join_is_materialized_in_tar_and_parquet(tmp_path, sample_image_bytes):
    database_path = tmp_path / "labels.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(
        "CREATE TABLE data (id BIGINT, tags VARCHAR, regular_summary VARCHAR, quality DOUBLE)"
    )
    connection.execute(
        "INSERT INTO data VALUES (?, ?, ?, ?)",
        [42, "joined tags", "joined summary", 0.95],
    )
    connection.close()

    input_tar = tmp_path / "input.tar"
    _write_tar(input_tar, sample_image_bytes)
    source = HFWebDatasetSource(
        input_files=[str(input_tar)],
        split="train",
        source_id="fixture",
        metadata_provider=DuckDBMetadataProvider(database_path),
    )
    root = tmp_path / "dataset"
    manifest = build_dataset(BuildJob(
        source=source,
        output_dir=root,
        dataset_id="duckdb",
        version="v1",
    ))

    assert manifest.splits == {"train": 1}
    with tarfile.open(root / "wds/train/train-000000.tar", "r") as archive:
        payload = json.loads(archive.extractfile("42.json").read())
    assert payload["captions"]["tags"] == "joined tags"
    assert payload["captions"]["regular_summary"] == "joined summary"
    assert payload["quality"] == 0.95

    table = pq.read_table(root / "metadata/train/train-000000.parquet")
    assert table.column("caption_tags").to_pylist() == ["joined tags"]
    assert table.column("caption_regular_summary").to_pylist() == ["joined summary"]
    assert table.column("quality").to_pylist() == [0.95]
