import json
from pathlib import Path

import pytest

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract.records import NormalizedSample


def test_build_reports_invalid_source_rows_without_swallowing_infrastructure(tmp_path: Path, sample_image_bytes: bytes):
    def source():
        yield NormalizedSample("good-0", sample_image_bytes, caption="ok")
        raise ValueError("malformed source row")

    manifest = build_dataset(BuildJob(
        source=source(), output_dir=tmp_path / "dataset", dataset_id="errors", version="v1",
    ))
    assert manifest.splits == {"train": 1}
    errors = (tmp_path / "dataset/reports/errors.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(errors) == 1
    assert json.loads(errors[0])["error"] == "malformed source row"


def test_build_propagates_publisher_io_errors(tmp_path: Path, sample_image_bytes: bytes):
    class FailingPublisher:
        def publish_shard(self, record, tar_path, metadata_path):
            raise OSError("R2 unavailable")

    with pytest.raises(OSError, match="R2 unavailable"):
        build_dataset(BuildJob(
            source=[NormalizedSample("one", sample_image_bytes, caption="ok")],
            output_dir=tmp_path / "dataset",
            dataset_id="errors",
            version="v1",
            publisher=FailingPublisher(),
        ))


def test_resume_replaces_stale_partial_shard(tmp_path: Path, sample_image_bytes: bytes):
    root = tmp_path / "dataset"
    sample = NormalizedSample("one", sample_image_bytes, caption="ok")
    build_dataset(BuildJob(source=[sample], output_dir=root, dataset_id="resume", version="v1"))
    stale = root / "wds/train/.train-000001.tar.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")
    manifest = build_dataset(BuildJob(
        source=[sample, NormalizedSample("two", sample_image_bytes, caption="ok")],
        output_dir=root,
        dataset_id="resume",
        version="v1",
        resume=True,
    ))
    assert manifest.splits == {"train": 2}
