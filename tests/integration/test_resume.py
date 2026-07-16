from pathlib import Path

from neko_data.build.job import BuildJob, build_dataset
from neko_data.contract.records import NormalizedSample


def test_build_resume_uses_completed_shard_journal(tmp_path: Path, sample_image_bytes: bytes):
    def samples():
        for index in range(4):
            yield NormalizedSample(
                sample_key=f"resume-{index}", image=sample_image_bytes, source_id="fixture",
                caption=f"caption {index}", width=32, height=16,
            )

    root = tmp_path / "resume-dataset"
    build_dataset(BuildJob(
        source=list(samples())[:2], output_dir=root, dataset_id="resume", version="v1",
        max_shard_size="1MiB", max_samples_per_shard=2,
    ))
    manifest = build_dataset(BuildJob(
        source=samples(), output_dir=root, dataset_id="resume", version="v1",
        max_shard_size="1MiB", max_samples_per_shard=2, resume=True,
    ))
    assert manifest.splits == {"train": 4}
    assert len(manifest.shards) == 2
    assert [shard.index for shard in manifest.shards] == [0, 1]

