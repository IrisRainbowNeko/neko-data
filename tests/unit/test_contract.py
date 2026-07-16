import json

from neko_data.contract import DatasetManifest, NormalizedSample, ShardRecord
from neko_data.contract.manifest import load_manifest, write_manifest


def test_sample_keeps_named_captions(sample_image_bytes):
    sample = NormalizedSample(
        sample_key="abc",
        image=sample_image_bytes,
        caption="a caption",
        captions={"tags": "one, two", "regular_summary": "a summary"},
        metadata={"id": "abc", "quality": 0.9},
        width=32,
        height=16,
    )
    assert sample.select_caption("tags") == "one, two"
    assert sample.select_caption("regular_summary") == "a summary"
    assert sample.metadata_payload()["width"] == 32
    assert json.loads(sample.metadata_json())["captions"]["tags"] == "one, two"


def test_manifest_round_trip(tmp_path):
    manifest = DatasetManifest(
        dataset_id="demo",
        version="v1",
        shards=[ShardRecord("wds/train-000000.tar", "train", 4, 12, "abc")],
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_manifest(path)
    assert loaded.dataset_id == "demo"
    assert loaded.splits == {"train": 4}
    assert loaded.shards[0].sha256 == "abc"

