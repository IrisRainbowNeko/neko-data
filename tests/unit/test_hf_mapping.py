from pathlib import Path

from neko_data.build.normalize import sample_from_mapping


def test_mapping_preserves_zero_id_and_resolves_relative_image(sample_image_bytes, tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(sample_image_bytes)
    sample = sample_from_mapping(
        {"id": 0, "image": {"path": "image.jpg", "bytes": None}, "caption": "zero"},
        image_root=tmp_path,
    )
    assert sample.sample_key == "0"
    assert sample.caption == "zero"
    assert sample.image == sample_image_bytes


def test_mapping_resolves_plain_relative_image(sample_image_bytes, tmp_path: Path):
    image_path = tmp_path / "plain.jpg"
    image_path.write_bytes(sample_image_bytes)
    sample = sample_from_mapping(
        {"id": "plain", "image": "plain.jpg", "caption": "plain path"},
        image_root=tmp_path,
    )
    assert sample.sample_key == "plain"
    assert sample.caption == "plain path"
    assert sample.image == sample_image_bytes
