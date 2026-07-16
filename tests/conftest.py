from pathlib import Path

import pytest


@pytest.fixture
def sample_image_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (32, 16), (120, 80, 40)).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    return tmp_path / "dataset"

