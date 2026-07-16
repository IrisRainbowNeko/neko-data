"""Normalize HF rows and WebDataset members into one sample contract."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..contract.records import NormalizedSample

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp", "avif", "gif", "bmp", "tif", "tiff"})
DEFAULT_CAPTION_KEYS = ("caption", "text", "prompt", "annotation", "regular_summary", "tags")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip() or None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        values = [_text(item) for item in value]
        values = [item for item in values if item]
        return ", ".join(values) if values else None
    return None


def json_safe(value: Any) -> Any:
    """Keep metadata JSON-compatible while preserving scalar dataset values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def image_extension(name: str | None, image: bytes) -> str:
    if name:
        suffix = Path(name).suffix.lower().lstrip(".")
        if suffix in IMAGE_EXTENSIONS:
            return suffix
    try:
        with Image.open(io.BytesIO(image)) as opened:
            extension = (opened.format or "jpg").lower()
    except Exception:
        extension = "jpg"
    return "jpeg" if extension == "jpg" else extension


def image_size(image: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(io.BytesIO(image)) as opened:
            return opened.width, opened.height
    except Exception:
        return None, None


def _caption_fields(payload: Mapping[str, Any], configured: Iterable[str] | None = None) -> dict[str, str]:
    keys = tuple(configured or DEFAULT_CAPTION_KEYS)
    nested = payload.get("captions")
    fields: dict[str, str] = {}
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            caption = _text(value)
            if caption is not None:
                fields[str(key)] = caption
    for key in keys:
        caption = _text(payload.get(key))
        if caption is not None:
            fields[key] = caption
    return fields


def _merge_provider(
    captions: dict[str, str],
    metadata: dict[str, Any],
    provider_data: Mapping[str, Any] | None,
) -> None:
    if not provider_data:
        return
    for key, value in provider_data.items():
        key = str(key)
        caption = _text(value)
        if caption is not None and key in DEFAULT_CAPTION_KEYS:
            captions[key] = caption
        metadata[key] = json_safe(value)


def sample_from_wds_parts(
    sample_key: str,
    image_name: str,
    image: bytes,
    text: bytes | str | None = None,
    json_data: bytes | str | Mapping[str, Any] | None = None,
    *,
    split: str = "train",
    source_id: str = "webdataset",
    metadata_provider: Mapping[str, Any] | None = None,
    caption_keys: Iterable[str] | None = None,
) -> NormalizedSample:
    payload: dict[str, Any] = {}
    if json_data is not None:
        if isinstance(json_data, Mapping):
            payload = dict(json_data)
        else:
            raw = json_data.decode("utf-8", errors="replace") if isinstance(json_data, bytes) else json_data
            parsed = json.loads(raw)
            if isinstance(parsed, Mapping):
                payload = dict(parsed)
    captions = _caption_fields(payload, caption_keys)
    default_caption = _text(text) or _text(payload.get("caption")) or _text(payload.get("text"))
    if default_caption is not None:
        captions.setdefault("caption", default_caption)
    metadata = {key: json_safe(value) for key, value in payload.items() if key not in {"captions"}}
    _merge_provider(captions, metadata, metadata_provider)
    width = payload.get("width")
    height = payload.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        width, height = image_size(image)
    return NormalizedSample(
        sample_key=str(sample_key),
        image=image,
        image_extension=image_extension(image_name, image),
        split=split,
        source_id=source_id,
        caption=default_caption or captions.get("caption") or next(iter(captions.values()), None),
        captions=captions,
        metadata=metadata,
        width=width,
        height=height,
    )


def sample_from_mapping(
    row: Mapping[str, Any],
    *,
    image_column: str = "image",
    key_column: str | None = None,
    caption_columns: Iterable[str] | None = None,
    split: str = "train",
    source_id: str = "hf-dataset",
    metadata_provider: Mapping[str, Any] | None = None,
) -> NormalizedSample:
    image_value = row.get(image_column)
    image_name: str | None = None
    if isinstance(image_value, Image.Image):
        image_name = (image_value.format or "jpg").lower()
        output = io.BytesIO()
        image_value.save(output, format="JPEG" if image_name == "jpg" else image_name.upper())
        image = output.getvalue()
    elif isinstance(image_value, Mapping):
        image_name = image_value.get("path")
        image = image_value.get("bytes")
        if image is None and image_name:
            image = Path(image_name).read_bytes()
    elif isinstance(image_value, (bytes, bytearray, memoryview)):
        image = bytes(image_value)
    elif isinstance(image_value, (str, os.PathLike)):
        image_name = os.fspath(image_value)
        image = Path(image_name).read_bytes()
    else:
        raise TypeError(f"Unsupported image value in column {image_column!r}: {type(image_value).__name__}")

    if not isinstance(image, bytes) or not image:
        raise ValueError(f"Image column {image_column!r} did not contain bytes")
    caption_keys = tuple(caption_columns or DEFAULT_CAPTION_KEYS)
    captions = _caption_fields(row, caption_keys)
    default_caption = _text(row.get("caption")) or _text(row.get("text")) or _text(row.get("prompt"))
    metadata = {
        str(key): json_safe(value)
        for key, value in row.items()
        if key not in {image_column, *caption_keys, "caption", "text", "prompt"}
    }
    _merge_provider(captions, metadata, metadata_provider)
    width, height = image_size(image)
    key_value = row.get(key_column) if key_column else row.get("id") or row.get("key")
    if key_value is None:
        raise ValueError("A sample key is required: configure key_column or provide id/key")
    return NormalizedSample(
        sample_key=str(key_value),
        image=image,
        image_extension=image_extension(image_name, image),
        split=split,
        source_id=source_id,
        caption=default_caption or captions.get("caption") or next(iter(captions.values()), None),
        captions=captions,
        metadata=metadata,
        width=width,
        height=height,
    )

