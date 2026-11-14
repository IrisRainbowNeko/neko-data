"""Bounded JSON sidecars for versioned tensor and blob samples."""

from __future__ import annotations

import json
from typing import Mapping

MAX_SAMPLE_METADATA_BYTES = 4 * 1024**2


def encode_sample_metadata(metadata: Mapping) -> bytes:
    payload = json.dumps(metadata, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_SAMPLE_METADATA_BYTES:
        raise ValueError("Versioned sample metadata exceeds maximum descriptor size")
    return payload
