"""Canonical keys for versioned tensor and file samples."""

from __future__ import annotations


def validate_sample_key(value: str) -> None:
    """Reject keys changed by tar naming or the legacy reader's normalization.

    Image samples retain their existing naming contract. Versioned samples
    carry an explicit key in metadata, so its spelling must survive unchanged.
    """
    if (not isinstance(value, str) or not value or value.startswith((".", "/"))
            or "\\" in value or ":" in value or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("sample_key must be a canonical, safe relative key")
