"""The canonical sample representation used by builders and readers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _json_default(value: Any) -> Any:
    """Convert common dataset values into JSON without hiding unsupported objects."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default, separators=(",", ":"))


@dataclass(frozen=True)
class NormalizedSample:
    """One image-text example before it is written to a WebDataset shard.

    ``captions`` keeps all named caption variants, while ``caption`` is the
    default field selected by a training source.  ``metadata`` is deliberately
    kept separate so analysis fields can be retained without becoming prompt
    fields by accident.
    """

    sample_key: str
    image: bytes
    image_extension: str = "jpg"
    split: str = "train"
    source_id: str = "unknown"
    caption: str | None = None
    captions: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if not self.sample_key or self.sample_key in {".", ".."}:
            raise ValueError("sample_key must be a non-empty value")
        if not isinstance(self.image, bytes) or not self.image:
            raise ValueError("image must contain non-empty bytes")
        extension = self.image_extension.lower().lstrip(".")
        object.__setattr__(self, "image_extension", extension or "jpg")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive when provided")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive when provided")

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image).hexdigest()

    @property
    def image_size(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    def select_caption(self, key: str = "caption") -> str | None:
        """Select a named caption without falling back to an arbitrary field."""
        if key in {"caption", "default", "text"}:
            return self.caption
        value = self.captions.get(key)
        if value is not None:
            return value
        value = self.metadata.get(key)
        return value if isinstance(value, str) else None

    def metadata_payload(self) -> dict[str, Any]:
        """Return the complete JSON payload stored beside the image in tar."""
        payload = dict(self.metadata)
        payload.update({
            "schema_version": "1.0",
            "sample_key": self.sample_key,
            "source_id": self.source_id,
            "split": self.split,
            "image_extension": self.image_extension,
            "image_sha256": self.image_sha256,
            "caption": self.caption,
            "captions": dict(self.captions),
        })
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        return payload

    def metadata_json(self) -> bytes:
        return (json_dumps(self.metadata_payload()) + "\n").encode("utf-8")

    def to_training_dict(
        self,
        caption_key: str = "caption",
        prompt_template: str | Sequence[str] | None = None,
        include_metadata: bool = True,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        """Adapt the canonical sample to RainbowNeko/HCP text-image input."""
        caption = self.select_caption(caption_key)
        if prompt_template is None:
            template = "{caption}"
        elif isinstance(prompt_template, str):
            template = prompt_template
        else:
            choices = list(prompt_template)
            if not choices:
                template = "{caption}"
            else:
                template = (rng or random).choice(choices)

        result = {
            "id": self.sample_key,
            "image": self.image,
            "prompt": {
                "template": template,
                "caption": caption,
            },
        }
        if include_metadata:
            result["metadata"] = self.metadata_payload()
        return result

