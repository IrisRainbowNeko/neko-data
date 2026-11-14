"""Streaming readers for WebDataset tars stored on HuggingFace Hub/KHub."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Iterator

from ..contract.blobs import BLOB_SAMPLE_FORMAT, BlobSampleReference
from ..contract.records import NormalizedSample
from ..contract.sample_metadata import MAX_SAMPLE_METADATA_BYTES
from ..contract.typed import TYPED_EXTENSION, TYPED_SAMPLE_FORMAT, TypedSample, decode_typed_tree
from .normalize import IMAGE_EXTENSIONS, sample_from_wds_parts

LOGGER = logging.getLogger(__name__)


def _retryable_stream_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [OSError, EOFError, tarfile.ReadError]
    try:
        import requests

        errors.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    try:
        import urllib3

        for name in ("ProtocolError", "IncompleteRead"):
            error_type = getattr(urllib3.exceptions, name, None)
            if error_type is not None:
                errors.append(error_type)
    except ImportError:
        pass
    return tuple(errors)


def sample_key(member_name: str) -> str:
    path = member_name.lstrip("./")
    return path.rsplit(".", 1)[0] if "." in path else path


def canonical_source_path(path: str) -> str:
    """Normalize an upstream repository path without using a signed URL."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


@dataclass
class TarSampleParts:
    key: str
    image_name: str | None = None
    image: bytes | None = None
    text: bytes | None = None
    json_data: bytes | None = None
    typed_data: bytes | None = None
    blob_seen: bool = False
    oversized_json: BinaryIO | None = None


def _completed_parts(parts):
    try:
        yield parts
    finally:
        if parts.oversized_json is not None:
            parts.oversized_json.close()


def iter_tar_parts(fileobj: BinaryIO) -> Iterator[TarSampleParts]:
    """Yield grouped sample members from a non-seekable tar stream.

    WebDataset writers conventionally place all members for one key together;
    relying on that invariant lets us process Hub tars without downloading
    them or retaining an unbounded dictionary of open samples.
    """
    current: TarSampleParts | None = None
    with ExitStack() as cleanup, tarfile.open(fileobj=fileobj, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            key = sample_key(member.name)
            if current is not None and key != current.key:
                yield from _completed_parts(current)
                current = None
            if current is None:
                current = TarSampleParts(key=key)
            suffix = Path(member.name).suffix.lower().lstrip(".")
            duplicate_part = (
                (suffix in IMAGE_EXTENSIONS and current.image is not None)
                or (suffix == "txt" and current.text is not None)
                or (suffix == "json" and current.json_data is not None)
                or (suffix == TYPED_EXTENSION and current.typed_data is not None)
            )
            if duplicate_part:
                yield from _completed_parts(current)
                current = TarSampleParts(key=key)
            if suffix not in IMAGE_EXTENSIONS and suffix not in {"txt", "json", TYPED_EXTENSION}:
                if suffix.startswith("nblob"):
                    current.blob_seen = True
                continue
            if suffix == TYPED_EXTENSION and member.size > 1024**3 + 1024**2:
                raise ValueError("Typed tar member exceeds maximum payload size")
            if suffix == "json" and member.size > MAX_SAMPLE_METADATA_BYTES:
                if current.typed_data is not None or current.blob_seen:
                    raise ValueError("Versioned sample metadata exceeds maximum descriptor size")
                # JSON may precede a legacy image or a versioned blob. Defer
                # this distinction without allocating an unbounded descriptor.
                # The final consumer may load it only for legacy image data.
                current.json_data = b""
                current.oversized_json = cleanup.enter_context(tempfile.TemporaryFile())
                try:
                    shutil.copyfileobj(extracted, current.oversized_json, length=8 * 1024**2)
                    current.oversized_json.seek(0)
                except BaseException:
                    current.oversized_json.close()
                    raise
                continue
            data = extracted.read()
            if suffix in IMAGE_EXTENSIONS and current.image is None:
                current.image_name = member.name
                current.image = data
            elif suffix == "txt" and current.text is None:
                current.text = data
            elif suffix == "json" and current.json_data is None:
                current.json_data = data
            elif suffix == TYPED_EXTENSION:
                current.typed_data = data
        if current is not None:
            yield from _completed_parts(current)


def normalized_samples_from_tar(
    fileobj: BinaryIO,
    *,
    split: str,
    source_id: str,
    metadata_provider=None,
    caption_keys: Iterable[str] | None = None,
) -> Iterator[NormalizedSample | TypedSample | BlobSampleReference]:
    for parts in iter_tar_parts(fileobj):
        if parts.oversized_json is not None:
            if parts.typed_data is not None or parts.blob_seen:
                raise ValueError("Versioned sample metadata exceeds maximum descriptor size")
            if parts.image is None:
                continue  # retain legacy orphan-JSON skipping without reading it
            parts.json_data = parts.oversized_json.read()  # legacy image metadata remains compatible
        if parts.blob_seen and (parts.typed_data is not None or parts.image is not None):
            raise ValueError("Blob samples cannot contain typed or image members")
        if parts.typed_data is not None:
            if parts.image is not None or parts.json_data is None:
                raise ValueError("Typed samples require JSON metadata and cannot contain an image")
            metadata = json.loads(parts.json_data)
            if (metadata.get("sample_format") != TYPED_SAMPLE_FORMAT
                    or metadata.get("sample_key") != parts.key
                    or metadata.get("payload_extension") != TYPED_EXTENSION):
                raise ValueError("Invalid typed sample metadata")
            if hashlib.sha256(parts.typed_data).hexdigest() != metadata.get("payload_sha256"):
                raise ValueError("Typed sample payload checksum mismatch")
            yield TypedSample(parts.key, decode_typed_tree(parts.typed_data),
                              split=metadata.pop("split", split), source_id=metadata.pop("source_id", source_id),
                              metadata=metadata)
            continue
        if parts.image is None or parts.image_name is None:
            if parts.json_data is not None:
                try:
                    metadata = json.loads(parts.json_data)
                except json.JSONDecodeError:
                    if parts.blob_seen:
                        raise ValueError("Invalid blob metadata JSON") from None
                    continue  # legacy orphan non-image records are skipped
                if not isinstance(metadata, dict):
                    if parts.blob_seen:
                        raise ValueError("Blob metadata must be a JSON object")
                    continue
                if metadata.get("sample_format") == BLOB_SAMPLE_FORMAT:
                    if metadata.get("sample_key") != parts.key:
                        raise ValueError("Blob sample key mismatch")
                    yield BlobSampleReference(parts.key, metadata["blobs"], metadata.get("split", split),
                                              metadata.get("source_id", source_id), metadata)
                elif parts.blob_seen:
                    raise ValueError("Blob payload has no versioned blob metadata")
            elif parts.blob_seen:
                raise ValueError("Blob payload is missing JSON metadata")
            continue
        joined = metadata_provider.lookup(parts.key) if metadata_provider is not None else None
        yield sample_from_wds_parts(
            parts.key,
            parts.image_name,
            parts.image,
            text=parts.text,
            json_data=parts.json_data,
            split=split,
            source_id=source_id,
            metadata_provider=joined,
            caption_keys=caption_keys,
        )


class HFWebDatasetSource:
    """Iterate Hub-hosted tar files directly over HTTP.

    ``input_files`` can be supplied to avoid a repository listing, which is
    useful for private mirrors and for reproducible build manifests.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        *,
        repo_type: str = "dataset",
        revision: str | None = None,
        endpoint: str | None = None,
        token: str | None = None,
        input_files: Iterable[str] | None = None,
        include: Iterable[str] = ("*.tar",),
        exclude: Iterable[str] = (),
        max_retries: int = 5,
        retry_backoff: float = 5.0,
        path_prefix: str = "",
        split: str = "train",
        source_id: str | None = None,
        request_timeout: float = 120.0,
        metadata_provider=None,
        caption_keys: Iterable[str] | None = None,
        sample_key_namespace: str | None = None,
    ) -> None:
        if repo_id is None and input_files is None:
            raise ValueError("repo_id or input_files must be provided")
        if sample_key_namespace not in {None, "none", "source_path"}:
            raise ValueError("sample_key_namespace must be None, 'none', or 'source_path'")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.endpoint = endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("KHUB_TOKEN")
        self.input_files = list(input_files) if input_files is not None else None
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.path_prefix = path_prefix.strip("/")
        self.split = split
        self.source_id = source_id or repo_id or "hf-webdataset"
        self.request_timeout = request_timeout
        self.metadata_provider = metadata_provider
        self.caption_keys = caption_keys
        self.sample_key_namespace = None if sample_key_namespace == "none" else sample_key_namespace

    def _apply_sample_key_namespace(self, sample: NormalizedSample, path: str) -> NormalizedSample:
        if self.sample_key_namespace is None:
            return sample
        upstream_shard = canonical_source_path(path)
        namespace = hashlib.sha256(upstream_shard.encode("utf-8")).hexdigest()
        raw_sample_key = sample.sample_key
        return replace(
            sample,
            sample_key=f"{namespace}/{raw_sample_key}",
            metadata={
                **sample.metadata,
                "raw_sample_key": raw_sample_key,
                "upstream_shard": upstream_shard,
            },
        )

    def list_files(self) -> list[str]:
        if self.input_files is not None:
            return sorted(dict.fromkeys(self.input_files))
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("HF Hub support requires the 'hf' extra: pip install neko-data[hf]") from exc
        kwargs = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "token": self.token,
        }
        api = HfApi(endpoint=self.endpoint) if "endpoint" in inspect.signature(HfApi).parameters else HfApi()
        files = api.list_repo_files(**kwargs)
        selected = []
        for path in files:
            if self.path_prefix and not path.startswith(self.path_prefix.rstrip("/") + "/"):
                continue
            if not any(fnmatch.fnmatch(path, pattern) for pattern in self.include):
                continue
            if any(fnmatch.fnmatch(path, pattern) for pattern in self.exclude):
                continue
            selected.append(path)
        return sorted(selected)

    @contextmanager
    def open_file(self, path: str) -> Iterator[BinaryIO]:
        local_path = Path(path)
        if local_path.exists():
            with local_path.open("rb") as fileobj:
                yield fileobj
            return
        try:
            import requests
            from huggingface_hub import hf_hub_url
            from huggingface_hub.utils import build_hf_headers
        except ImportError as exc:
            raise RuntimeError("Streaming Hub tars require the 'hf' extra: pip install neko-data[hf]") from exc
        kwargs = {
            "repo_id": self.repo_id,
            "filename": path,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }
        if "endpoint" in inspect.signature(hf_hub_url).parameters:
            kwargs["endpoint"] = self.endpoint
        url = hf_hub_url(**kwargs)
        headers = build_hf_headers(token=self.token)
        with requests.get(url, headers=headers, stream=True, timeout=self.request_timeout) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            yield response.raw

    def __iter__(self) -> Iterator[NormalizedSample]:
        files = self.list_files()
        if not files:
            raise RuntimeError("No WebDataset tar files matched the configured Hub source")
        retryable_errors = _retryable_stream_errors()
        for path in files:
            source = f"{self.source_id}:{path}"
            emitted_from_file = 0
            skip_samples = 0
            retries = 0
            while True:
                try:
                    seen_raw_keys: set[str] = set()
                    with self.open_file(path) as fileobj:
                        for sample in normalized_samples_from_tar(
                            fileobj,
                            split=self.split,
                            source_id=source,
                            metadata_provider=self.metadata_provider,
                            caption_keys=self.caption_keys,
                        ):
                            raw_sample_key = sample.sample_key
                            if raw_sample_key in seen_raw_keys:
                                raise ValueError(
                                    f"Duplicate sample key {raw_sample_key!r} in upstream shard "
                                    f"{canonical_source_path(path)!r}"
                                )
                            seen_raw_keys.add(raw_sample_key)
                            sample = self._apply_sample_key_namespace(sample, path)
                            if skip_samples:
                                skip_samples -= 1
                                continue
                            emitted_from_file += 1
                            yield sample
                    break
                except retryable_errors as exc:
                    if retries >= self.max_retries:
                        raise
                    retries += 1
                    delay = min(self.retry_backoff * (2 ** (retries - 1)), 60.0)
                    LOGGER.warning(
                        "Retrying %s after %d samples; retry %d/%d in %.1fs: %s",
                        source,
                        emitted_from_file,
                        retries,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    skip_samples = emitted_from_file
                    if delay:
                        time.sleep(delay)

    def __len__(self) -> int:
        raise TypeError("HFWebDatasetSource is streaming and has no implicit length")
