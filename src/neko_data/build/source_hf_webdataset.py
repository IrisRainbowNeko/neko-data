"""Streaming readers for WebDataset tars stored on HuggingFace Hub/KHub."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import logging
import os
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

from ..contract.records import NormalizedSample
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


@dataclass
class TarSampleParts:
    key: str
    image_name: str | None = None
    image: bytes | None = None
    text: bytes | None = None
    json_data: bytes | None = None


def iter_tar_parts(fileobj: BinaryIO) -> Iterator[TarSampleParts]:
    """Yield grouped sample members from a non-seekable tar stream.

    WebDataset writers conventionally place all members for one key together;
    relying on that invariant lets us process Hub tars without downloading
    them or retaining an unbounded dictionary of open samples.
    """
    current: TarSampleParts | None = None
    with tarfile.open(fileobj=fileobj, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            key = sample_key(member.name)
            if current is not None and key != current.key:
                yield current
                current = None
            if current is None:
                current = TarSampleParts(key=key)
            suffix = Path(member.name).suffix.lower().lstrip(".")
            data = extracted.read()
            if suffix in IMAGE_EXTENSIONS and current.image is None:
                current.image_name = member.name
                current.image = data
            elif suffix == "txt" and current.text is None:
                current.text = data
            elif suffix == "json" and current.json_data is None:
                current.json_data = data
    if current is not None:
        yield current


def normalized_samples_from_tar(
    fileobj: BinaryIO,
    *,
    split: str,
    source_id: str,
    metadata_provider=None,
    caption_keys: Iterable[str] | None = None,
) -> Iterator[NormalizedSample]:
    for parts in iter_tar_parts(fileobj):
        if parts.image is None or parts.image_name is None:
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
        sample_key_namespace: str = "none",
        request_timeout: float = 120.0,
        metadata_provider=None,
        caption_keys: Iterable[str] | None = None,
    ) -> None:
        if repo_id is None and input_files is None:
            raise ValueError("repo_id or input_files must be provided")
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
        if sample_key_namespace not in {"none", "source_path"}:
            raise ValueError("sample_key_namespace must be none or source_path")
        self.sample_key_namespace = sample_key_namespace
        self.request_timeout = request_timeout
        self.metadata_provider = metadata_provider
        self.caption_keys = caption_keys

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
                    with self.open_file(path) as fileobj:
                        for sample in normalized_samples_from_tar(
                            fileobj,
                            split=self.split,
                            source_id=source,
                            metadata_provider=self.metadata_provider,
                            caption_keys=self.caption_keys,
                        ):
                            if skip_samples:
                                skip_samples -= 1
                                continue
                            if self.sample_key_namespace == "source_path":
                                raw_key = sample.sample_key
                                namespace = hashlib.sha256(path.encode("utf-8")).hexdigest()
                                sample = replace(
                                    sample,
                                    sample_key=f"{namespace}/{raw_key}",
                                    metadata={
                                        **sample.metadata,
                                        "raw_sample_key": raw_key,
                                        "upstream_shard": path,
                                    },
                                )
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

