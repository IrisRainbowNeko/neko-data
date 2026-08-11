"""Ordered image acquisition from Parquet URL manifests.

The implementation is dataset-neutral.  DataComp-specific selection and
publication live in ``datas.datacomp_1b`` and pass only configuration here.
"""

from __future__ import annotations

import io
import os
import random
import threading
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
from neko_data.build.normalize import json_safe
from neko_data.contract.records import NormalizedSample
from neko_data.storage import S3Storage


class URLSampleRejected(ValueError):
    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason


@dataclass(frozen=True)
class URLSourceError:
    sample_key: str | None
    url: str | None
    reason: str
    detail: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "sample_key": self.sample_key,
            "url": self.url,
            "reason": self.reason,
            "detail": self.detail,
        }


def _local_path(uri: str | os.PathLike[str]) -> Path | None:
    value = os.fspath(uri)
    if value.startswith("file://"):
        return Path(value.removeprefix("file://"))
    return None if "://" in value else Path(value)


def _read_uri(uri: str) -> bytes:
    if uri.startswith(("s3://", "r2://")):
        with S3Storage().open(uri) as body:
            return body.read()
    request = Request(uri, headers={"User-Agent": "anime-dino-url-parquet/1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


class URLParquetSource(Iterable[NormalizedSample]):
    """Yield decoded image samples in the exact input Parquet row order."""

    def __init__(
        self,
        parquet_paths: Iterable[str | os.PathLike[str]],
        *,
        key_column: str = "sample_key",
        url_column: str = "url",
        caption_column: str | None = "text",
        metadata_columns: Iterable[str] | None = None,
        source_id: str = "url-parquet",
        split: str = "train",
        concurrency: int = 32,
        retries: int = 4,
        timeout_seconds: float = 30.0,
        max_bytes: int = 100 * 1024**2,
        min_side: int = 256,
        max_aspect_ratio: float = 4.0,
        max_pixels: int = 64 * 1024**2,
        batch_size: int = 8192,
        prefetch_factor: int = 4,
        error_handler: Callable[[URLSourceError], None] | None = None,
        progress_handler: Callable[[bool], None] | None = None,
    ) -> None:
        self.parquet_paths = tuple(os.fspath(path) for path in parquet_paths)
        if not self.parquet_paths:
            raise ValueError("at least one Parquet path is required")
        if concurrency <= 0 or retries <= 0 or batch_size <= 0 or prefetch_factor <= 0:
            raise ValueError("concurrency, retries and batch_size must be positive")
        if min_side <= 0 or max_aspect_ratio < 1 or max_pixels <= 0 or max_bytes <= 0:
            raise ValueError("invalid image validation limits")
        self.key_column = key_column
        self.url_column = url_column
        self.caption_column = caption_column
        self.metadata_columns = None if metadata_columns is None else tuple(metadata_columns)
        self.source_id = source_id
        self.split = split
        self.concurrency = concurrency
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.min_side = min_side
        self.max_aspect_ratio = max_aspect_ratio
        self.max_pixels = max_pixels
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.error_handler = error_handler
        self.progress_handler = progress_handler
        self.stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()

    def _error(self, error: URLSourceError) -> None:
        with self._stats_lock:
            self.stats[error.reason] += 1
        if self.error_handler is not None:
            self.error_handler(error)

    def _rows(self) -> Iterator[dict[str, Any]]:
        import pyarrow.parquet as pq

        for uri in self.parquet_paths:
            path = _local_path(uri)
            source: Any = path if path is not None else io.BytesIO(_read_uri(uri))
            parquet = pq.ParquetFile(source)
            required = {self.key_column, self.url_column}
            if self.caption_column:
                required.add(self.caption_column)
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{uri} is missing columns: {sorted(missing)}")
            columns = None if self.metadata_columns is None else list(required | set(self.metadata_columns))
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=columns):
                for row in batch.to_pylist():
                    yield {str(key): json_safe(value) for key, value in row.items()}

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = Request(url, headers={"User-Agent": "anime-dino-url-parquet/1"})
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) > self.max_bytes:
                        raise URLSampleRejected("max_bytes", f"declared length {length}")
                    chunks, size = [], 0
                    while True:
                        chunk = response.read(min(1024 * 1024, self.max_bytes + 1 - size))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise URLSampleRejected("max_bytes", f"received {size} bytes")
                        chunks.append(chunk)
                    if not chunks:
                        raise URLSampleRejected("empty_response")
                    return b"".join(chunks)
            except URLSampleRejected:
                raise
            except (HTTPError, URLError, OSError, TimeoutError, ValueError, HTTPException) as error:
                last_error = error
                if attempt + 1 == self.retries:
                    break
                # Most 4xx responses are permanent; only 429 can become available later.
                if isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429:
                    break
                with self._stats_lock:
                    self.stats["download_retries"] += 1
                # A burst of retries from many workers magnifies origin-side rate limits.
                time.sleep(min(0.25 * (2**attempt), 2.0) + random.uniform(0.0, 0.25))
        raise URLSampleRejected("download_failed", str(last_error))

    def _sample(self, row: Mapping[str, Any]) -> NormalizedSample:
        key, url = row.get(self.key_column), row.get(self.url_column)
        if key is None or not str(key).strip():
            raise URLSampleRejected("missing_key")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise URLSampleRejected("invalid_url")
        image = self._download(url)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image)) as opened:
                    if getattr(opened, "is_animated", False):
                        raise URLSampleRejected("animated_image")
                    width, height = opened.size
                    if min(width, height) < self.min_side:
                        raise URLSampleRejected("min_side", f"{width}x{height}")
                    if max(width, height) / min(width, height) > self.max_aspect_ratio:
                        raise URLSampleRejected("aspect_ratio", f"{width}x{height}")
                    if width * height > self.max_pixels:
                        raise URLSampleRejected("max_pixels", f"{width}x{height}")
                    extension = {"jpeg": "jpg", "tiff": "tif"}.get((opened.format or "jpg").lower(), (opened.format or "jpg").lower())
                    opened.verify()
                with Image.open(io.BytesIO(image)) as verified:
                    verified.load()
        except URLSampleRejected:
            raise
        except Exception as error:
            raise URLSampleRejected("decode_failed", str(error)) from error
        caption = row.get(self.caption_column) if self.caption_column else None
        caption = caption.strip() if isinstance(caption, str) and caption.strip() else None
        if self.metadata_columns is None:
            metadata = {name: value for name, value in row.items() if name not in {self.key_column, self.url_column, self.caption_column}}
        else:
            metadata = {name: row.get(name) for name in self.metadata_columns if name in row}
        metadata["url"] = url
        return NormalizedSample(
            sample_key=str(key), image=image, image_extension=extension, split=self.split,
            source_id=self.source_id, caption=caption, captions={"text": caption} if caption else {},
            metadata=metadata, width=width, height=height,
        )

    def _process(self, row: Mapping[str, Any]) -> NormalizedSample | None:
        key = None if row.get(self.key_column) is None else str(row.get(self.key_column))
        url = row.get(self.url_column)
        with self._stats_lock:
            self.stats["seen"] += 1
        try:
            sample = self._sample(row)
        except URLSampleRejected as error:
            self._error(URLSourceError(key, url if isinstance(url, str) else None, error.reason, str(error)))
            if self.progress_handler is not None:
                self.progress_handler(False)
            return None
        with self._stats_lock:
            self.stats["accepted"] += 1
        if self.progress_handler is not None:
            self.progress_handler(True)
        return sample

    def __iter__(self) -> Iterator[NormalizedSample]:
        rows = iter(self._rows())
        pending: dict[Future[NormalizedSample | None], int] = {}
        completed: dict[int, NormalizedSample | None] = {}
        exhausted = False
        next_submit = 0
        next_yield = 0
        max_pending = self.concurrency * self.prefetch_factor
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="url-parquet") as pool:
            while pending or completed or not exhausted:
                while not exhausted and len(pending) + len(completed) < max_pending:
                    try:
                        row = next(rows)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(self._process, row)] = next_submit
                    next_submit += 1
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        completed[pending.pop(future)] = future.result()
                while next_yield in completed:
                    sample = completed.pop(next_yield)
                    next_yield += 1
                    if sample is not None:
                        yield sample
