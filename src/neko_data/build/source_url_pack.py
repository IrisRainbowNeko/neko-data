"""High-throughput unordered URL acquisition into an indexed byte pack."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import json_safe


@dataclass(frozen=True)
class URLPackConfig:
    concurrency: int = 512
    per_host_concurrency: int = 16
    max_attempts: int = 4
    timeout_seconds: float = 15.0
    connect_timeout_seconds: float = 4.0
    max_bytes: int = 100 * 1024**2
    progress_interval: int = 500
    user_agent: str = "anime-dino-url-pack/1"

    def __post_init__(self) -> None:
        if self.concurrency <= 0 or self.per_host_concurrency <= 0:
            raise ValueError("URL pack concurrency values must be positive")
        if self.max_attempts <= 0 or self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError("URL pack retry and timeout values must be positive")
        if self.max_bytes <= 0 or self.progress_interval <= 0:
            raise ValueError("URL pack size and progress interval must be positive")


@dataclass(frozen=True)
class URLPackResult:
    block_index: int
    stage_dir: str
    report: dict[str, Any]


@dataclass(frozen=True)
class _DownloadResult:
    row: dict[str, Any]
    image: bytes | None
    image_sha256: str | None
    attempts: int
    reason: str | None
    detail: str | None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class URLPackAcquirer:
    """Download one Parquet candidate block without imposing output row order."""

    def __init__(self, config: URLPackConfig) -> None:
        self.config = config

    async def _download(self, session: aiohttp.ClientSession, row: dict[str, Any]) -> _DownloadResult:
        url = row.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return _DownloadResult(row, None, None, 0, "invalid_url", str(url))
        last_error = "download failed"
        for attempt in range(1, self.config.max_attempts + 1):
            retryable = True
            try:
                async with session.get(url, allow_redirects=True) as response:
                    status = response.status
                    if status >= 400:
                        retryable = status == 429 or status >= 500
                        raise RuntimeError(f"HTTP {status}: {response.reason or 'unknown'}")
                    declared = response.content_length
                    if declared is not None and declared > self.config.max_bytes:
                        return _DownloadResult(
                            row, None, None, attempt, "max_bytes", f"declared length {declared}"
                        )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        size += len(chunk)
                        if size > self.config.max_bytes:
                            return _DownloadResult(
                                row, None, None, attempt, "max_bytes", f"received {size} bytes"
                            )
                        chunks.append(chunk)
                    if not chunks:
                        return _DownloadResult(row, None, None, attempt, "empty_response", None)
                    image = b"".join(chunks)
                    return _DownloadResult(
                        row, image, hashlib.sha256(image).hexdigest(), attempt, None, None
                    )
            except (aiohttp.TooManyRedirects, ValueError) as error:
                last_error = str(error)
                retryable = False
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as error:
                last_error = str(error)
            if not retryable or attempt == self.config.max_attempts:
                return _DownloadResult(row, None, None, attempt, "download_failed", last_error)
            await asyncio.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0) + random.uniform(0.0, 0.25))
        raise AssertionError("unreachable retry state")

    async def _run(
        self,
        block_index: int,
        candidate_path: Path,
        partial_dir: Path,
        progress_queue: Any | None,
    ) -> dict[str, Any]:
        parquet = pq.ParquetFile(candidate_path)
        required = {"sample_key", "url", "candidate_rank"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{candidate_path} is missing columns: {sorted(missing)}")
        total = parquet.metadata.num_rows
        row_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(self.config.concurrency * 2)
        result_queue: asyncio.Queue[_DownloadResult] = asyncio.Queue(self.config.concurrency * 2)
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )
        try:
            resolver = aiohttp.AsyncResolver()
        except RuntimeError:
            resolver = None
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrency,
            limit_per_host=self.config.per_host_concurrency,
            ttl_dns_cache=3600,
            enable_cleanup_closed=True,
            resolver=resolver,
        )
        headers = {"User-Agent": self.config.user_agent}

        async def produce() -> None:
            for batch in parquet.iter_batches(batch_size=8192):
                for row in batch.to_pylist():
                    await row_queue.put({str(key): json_safe(value) for key, value in row.items()})
            for _ in range(self.config.concurrency):
                await row_queue.put(None)

        async def download_worker(session: aiohttp.ClientSession) -> None:
            while True:
                row = await row_queue.get()
                try:
                    if row is None:
                        return
                    await result_queue.put(await self._download(session, row))
                finally:
                    row_queue.task_done()

        stats: Counter[str] = Counter()
        index_rows: list[dict[str, Any]] = []
        pack_path = partial_dir / "images.pack"
        errors_path = partial_dir / "errors.jsonl"
        pack_digest = hashlib.sha256()
        processed_since_update = 0
        with pack_path.open("wb") as pack, errors_path.open("w", encoding="utf-8") as errors:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                auto_decompress=False,
            ) as session:
                producer = asyncio.create_task(produce())
                workers = [
                    asyncio.create_task(download_worker(session))
                    for _ in range(self.config.concurrency)
                ]
                for _ in range(total):
                    result = await result_queue.get()
                    try:
                        stats["seen"] += 1
                        stats["attempts"] += result.attempts
                        stats["retries"] += max(0, result.attempts - 1)
                        if result.image is None:
                            reason = result.reason or "download_failed"
                            stats[reason] += 1
                            errors.write(json.dumps({
                                "sample_key": result.row.get("sample_key"),
                                "url": result.row.get("url"),
                                "candidate_rank": result.row.get("candidate_rank"),
                                "reason": reason,
                                "detail": result.detail,
                                "attempts": result.attempts,
                            }, ensure_ascii=False, sort_keys=True) + "\n")
                        else:
                            offset = pack.tell()
                            pack.write(result.image)
                            pack_digest.update(result.image)
                            index_rows.append({
                                "candidate_rank": int(result.row["candidate_rank"]),
                                "sample_key": str(result.row["sample_key"]),
                                "url": str(result.row["url"]),
                                "text": result.row.get("text"),
                                "offset": offset,
                                "length": len(result.image),
                                "image_sha256": result.image_sha256,
                                "attempts": result.attempts,
                                "metadata_json": json.dumps(
                                    result.row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                                ),
                            })
                            stats["downloaded"] += 1
                        processed_since_update += 1
                        if progress_queue is not None and processed_since_update >= self.config.progress_interval:
                            progress_queue.put(("progress", block_index, processed_since_update, dict(stats)))
                            processed_since_update = 0
                    finally:
                        result_queue.task_done()
                await producer
                await asyncio.gather(*workers)
        if progress_queue is not None and processed_since_update:
            progress_queue.put(("progress", block_index, processed_since_update, dict(stats)))
        table = pa.Table.from_pylist(index_rows, schema=pa.schema([
            pa.field("candidate_rank", pa.int64(), nullable=False),
            pa.field("sample_key", pa.string(), nullable=False),
            pa.field("url", pa.string(), nullable=False),
            pa.field("text", pa.string()),
            pa.field("offset", pa.int64(), nullable=False),
            pa.field("length", pa.int64(), nullable=False),
            pa.field("image_sha256", pa.string(), nullable=False),
            pa.field("attempts", pa.int16(), nullable=False),
            pa.field("metadata_json", pa.large_string(), nullable=False),
        ]))
        pq.write_table(table, partial_dir / "index.parquet", compression="zstd", row_group_size=8192)
        return {
            "complete": True,
            "block_index": block_index,
            "candidate_path": str(candidate_path),
            "candidate_sha256": _sha256_path(candidate_path),
            "pack_sha256": pack_digest.hexdigest(),
            "pack_bytes": pack_path.stat().st_size,
            "stats": dict(stats),
            "config": asdict(self.config),
        }

    def acquire(
        self,
        block_index: int,
        candidate_path: str | os.PathLike[str],
        stage_root: str | os.PathLike[str],
        progress_queue: Any | None = None,
    ) -> URLPackResult:
        candidate = Path(candidate_path)
        root = Path(stage_root)
        final_dir = root / f"block-{block_index:06d}"
        report_path = final_dir / "report.json"
        candidate_sha256 = _sha256_path(candidate)
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            pack_path = final_dir / "images.pack"
            index_path = final_dir / "index.parquet"
            if (
                report.get("complete")
                and report.get("candidate_sha256") == candidate_sha256
                and pack_path.exists()
                and pack_path.stat().st_size == report.get("pack_bytes")
                and index_path.exists()
            ):
                return URLPackResult(block_index, str(final_dir), report)
        partial_dir = root / f".block-{block_index:06d}.partial"
        shutil.rmtree(partial_dir, ignore_errors=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
        report = asyncio.run(self._run(block_index, candidate, partial_dir, progress_queue))
        _atomic_json(partial_dir / "report.json", report)
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(partial_dir, final_dir)
        return URLPackResult(block_index, str(final_dir), report)


def acquire_url_pack_block(
    config: URLPackConfig,
    block_index: int,
    candidate_path: str,
    stage_root: str,
    progress_queue: Any | None = None,
) -> URLPackResult:
    """Pickle-friendly process entry point for a URL pack acquisition worker."""
    return URLPackAcquirer(config).acquire(block_index, candidate_path, stage_root, progress_queue)
