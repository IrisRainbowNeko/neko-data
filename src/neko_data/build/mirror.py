"""Mirror prebuilt HuggingFace/KHub WebDataset shards without repacking."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..contract import DatasetManifest, write_manifest, write_shard_index
from ..contract.schema import ShardRecord
from ..storage.r2 import S3Storage
from .publisher import join_uri

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp", "avif", "gif", "bmp", "tif", "tiff"})


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: str) -> str:
    value = path.lstrip("/")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\x00" in value:
        raise ValueError(f"Unsafe Hub path: {path!r}")
    return parsed.as_posix()


def _image_member_count(files: Mapping[str, Any]) -> int:
    return sum(
        1
        for name in files
        if PurePosixPath(str(name)).suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True)
class HubFile:
    path: str
    size_bytes: int
    sha256: str | None = None


@dataclass(frozen=True)
class MirrorPlan:
    revision: str
    shards: tuple[HubFile, ...]
    auxiliary: tuple[HubFile, ...]

    @property
    def tar_bytes(self) -> int:
        return sum(item.size_bytes for item in self.shards)


@dataclass
class MirrorJob:
    dataset_id: str
    version: str
    repo_id: str
    destination: str
    output_dir: Path
    staging_dir: Path
    split: str = "train"
    repo_type: str = "dataset"
    endpoint: str = "https://huggingface.co"
    revision: str = "main"
    include: tuple[str, ...] = ("*.tar",)
    exclude: tuple[str, ...] = ()
    sidecar_suffix: str = ".json"
    auxiliary_files: dict[str, str] = field(default_factory=dict)
    workers: int = 8
    request_timeout: float = 120.0
    max_retries: int = 8
    retry_backoff: float = 5.0
    r2_endpoint: str | None = None
    r2_region: str = "auto"
    enable_multipart: bool = False
    upload_concurrency: int = 1
    resume: bool = False
    expected_shards: int | None = None
    expected_samples: int | None = None
    expected_tar_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "MirrorJob":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML mirror jobs require PyYAML") from exc
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(config, Mapping):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        dataset = config.get("dataset", {})
        source = config.get("source", {})
        publish = config.get("publish", {})
        mirror = config.get("mirror", {})
        expected = config.get("expected", {})
        if source.get("type", "hf_prebuilt_webdataset") != "hf_prebuilt_webdataset":
            raise ValueError(f"Unsupported mirror source type: {source.get('type')}")
        output_dir = Path(mirror.get("output_dir", f"./mirror/{dataset['dataset_id']}/{dataset['version']}"))
        auxiliary = source.get("auxiliary_files", {})
        if isinstance(auxiliary, Sequence) and not isinstance(auxiliary, (str, bytes, Mapping)):
            auxiliary = {str(item): str(item) for item in auxiliary}
        if not isinstance(auxiliary, Mapping):
            raise ValueError("source.auxiliary_files must be a mapping or list")
        return cls(
            dataset_id=str(dataset["dataset_id"]),
            version=str(dataset["version"]),
            split=str(dataset.get("split", "train")),
            repo_id=str(source["repo_id"]),
            repo_type=str(source.get("repo_type", "dataset")),
            endpoint=str(source.get("endpoint", "https://huggingface.co")),
            revision=str(source.get("revision", "main")),
            include=tuple(str(item) for item in source.get("include", ["*.tar"])),
            exclude=tuple(str(item) for item in source.get("exclude", [])),
            sidecar_suffix=str(source.get("sidecar_suffix", ".json")),
            auxiliary_files={str(key): str(value) for key, value in auxiliary.items()},
            destination=str(publish["destination"]),
            r2_endpoint=publish.get("endpoint"),
            r2_region=str(publish.get("region", "auto")),
            enable_multipart=bool(publish.get("enable_multipart", False)),
            upload_concurrency=int(publish.get("upload_concurrency", 1)),
            output_dir=output_dir,
            staging_dir=Path(mirror.get("staging_dir", output_dir / "staging")),
            workers=int(mirror.get("workers", 8)),
            request_timeout=float(mirror.get("request_timeout", 120.0)),
            max_retries=int(mirror.get("max_retries", 8)),
            retry_backoff=float(mirror.get("retry_backoff", 5.0)),
            resume=bool(mirror.get("resume", False)),
            expected_shards=_optional_int(expected.get("shards")),
            expected_samples=_optional_int(expected.get("samples")),
            expected_tar_bytes=_optional_int(expected.get("tar_bytes")),
            metadata=dict(dataset.get("metadata", {})),
        )


def _token() -> str | None:
    return os.environ.get("KHUB_TOKEN") or os.environ.get("HF_TOKEN")


def _hub_api(job: MirrorJob):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Hub mirroring requires the 'hf' extra: pip install neko-data[hf]") from exc
    kwargs: dict[str, Any] = {}
    if "endpoint" in inspect.signature(HfApi).parameters:
        kwargs["endpoint"] = job.endpoint
    if "token" in inspect.signature(HfApi).parameters:
        kwargs["token"] = _token()
    return HfApi(**kwargs)


def _lfs_sha256(item: Any) -> str | None:
    lfs = getattr(item, "lfs", None)
    value = getattr(lfs, "sha256", None) if lfs is not None else None
    return str(value) if value else None


def inspect_mirror(
    job: MirrorJob,
    *,
    limit: int | None = None,
    copy_auxiliary: bool = True,
) -> MirrorPlan:
    api = _hub_api(job)
    info = api.repo_info(
        job.repo_id,
        repo_type=job.repo_type,
        revision=job.revision,
        token=_token(),
    )
    revision = str(info.sha)
    entries = list(api.list_repo_tree(
        job.repo_id,
        repo_type=job.repo_type,
        revision=revision,
        recursive=True,
        expand=True,
        token=_token(),
    ))
    files = {
        str(item.path): HubFile(
            str(item.path),
            int(getattr(item, "size", 0) or 0),
            _lfs_sha256(item),
        )
        for item in entries
        if hasattr(item, "size")
    }
    selected = [
        item
        for path, item in files.items()
        if any(fnmatch.fnmatch(path, pattern) for pattern in job.include)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in job.exclude)
    ]
    selected.sort(key=lambda item: item.path)
    if limit is not None:
        selected = selected[:max(0, limit)]
    if not selected:
        raise RuntimeError("No prebuilt WebDataset tar files matched the mirror source")
    missing = []
    for item in selected:
        sidecar_path = str(PurePosixPath(item.path).with_suffix(job.sidecar_suffix))
        if sidecar_path not in files:
            missing.append(sidecar_path)
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing sidecars for {len(missing)} shards: {preview}")
    auxiliary = []
    if copy_auxiliary:
        for source_path in job.auxiliary_files:
            if source_path not in files:
                raise RuntimeError(f"Auxiliary Hub file is missing: {source_path}")
            auxiliary.append(files[source_path])
    plan = MirrorPlan(revision, tuple(selected), tuple(auxiliary))
    if limit is None:
        if job.expected_shards is not None and len(plan.shards) != job.expected_shards:
            raise RuntimeError(f"Expected {job.expected_shards} shards, found {len(plan.shards)}")
        if job.expected_tar_bytes is not None and plan.tar_bytes != job.expected_tar_bytes:
            raise RuntimeError(f"Expected {job.expected_tar_bytes} tar bytes, found {plan.tar_bytes}")
    return plan


def mirror_status(job: MirrorJob) -> dict[str, Any]:
    records = _load_records(job.output_dir)
    shards = len(records)
    expected = job.expected_shards
    return {
        "dataset_id": job.dataset_id,
        "version": job.version,
        "shards": shards,
        "expected_shards": expected,
        "completion": None if not expected else shards / expected,
        "samples": sum(record.num_samples for record in records.values()),
        "tar_bytes": sum(record.size_bytes for record in records.values()),
        "staging_bytes": sum(
            path.stat().st_size
            for path in job.staging_dir.rglob("*")
            if path.is_file()
        ) if job.staging_dir.exists() else 0,
    }


def _hub_url(job: MirrorJob, revision: str, path: str) -> str:
    try:
        from huggingface_hub import hf_hub_url
    except ImportError as exc:
        raise RuntimeError("Hub mirroring requires the 'hf' extra: pip install neko-data[hf]") from exc
    kwargs = {
        "repo_id": job.repo_id,
        "filename": path,
        "repo_type": job.repo_type,
        "revision": revision,
    }
    if "endpoint" in inspect.signature(hf_hub_url).parameters:
        kwargs["endpoint"] = job.endpoint
    return str(hf_hub_url(**kwargs))


def _hub_headers() -> dict[str, str]:
    try:
        from huggingface_hub.utils import build_hf_headers
    except ImportError as exc:
        raise RuntimeError("Hub mirroring requires the 'hf' extra: pip install neko-data[hf]") from exc
    return dict(build_hf_headers(token=_token()))


def _retry_delay(job: MirrorJob, attempt: int) -> float:
    return min(job.retry_backoff * (2 ** max(0, attempt - 1)), 60.0)


def _request_bytes(job: MirrorJob, revision: str, path: str) -> bytes:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Hub mirroring requires requests") from exc
    url = _hub_url(job, revision, path)
    for attempt in range(1, job.max_retries + 2):
        try:
            with requests.get(url, headers=_hub_headers(), timeout=job.request_timeout) as response:
                response.raise_for_status()
                return bytes(response.content)
        except Exception:
            if attempt > job.max_retries:
                raise
            time.sleep(_retry_delay(job, attempt))
    raise AssertionError("unreachable")


def _download_hub_file(
    job: MirrorJob,
    revision: str,
    source: HubFile,
    destination: Path,
) -> tuple[Path, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Hub mirroring requires requests") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _hub_url(job, revision, source.path)
    for attempt in range(1, job.max_retries + 2):
        current_size = destination.stat().st_size if destination.exists() else 0
        if current_size > source.size_bytes:
            destination.unlink()
            current_size = 0
        headers = _hub_headers()
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=job.request_timeout,
            ) as response:
                if current_size and response.status_code == 416 and current_size == source.size_bytes:
                    pass
                else:
                    response.raise_for_status()
                    append = current_size > 0 and response.status_code == 206
                    mode = "ab" if append else "wb"
                    with destination.open(mode) as output:
                        for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                            if block:
                                output.write(block)
            downloaded_size = destination.stat().st_size
            if downloaded_size != source.size_bytes:
                raise IOError(
                    f"Downloaded size mismatch for {source.path}: "
                    f"{downloaded_size} != {source.size_bytes}"
                )
            digest = _sha256_file(destination)
            if source.sha256 and digest != source.sha256:
                destination.unlink(missing_ok=True)
                raise IOError(f"Downloaded SHA-256 mismatch for {source.path}")
            return destination, digest
        except Exception:
            if attempt > job.max_retries:
                raise
            time.sleep(_retry_delay(job, attempt))
    raise AssertionError("unreachable")


def _storage(job: MirrorJob) -> S3Storage:
    return S3Storage(
        endpoint_url=job.r2_endpoint,
        region_name=job.r2_region,
        enable_multipart=job.enable_multipart,
    )


def _object_metadata(job: MirrorJob, revision: str, sha256: str) -> dict[str, str]:
    return {
        "sha256": sha256,
        "source-repo": job.repo_id,
        "source-revision": revision,
    }


def _head_or_none(storage: S3Storage, uri: str) -> dict[str, object] | None:
    try:
        return storage.head(uri)
    except Exception as exc:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return None
        raise


def _remote_valid(
    storage: S3Storage,
    uri: str,
    size_bytes: int | None,
    sha256: str,
) -> bool:
    head = _head_or_none(storage, uri)
    if head is None:
        return False
    if size_bytes is not None and int(head.get("size_bytes", -1)) != size_bytes:
        return False
    metadata = head.get("metadata", {})
    return isinstance(metadata, Mapping) and metadata.get("sha256") == sha256


def _upload_file(
    job: MirrorJob,
    storage: S3Storage,
    revision: str,
    source: Path,
    target: str,
    sha256: str,
) -> None:
    uri = join_uri(job.destination, target)
    if _remote_valid(storage, uri, source.stat().st_size, sha256):
        return
    for attempt in range(1, job.max_retries + 2):
        try:
            storage.upload_file(
                source,
                uri,
                max_concurrency=job.upload_concurrency,
                metadata=_object_metadata(job, revision, sha256),
            )
            if not _remote_valid(storage, uri, source.stat().st_size, sha256):
                raise IOError(f"R2 object verification failed: {uri}")
            return
        except Exception:
            if attempt > job.max_retries:
                raise
            time.sleep(_retry_delay(job, attempt))


def _upload_bytes(
    job: MirrorJob,
    storage: S3Storage,
    revision: str,
    data: bytes,
    target: str,
    sha256: str,
) -> None:
    uri = join_uri(job.destination, target)
    if _remote_valid(storage, uri, len(data), sha256):
        return
    for attempt in range(1, job.max_retries + 2):
        try:
            storage.upload_bytes(
                data,
                uri,
                metadata=_object_metadata(job, revision, sha256),
            )
            if not _remote_valid(storage, uri, len(data), sha256):
                raise IOError(f"R2 object verification failed: {uri}")
            return
        except Exception:
            if attempt > job.max_retries:
                raise
            time.sleep(_retry_delay(job, attempt))


def _sidecar_record(
    job: MirrorJob,
    source: HubFile,
    sidecar_path: str,
    sidecar_data: bytes,
    index: int,
) -> ShardRecord:
    payload = json.loads(sidecar_data)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Shard sidecar must be a mapping: {sidecar_path}")
    size_bytes = int(payload.get("filesize", -1))
    sha256 = str(payload.get("hash_lfs", ""))
    files = payload.get("files")
    if size_bytes != source.size_bytes:
        raise ValueError(f"Sidecar size mismatch for {source.path}: {size_bytes} != {source.size_bytes}")
    if not _is_sha256(sha256):
        raise ValueError(f"Sidecar does not contain a valid hash_lfs SHA-256: {sidecar_path}")
    if source.sha256 is not None and source.sha256 != sha256:
        raise ValueError(f"Hub LFS and sidecar SHA-256 disagree for {source.path}")
    if not isinstance(files, Mapping):
        raise ValueError(f"Sidecar files must be a mapping: {sidecar_path}")
    num_samples = _image_member_count(files)
    if num_samples <= 0:
        raise ValueError(f"Sidecar contains no supported images: {sidecar_path}")
    tar_target = (PurePosixPath("wds") / job.split / _relative_path(source.path)).as_posix()
    metadata_target = (
        PurePosixPath("metadata")
        / "shards"
        / PurePosixPath(_relative_path(sidecar_path))
    ).as_posix()
    return ShardRecord(
        path=tar_target,
        split=job.split,
        num_samples=num_samples,
        size_bytes=size_bytes,
        sha256=sha256,
        metadata_path=metadata_target,
        metadata_sha256=_sha256_bytes(sidecar_data),
        index=index,
        source_ids=(job.repo_id,),
    )


def _append_record(path: Path, record: ShardRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _load_records(root: Path) -> dict[str, ShardRecord]:
    records: dict[str, ShardRecord] = {}
    reports = root / "reports" / "workers"
    if not reports.exists():
        return records
    for path in sorted(reports.glob("worker-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = ShardRecord.from_dict(json.loads(line))
                records[record.path] = record
    return records


def _completed_remote(job: MirrorJob, storage: S3Storage, record: ShardRecord) -> bool:
    if not record.metadata_path or not record.metadata_sha256:
        return False
    return (
        _remote_valid(
            storage,
            join_uri(job.destination, record.path),
            record.size_bytes,
            record.sha256,
        )
        and _remote_valid(
            storage,
            join_uri(job.destination, record.metadata_path),
            None,
            record.metadata_sha256,
        )
    )


def _mirror_one(
    job: MirrorJob,
    revision: str,
    source: HubFile,
    index: int,
    worker_id: int,
    previous: ShardRecord | None,
) -> tuple[ShardRecord, bool]:
    storage = _storage(job)
    prefix = f"[worker-{worker_id:02d}]"
    if previous is not None and previous.size_bytes == source.size_bytes:
        if _completed_remote(job, storage, previous):
            print(f"{prefix} verified {source.path}", flush=True)
            return previous, True

    sidecar_path = str(PurePosixPath(source.path).with_suffix(job.sidecar_suffix))
    sidecar_data = _request_bytes(job, revision, sidecar_path)
    record = _sidecar_record(job, source, sidecar_path, sidecar_data, index)
    assert record.metadata_path is not None and record.metadata_sha256 is not None
    _upload_bytes(
        job,
        storage,
        revision,
        sidecar_data,
        record.metadata_path,
        record.metadata_sha256,
    )

    tar_uri = join_uri(job.destination, record.path)
    if _remote_valid(storage, tar_uri, record.size_bytes, record.sha256):
        print(f"{prefix} recovered {source.path} from R2", flush=True)
        return record, False

    stage_path = job.staging_dir / f"worker-{worker_id:02d}" / _relative_path(source.path)
    print(f"{prefix} downloading {source.path} ({source.size_bytes / 1024**3:.2f} GiB)", flush=True)
    expected = HubFile(source.path, source.size_bytes, record.sha256)
    stage_path, digest = _download_hub_file(job, revision, expected, stage_path)
    try:
        print(f"{prefix} uploading {record.path}", flush=True)
        _upload_file(job, storage, revision, stage_path, record.path, digest)
    finally:
        stage_path.unlink(missing_ok=True)
    print(f"{prefix} completed {source.path}: {record.num_samples:,} samples", flush=True)
    return record, False


def _mirror_partition(
    job: MirrorJob,
    revision: str,
    worker_id: int,
    items: Sequence[tuple[int, HubFile]],
    previous: Mapping[str, ShardRecord],
) -> list[ShardRecord]:
    journal = job.output_dir / "reports" / "workers" / f"worker-{worker_id:02d}.jsonl"
    records = []
    for index, source in items:
        target = (PurePosixPath("wds") / job.split / _relative_path(source.path)).as_posix()
        record, already_recorded = _mirror_one(
            job,
            revision,
            source,
            index,
            worker_id,
            previous.get(target),
        )
        if not already_recorded:
            _append_record(journal, record)
        records.append(record)
    return records


def _partition_by_size(
    items: Sequence[tuple[int, HubFile]],
    workers: int,
) -> list[list[tuple[int, HubFile]]]:
    count = max(1, min(workers, len(items)))
    groups: list[list[tuple[int, HubFile]]] = [[] for _ in range(count)]
    totals = [0] * count
    for item in sorted(items, key=lambda value: value[1].size_bytes, reverse=True):
        group_index = min(range(count), key=totals.__getitem__)
        groups[group_index].append(item)
        totals[group_index] += item[1].size_bytes
    for group in groups:
        group.sort(key=lambda value: value[0])
    return groups


def _mirror_auxiliary(job: MirrorJob, plan: MirrorPlan) -> dict[str, dict[str, Any]]:
    storage = _storage(job)
    source_entries = {item.path: item for item in plan.auxiliary}
    result = {}
    for source_path, target_path in job.auxiliary_files.items():
        source = source_entries[source_path]
        if source.sha256 and _remote_valid(
            storage,
            join_uri(job.destination, target_path),
            source.size_bytes,
            source.sha256,
        ):
            digest = source.sha256
        else:
            stage_path = job.staging_dir / "auxiliary" / _relative_path(source_path)
            stage_path, digest = _download_hub_file(job, plan.revision, source, stage_path)
            try:
                _upload_file(job, storage, plan.revision, stage_path, target_path, digest)
            finally:
                stage_path.unlink(missing_ok=True)
        result[source_path] = {
            "path": target_path,
            "size_bytes": source.size_bytes,
            "sha256": digest,
        }
        print(f"[auxiliary] completed {source_path}", flush=True)
    return result


def _publish_control_file(
    job: MirrorJob,
    revision: str,
    local_path: Path,
    target: str,
) -> None:
    digest = _sha256_file(local_path)
    _upload_file(job, _storage(job), revision, local_path, target, digest)


def _ensure_destination_revision(job: MirrorJob, revision: str) -> None:
    uri = join_uri(job.destination, "manifest.json")
    head = _head_or_none(_storage(job), uri)
    if head is None:
        return
    metadata = head.get("metadata", {})
    source_revision = metadata.get("source-revision") if isinstance(metadata, Mapping) else None
    if source_revision != revision:
        raise RuntimeError(
            f"Destination manifest belongs to source revision {source_revision!r}, "
            f"not {revision!r}: {uri}"
        )
    if not job.resume:
        raise FileExistsError(f"Destination manifest already exists: {uri}; set resume=True")


def mirror_dataset(
    job: MirrorJob,
    *,
    limit: int | None = None,
    copy_auxiliary: bool = True,
) -> DatasetManifest:
    reports = job.output_dir / "reports" / "workers"
    if not job.resume and reports.exists():
        raise FileExistsError(
            f"Mirror journals already exist under {job.output_dir}; "
            "set resume=True or use a new output directory"
        )
    plan = inspect_mirror(job, limit=limit, copy_auxiliary=copy_auxiliary)
    _ensure_destination_revision(job, plan.revision)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.staging_dir.mkdir(parents=True, exist_ok=True)
    previous = _load_records(job.output_dir)
    indexed = list(enumerate(plan.shards))
    groups = _partition_by_size(indexed, job.workers)
    started_at = time.time()
    records: list[ShardRecord] = []
    if len(groups) == 1:
        records = _mirror_partition(job, plan.revision, 0, groups[0], previous)
    else:
        with ProcessPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(
                    _mirror_partition,
                    job,
                    plan.revision,
                    worker_id,
                    group,
                    previous,
                ): worker_id
                for worker_id, group in enumerate(groups)
            }
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda record: record.index if record.index is not None else record.path)
    if len(records) != len(plan.shards):
        raise RuntimeError(f"Mirrored {len(records)} of {len(plan.shards)} shards")
    samples = sum(record.num_samples for record in records)
    tar_bytes = sum(record.size_bytes for record in records)
    if limit is None and job.expected_samples is not None and samples != job.expected_samples:
        raise RuntimeError(f"Expected {job.expected_samples} samples, mirrored {samples}")
    auxiliary = _mirror_auxiliary(job, plan) if copy_auxiliary else {}
    manifest = DatasetManifest(
        dataset_id=job.dataset_id,
        version=job.version,
        shards=records,
        metadata={
            **job.metadata,
            "source": "hf_prebuilt_webdataset",
            "repo_id": job.repo_id,
            "repo_type": job.repo_type,
            "endpoint": job.endpoint,
            "source_revision": plan.revision,
            "member_policy": "preserve",
            "num_samples": samples,
            "tar_bytes": tar_bytes,
            "auxiliary_files": auxiliary,
        },
    )
    manifest_path = write_manifest(job.output_dir / "manifest.json", manifest)
    index_path = write_shard_index(job.output_dir / "indexes" / job.split / "shards.jsonl", records)
    summary = {
        "status": "complete",
        "dataset_id": job.dataset_id,
        "version": job.version,
        "source_revision": plan.revision,
        "shards": len(records),
        "samples": samples,
        "tar_bytes": tar_bytes,
        "elapsed_seconds": time.time() - started_at,
        "destination": job.destination,
    }
    summary_path = job.output_dir / "upload-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_control_file(job, plan.revision, index_path, f"indexes/{job.split}/shards.jsonl")
    _publish_control_file(job, plan.revision, summary_path, "upload-summary.json")
    _publish_control_file(job, plan.revision, manifest_path, "manifest.json")
    return manifest
