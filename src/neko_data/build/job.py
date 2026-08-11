"""Declarative dataset build job and resumable-friendly output layout."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contract import DatasetManifest, write_manifest, write_shard_index
from ..contract.schema import ShardRecord
from .metadata import DuckDBMetadataProvider
from .publisher import DatasetPublisher
from .shard_writer_stream import StreamingWebDatasetShardWriter
from .source_hf_dataset import HFDatasetsSource
from .source_hf_webdataset import HFWebDatasetSource
from .source_url_parquet import URLParquetSource


@dataclass
class BuildJob:
    source: Iterable
    output_dir: Path
    dataset_id: str
    version: str
    split: str = "train"
    max_shard_size: int | str = "1GiB"
    max_samples_per_shard: int = 0
    shard_pattern: str = "{split}-{index:06d}.tar"
    start_index: int = 0
    resume: bool = False
    skip_invalid_samples: bool = True
    overwrite: bool = False
    publisher: DatasetPublisher | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "BuildJob":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML job files require PyYAML") from exc
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        dataset = config.get("dataset", config)
        source_config = config.get("source", {})
        provider_config = config.get("metadata_provider")
        provider = None
        if provider_config:
            if provider_config.get("type", "duckdb") != "duckdb":
                raise ValueError(f"Unsupported metadata provider: {provider_config.get('type')}")
            provider = DuckDBMetadataProvider(
                provider_config["path"],
                table=provider_config.get("table", "data"),
                index_key=provider_config.get("index_key", "id"),
            )
        source_type = source_config.get("type", "hf_webdataset")
        if source_type == "hf_webdataset":
            source = HFWebDatasetSource(metadata_provider=provider, **{
                key: value for key, value in source_config.items() if key != "type"
            })
        elif source_type == "hf_dataset":
            source = HFDatasetsSource(metadata_provider=provider, **{
                key: value for key, value in source_config.items() if key != "type"
            })
        elif source_type == "url_parquet":
            options = {key: value for key, value in source_config.items() if key != "type"}
            paths = options.pop("parquet_paths", options.pop("path", None))
            if paths is None:
                raise ValueError("url_parquet source requires parquet_paths")
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            source = URLParquetSource(paths, **options)
        else:
            raise ValueError(f"Unsupported build source type: {source_type}")
        publish_config = config.get("publish")
        publisher = None
        if publish_config:
            publisher = DatasetPublisher(
                publish_config["destination"],
                delete_after_upload=publish_config.get("delete_after_upload", False),
                upload_concurrency=publish_config.get("upload_concurrency", 8),
            )
        build = config.get("build", {})
        return cls(
            source=source,
            output_dir=Path(dataset.get("output_dir", "./dataset")),
            dataset_id=str(dataset["dataset_id"]),
            version=str(dataset["version"]),
            split=str(dataset.get("split", source_config.get("split", "train"))),
            max_shard_size=build.get("max_shard_size", "1GiB"),
            max_samples_per_shard=int(build.get("max_samples_per_shard", 0)),
            shard_pattern=build.get("shard_pattern", "{split}-{index:06d}.tar"),
            start_index=int(build.get("start_index", 0)),
            resume=bool(build.get("resume", False)),
            skip_invalid_samples=bool(build.get("skip_invalid_samples", True)),
            overwrite=bool(build.get("overwrite", False)),
            publisher=publisher,
            metadata=dict(dataset.get("metadata", {})),
        )


def build_dataset(job: BuildJob) -> DatasetManifest:
    root = job.output_dir.resolve()
    tar_dir = root / "wds" / job.split
    metadata_dir = root / "metadata" / job.split
    reports_dir = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    journal_path = reports_dir / "build-journal.jsonl"
    previous_records: list[ShardRecord] = []
    if journal_path.exists() and not job.overwrite:
        if not job.resume:
            raise FileExistsError(f"Build journal exists: {journal_path}; set resume=True or overwrite=True")
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous_records.append(ShardRecord.from_dict(json.loads(line)))
    elif job.overwrite:
        journal_path.unlink(missing_ok=True)

    completed_samples = sum(record.num_samples for record in previous_records)
    next_index = job.start_index
    if previous_records:
        last_index = max(
            record.index if record.index is not None else -1
            for record in previous_records
        )
        next_index = max(next_index, last_index + 1)

    def on_shard(record, tar_path, metadata_path):
        if job.publisher is not None:
            job.publisher.publish_shard(record, tar_path, metadata_path)
        with journal_path.open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

    writer = StreamingWebDatasetShardWriter(
        tar_dir,
        metadata_dir,
        split=job.split,
        max_shard_size=job.max_shard_size,
        max_samples_per_shard=job.max_samples_per_shard,
        pattern=job.shard_pattern,
        start_index=next_index,
        overwrite=job.overwrite or job.resume,
        on_shard=on_shard,
    )
    skipped = completed_samples
    total_samples = completed_samples
    error_path = reports_dir / "errors.jsonl"
    error_mode = "a" if job.resume and not job.overwrite else "w"
    source_iterator = iter(job.source)
    with error_path.open(error_mode, encoding="utf-8") as errors:
        while True:
            try:
                sample = next(source_iterator)
            except StopIteration:
                break
            except (TypeError, ValueError, UnicodeError) as exc:
                if not job.skip_invalid_samples:
                    raise
                errors.write(json.dumps({"sample_key": None, "error": str(exc)}, ensure_ascii=False) + "\n")
                continue
            if skipped:
                skipped -= 1
                continue
            try:
                writer.add(sample)
                total_samples += 1
            except (TypeError, ValueError, UnicodeError) as exc:
                if not job.skip_invalid_samples:
                    raise
                errors.write(json.dumps({
                    "sample_key": getattr(sample, "sample_key", None),
                    "error": str(exc),
                }, ensure_ascii=False) + "\n")
    if skipped:
        raise RuntimeError(f"Resume journal contains {skipped} samples not present in the input stream")

    records = previous_records + writer.close()
    manifest = DatasetManifest(
        dataset_id=job.dataset_id,
        version=job.version,
        shards=records,
        metadata={**job.metadata, "num_samples": total_samples, "resumed_samples": completed_samples},
    )
    manifest_path = write_manifest(root / "manifest.json", manifest)
    index_path = write_shard_index(root / "indexes" / job.split / "shards.jsonl", records)
    (reports_dir / "build-summary.json").write_text(
        json.dumps({"dataset_id": job.dataset_id, "version": job.version, "samples": total_samples,
                    "shards": len(records), "resumed_samples": completed_samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    if job.publisher:
        job.publisher.publish_manifest(manifest_path)
        job.publisher.publish_index(index_path, job.split)
    return manifest
