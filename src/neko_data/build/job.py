"""Declarative dataset build job and resumable-friendly output layout."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contract import DatasetManifest, load_manifest, write_manifest, write_shard_index
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
    group_key: str | None = None

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
        elif source_type == "python":
            options = {key: value for key, value in source_config.items() if key != "type"}
            factory = options.pop("factory", None)
            if not factory:
                raise ValueError("python source requires factory: 'module:function'")
            source = _load_factory(factory)(**options.pop("kwargs", {}), **options)
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
            group_key=build.get("group_key"),
        )


def _load_factory(spec: str):
    import importlib

    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"python source factory must look like 'module:function', got {spec!r}")
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def _journal_path(reports_dir: Path, split: str) -> Path:
    """Per-split journal; a legacy single-split journal is reused when it matches."""
    path = reports_dir / f"build-journal-{split}.jsonl"
    legacy = reports_dir / "build-journal.jsonl"
    if not path.exists() and legacy.exists():
        records = [json.loads(line) for line in legacy.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records and all(record.get("split") == split for record in records):
            return legacy
    return path


def _other_split_state(root: Path, job: BuildJob) -> tuple[list[ShardRecord], dict[str, Any]]:
    """Shards and metadata already published for other splits of the same dataset."""
    path = root / "manifest.json"
    if not path.exists():
        return [], {}
    existing = load_manifest(path)
    if existing.dataset_id != job.dataset_id or existing.version != job.version:
        return [], {}
    return [shard for shard in existing.shards if shard.split != job.split], dict(existing.metadata)


def _rewrite_errors(path: Path, split: str) -> None:
    """Drop error lines of ``split`` while keeping those of other splits."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if line.strip() and json.loads(line).get("split", split) != split:
                kept.append(line + "\n")
        except json.JSONDecodeError:
            continue
    path.write_text("".join(kept), encoding="utf-8")


def build_dataset(job: BuildJob) -> DatasetManifest:
    root = job.output_dir.resolve()
    tar_dir = root / "wds" / job.split
    metadata_dir = root / "metadata" / job.split
    reports_dir = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    journal_path = _journal_path(reports_dir, job.split)
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
        group_key=job.group_key,
    )
    skipped = completed_samples
    total_samples = completed_samples
    error_path = reports_dir / "errors.jsonl"
    if not job.resume or job.overwrite:
        _rewrite_errors(error_path, job.split)
    source_iterator = iter(job.source)
    with error_path.open("a", encoding="utf-8") as errors:
        while True:
            try:
                sample = next(source_iterator)
            except StopIteration:
                break
            except (TypeError, ValueError, UnicodeError) as exc:
                if not job.skip_invalid_samples:
                    raise
                errors.write(json.dumps({"sample_key": None, "split": job.split, "error": str(exc)},
                                        ensure_ascii=False) + "\n")
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
                    "split": job.split,
                    "error": str(exc),
                }, ensure_ascii=False) + "\n")
    if skipped:
        raise RuntimeError(f"Resume journal contains {skipped} samples not present in the input stream")

    records = previous_records + writer.close()
    other_records, existing_metadata = _other_split_state(root, job)
    split_stats = dict(existing_metadata.get("split_stats", {}))
    split_stats[job.split] = {"num_samples": total_samples, "resumed_samples": completed_samples,
                              "shards": len(records)}
    all_records = other_records + records
    manifest = DatasetManifest(
        dataset_id=job.dataset_id,
        version=job.version,
        shards=all_records,
        metadata={
            **existing_metadata,
            **job.metadata,
            "num_samples": sum(record.num_samples for record in all_records),
            "resumed_samples": completed_samples,
            "split_stats": split_stats,
        },
    )
    manifest_path = write_manifest(root / "manifest.json", manifest)
    index_path = write_shard_index(root / "indexes" / job.split / "shards.jsonl", records)
    summary = {"dataset_id": job.dataset_id, "version": job.version, "split": job.split,
               "samples": total_samples, "shards": len(records), "resumed_samples": completed_samples}
    (reports_dir / f"build-summary-{job.split}.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                 encoding="utf-8")
    (reports_dir / "build-summary.json").write_text(
        json.dumps({"dataset_id": job.dataset_id, "version": job.version,
                    "samples": manifest.metadata["num_samples"], "shards": len(all_records),
                    "resumed_samples": completed_samples, "splits": split_stats}, indent=2) + "\n",
        encoding="utf-8",
    )
    if job.publisher:
        job.publisher.publish_index(index_path, job.split)
        job.publisher.publish_manifest(manifest_path)
    return manifest
