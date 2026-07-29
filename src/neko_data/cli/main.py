"""Command line entry point for dataset builds, mirrors, and inspection."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..build.job import BuildJob, build_dataset
from ..build.mirror import MirrorJob, inspect_mirror, mirror_dataset, mirror_status
from ..runtime.manifest_loader import ManifestLoader


def _build(args: argparse.Namespace) -> int:
    manifest = build_dataset(BuildJob.from_yaml(args.config))
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _configured_mirror_job(args: argparse.Namespace) -> MirrorJob:
    job = MirrorJob.from_yaml(args.config)
    updates = {}
    if args.workers is not None:
        updates["workers"] = args.workers
    if args.resume:
        updates["resume"] = True
    if args.destination:
        updates["destination"] = args.destination
    if args.output_dir:
        output_dir = args.output_dir
        updates["output_dir"] = output_dir
        updates["staging_dir"] = output_dir / "staging"
    if args.limit is not None:
        updates.update({
            "expected_shards": None,
            "expected_samples": None,
            "expected_tar_bytes": None,
        })
    return replace(job, **updates)


def _mirror(args: argparse.Namespace) -> int:
    job = _configured_mirror_job(args)
    if args.status:
        print(json.dumps(mirror_status(job), ensure_ascii=False, indent=2))
        return 0
    copy_auxiliary = not args.skip_auxiliary
    if not args.execute:
        plan = inspect_mirror(job, limit=args.limit, copy_auxiliary=copy_auxiliary)
        print(json.dumps({
            "dataset_id": job.dataset_id,
            "version": job.version,
            "source_revision": plan.revision,
            "destination": job.destination,
            "shards": len(plan.shards),
            "tar_bytes": plan.tar_bytes,
            "max_shard_bytes": max(item.size_bytes for item in plan.shards),
            "workers": job.workers,
            "auxiliary_files": [item.path for item in plan.auxiliary],
            "execute": False,
        }, ensure_ascii=False, indent=2))
        return 0
    manifest = mirror_dataset(job, limit=args.limit, copy_auxiliary=copy_auxiliary)
    print(json.dumps({
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "splits": manifest.splits,
        "shards": len(manifest.shards),
        "destination": job.destination,
    }, ensure_ascii=False, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    loaded = ManifestLoader().load(args.manifest)
    print(json.dumps({
        "dataset_id": loaded.manifest.dataset_id,
        "version": loaded.manifest.version,
        "base_uri": loaded.base_uri,
        "splits": loaded.manifest.splits,
        "shards": len(loaded.manifest.shards),
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neko-data", description="Build and read streaming training datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="stream an input source into WebDataset and Parquet")
    build.add_argument("--config", type=Path, required=True, help="YAML BuildJob configuration")
    build.set_defaults(function=_build)

    mirror = subparsers.add_parser("mirror", help="mirror prebuilt Hub WebDataset shards without repacking")
    mirror.add_argument("--config", type=Path, required=True, help="YAML MirrorJob configuration")
    mirror.add_argument("--workers", type=int, help="override the configured process count")
    mirror.add_argument("--execute", action="store_true", help="perform uploads; default is a read-only plan")
    mirror.add_argument("--resume", action="store_true", help="resume and verify completed objects")
    mirror.add_argument("--status", action="store_true", help="show local journal progress without network access")
    mirror.add_argument("--limit", type=int, help="process only the first N matched shards")
    mirror.add_argument("--skip-auxiliary", action="store_true", help="do not mirror root metadata files")
    mirror.add_argument("--destination", help="override the destination URI, useful for a canary")
    mirror.add_argument("--output-dir", type=Path, help="override local journals and staging directory")
    mirror.set_defaults(function=_mirror)

    inspect = subparsers.add_parser("inspect", help="show manifest summary")
    inspect.add_argument("manifest", help="local path or s3/http manifest URI")
    inspect.set_defaults(function=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
