"""Command line entry point for dataset builds and manifest inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..build.job import BuildJob, build_dataset
from ..runtime.manifest_loader import ManifestLoader


def _build(args: argparse.Namespace) -> int:
    manifest = build_dataset(BuildJob.from_yaml(args.config))
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
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
    parser = argparse.ArgumentParser(prog="neko-data", description="Build and read streaming diffusion datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="stream an input source into WebDataset and Parquet")
    build.add_argument("--config", type=Path, required=True, help="YAML BuildJob configuration")
    build.set_defaults(function=_build)
    inspect = subparsers.add_parser("inspect", help="show manifest summary")
    inspect.add_argument("manifest", help="local path or s3/http manifest URI")
    inspect.set_defaults(function=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())

