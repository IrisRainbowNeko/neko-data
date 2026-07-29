"""S3-compatible storage adapter, including Cloudflare R2."""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme not in {"s3", "r2"} or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


class S3Storage:
    """Minimal boto3 adapter with the operations used by neko-data.

    The client is created lazily so importing neko-data does not require boto3
    when all data is local.  R2 uses the same API with ``region_name='auto'``.
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        region_name: str = "auto",
        profile: str | None = None,
        enable_multipart: bool | None = None,
        client=None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ.get("R2_ENDPOINT") or os.environ.get("S3_ENDPOINT_URL")
        self.region_name = region_name
        self.profile = profile
        self.enable_multipart = enable_multipart
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("S3/R2 support requires the 'r2' extra: pip install neko-data[r2]") from exc
            session_kwargs = {"profile_name": self.profile} if self.profile else {}
            session = boto3.Session(**session_kwargs)
            client_kwargs = {"region_name": self.region_name}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            self._client = session.client("s3", **client_kwargs)
        return self._client

    @contextmanager
    def open(self, uri: str):
        bucket, key = _split_s3_uri(uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            yield body
        finally:
            body.close()

    def download(self, uri: str, destination: Path) -> None:
        bucket, key = _split_s3_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(bucket, key, str(destination))

    def upload_file(
        self,
        source: Path,
        uri: str,
        max_concurrency: int = 8,
        *,
        enable_multipart: bool | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        bucket, key = _split_s3_uri(uri)
        multipart = self.enable_multipart if enable_multipart is None else enable_multipart
        object_metadata = {str(name): str(value) for name, value in (metadata or {}).items()}
        if multipart is None:
            value = os.environ.get("R2_ENABLE_MULTIPART") or os.environ.get("S3_ENABLE_MULTIPART")
            multipart = value is None or value.strip().lower() not in {"0", "false", "no", "off"}
        if not multipart:
            request = {
                "Bucket": bucket,
                "Key": key,
                "ContentLength": source.stat().st_size,
            }
            if object_metadata:
                request["Metadata"] = object_metadata
            with source.open("rb") as body:
                self.client.put_object(Body=body, **request)
            return
        try:
            from boto3.s3.transfer import TransferConfig
        except ImportError as exc:
            raise RuntimeError("S3/R2 support requires boto3") from exc
        config = TransferConfig(max_concurrency=max_concurrency, use_threads=max_concurrency > 1)
        kwargs = {"Config": config}
        if object_metadata:
            kwargs["ExtraArgs"] = {"Metadata": object_metadata}
        self.client.upload_file(str(source), bucket, key, **kwargs)

    def upload_bytes(
        self,
        data: bytes,
        uri: str,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        bucket, key = _split_s3_uri(uri)
        object_metadata = {str(name): str(value) for name, value in (metadata or {}).items()}
        if self.enable_multipart is False:
            request = {"Bucket": bucket, "Key": key, "ContentLength": len(data)}
            if object_metadata:
                request["Metadata"] = object_metadata
            self.client.put_object(Body=io.BytesIO(data), **request)
            return
        kwargs = {}
        if object_metadata:
            kwargs["ExtraArgs"] = {"Metadata": object_metadata}
        self.client.upload_fileobj(io.BytesIO(data), bucket, key, **kwargs)

    def head(self, uri: str) -> dict[str, object]:
        bucket, key = _split_s3_uri(uri)
        response = self.client.head_object(Bucket=bucket, Key=key)
        return {
            "size_bytes": int(response["ContentLength"]),
            "metadata": {
                str(name).lower(): str(value)
                for name, value in response.get("Metadata", {}).items()
            },
            "etag": str(response.get("ETag", "")).strip('"'),
        }
