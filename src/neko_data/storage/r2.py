"""S3-compatible storage adapter, including Cloudflare R2."""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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
        multipart_threshold: int = 64 * 1024**2,
        multipart_chunksize: int = 64 * 1024**2,
        upload_attempts: int = 5,
        upload_retry_base_seconds: float = 1.0,
        client=None,
    ) -> None:
        if multipart_threshold <= 0 or multipart_chunksize <= 0:
            raise ValueError("multipart sizes must be positive")
        if upload_attempts <= 0 or upload_retry_base_seconds < 0:
            raise ValueError("upload retry settings are invalid")
        self.endpoint_url = endpoint_url or os.environ.get("R2_ENDPOINT") or os.environ.get("S3_ENDPOINT_URL")
        self.region_name = region_name
        self.profile = profile
        self.enable_multipart = enable_multipart
        self.multipart_threshold = multipart_threshold
        self.multipart_chunksize = multipart_chunksize
        self.upload_attempts = upload_attempts
        self.upload_retry_base_seconds = upload_retry_base_seconds
        self._client = client

    def _uploaded_object_matches(
        self,
        uri: str,
        expected_size: int,
        expected_metadata: Mapping[str, str],
    ) -> bool:
        if not expected_metadata:
            return False
        bucket, key = _split_s3_uri(uri)
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
        except Exception:  # noqa: BLE001 - a failed probe means the upload must retry
            return False
        actual_metadata = {
            str(name).lower(): str(value)
            for name, value in response.get("Metadata", {}).items()
        }
        return int(response["ContentLength"]) == expected_size and all(
            actual_metadata.get(name.lower()) == value
            for name, value in expected_metadata.items()
        )

    def _retry_upload(
        self,
        uri: str,
        operation,
        *,
        expected_size: int,
        expected_metadata: Mapping[str, str],
    ) -> None:
        for attempt in range(1, self.upload_attempts + 1):
            try:
                operation()
                return
            except Exception as error:  # noqa: BLE001 - retry the complete boto transfer session
                if self._uploaded_object_matches(uri, expected_size, expected_metadata):
                    logger.warning(
                        "S3 upload for %s raised %s, but HEAD confirms the complete object",
                        uri,
                        error,
                    )
                    return
                if attempt == self.upload_attempts:
                    raise
                delay = self.upload_retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "S3 upload failed for %s (%s); rebuilding transfer session in %.1fs (%d/%d)",
                    uri,
                    error,
                    delay,
                    attempt + 1,
                    self.upload_attempts,
                )
                time.sleep(delay)

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

    def copy(
        self,
        source_uri: str,
        destination_uri: str,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Copy an object within one bucket without routing bytes through the client."""
        source_bucket, source_key = _split_s3_uri(source_uri)
        destination_bucket, destination_key = _split_s3_uri(destination_uri)
        if source_bucket != destination_bucket:
            raise ValueError("S3Storage.copy supports same-bucket copies only")
        request = {
            "Bucket": destination_bucket,
            "Key": destination_key,
            "CopySource": {"Bucket": source_bucket, "Key": source_key},
        }
        object_metadata = {str(name): str(value) for name, value in (metadata or {}).items()}
        if object_metadata:
            request["Metadata"] = object_metadata
            request["MetadataDirective"] = "REPLACE"
        self.client.copy_object(**request)

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
            def put() -> None:
                request = {
                    "Bucket": bucket,
                    "Key": key,
                    "ContentLength": source.stat().st_size,
                }
                if object_metadata:
                    request["Metadata"] = object_metadata
                with source.open("rb") as body:
                    self.client.put_object(Body=body, **request)

            self._retry_upload(
                uri,
                put,
                expected_size=source.stat().st_size,
                expected_metadata=object_metadata,
            )
            return
        try:
            from boto3.s3.transfer import TransferConfig
        except ImportError as exc:
            raise RuntimeError("S3/R2 support requires boto3") from exc
        config = TransferConfig(
            max_concurrency=max_concurrency,
            multipart_threshold=self.multipart_threshold,
            multipart_chunksize=self.multipart_chunksize,
            use_threads=max_concurrency > 1,
        )
        kwargs = {"Config": config}
        if object_metadata:
            kwargs["ExtraArgs"] = {"Metadata": object_metadata}
        self._retry_upload(
            uri,
            lambda: self.client.upload_file(str(source), bucket, key, **kwargs),
            expected_size=source.stat().st_size,
            expected_metadata=object_metadata,
        )

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
            def put() -> None:
                request = {"Bucket": bucket, "Key": key, "ContentLength": len(data)}
                if object_metadata:
                    request["Metadata"] = object_metadata
                self.client.put_object(Body=io.BytesIO(data), **request)

            self._retry_upload(
                uri,
                put,
                expected_size=len(data),
                expected_metadata=object_metadata,
            )
            return
        kwargs = {}
        if object_metadata:
            kwargs["ExtraArgs"] = {"Metadata": object_metadata}
        self._retry_upload(
            uri,
            lambda: self.client.upload_fileobj(io.BytesIO(data), bucket, key, **kwargs),
            expected_size=len(data),
            expected_metadata=object_metadata,
        )

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
