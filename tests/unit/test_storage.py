import io
import sys
from pathlib import Path

from neko_data.storage.http import HTTPStorage
from neko_data.storage.r2 import S3Storage


class Body:
    def __init__(self, data: bytes):
        self.data = data
        self.closed = False

    def read(self, size=-1):
        data, self.data = self.data, b""
        return data

    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.body = Body(b"manifest")
        self.download_calls = []
        self.put_body = None
        self.put_kwargs = None

    def get_object(self, **kwargs):
        self.get_kwargs = kwargs
        return {"Body": self.body}

    def download_file(self, bucket, key, destination):
        self.download_calls.append((bucket, key, destination))
        Path(destination).write_bytes(b"download")


    def put_object(self, **kwargs):
        self.put_kwargs = {key: value for key, value in kwargs.items() if key != "Body"}
        self.put_body = kwargs["Body"].read()

    def head_object(self, **kwargs):
        return {"ContentLength": 8}


def test_s3_storage_manages_stream_body_and_object_paths(tmp_path):
    client = Client()
    storage = S3Storage(client=client)
    with storage.open("s3://bucket/path/file") as body:
        assert body.read() == b"manifest"
    assert client.body.closed
    destination = tmp_path / "file"
    storage.download("s3://bucket/path/file", destination)
    assert destination.read_bytes() == b"download"
    assert storage.head("s3://bucket/path/file")["size_bytes"] == 8


def test_s3_storage_can_use_single_put(tmp_path):
    client = Client()
    source = tmp_path / "shard.tar"
    source.write_bytes(b"tar payload")
    storage = S3Storage(client=client, enable_multipart=False)

    storage.upload_file(source, "s3://bucket/path/shard.tar")

    assert client.put_kwargs == {
        "Bucket": "bucket",
        "Key": "path/shard.tar",
        "ContentLength": len(b"tar payload"),
    }
    assert client.put_body == b"tar payload"

class Response:
    def __init__(self, data: bytes):
        self.raw = io.BytesIO(data)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.raw.close()
        self.closed = True

    def raise_for_status(self):
        return None


class Requests:
    def __init__(self, data: bytes):
        self.data = data
        self.calls = []
        self.responses = []

    def get(self, uri, *, stream, timeout):
        self.calls.append((uri, stream, timeout))
        response = Response(self.data)
        self.responses.append(response)
        return response


def test_http_storage_streams_and_downloads(monkeypatch, tmp_path):
    requests = Requests(b"http payload")
    monkeypatch.setitem(sys.modules, "requests", requests)
    storage = HTTPStorage(timeout=3)

    with storage.open("https://example.test/shard.tar") as stream:
        assert stream.read() == b"http payload"
    destination = tmp_path / "shard.tar"
    storage.download("https://example.test/shard.tar", destination)

    assert destination.read_bytes() == b"http payload"
    assert requests.calls == [
        ("https://example.test/shard.tar", True, 3),
        ("https://example.test/shard.tar", True, 3),
    ]
    assert all(response.closed for response in requests.responses)
