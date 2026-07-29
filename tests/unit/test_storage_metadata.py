
from neko_data.storage.r2 import S3Storage


class Client:
    def __init__(self):
        self.put = None

    def put_object(self, **kwargs):
        self.put = {key: value for key, value in kwargs.items() if key != "Body"}
        self.body = kwargs["Body"].read()

    def head_object(self, **kwargs):
        return {
            "ContentLength": 7,
            "Metadata": {"SHA256": "abc"},
            "ETag": '"etag-value"',
        }


def test_single_put_bytes_and_files_preserve_object_metadata(tmp_path):
    client = Client()
    storage = S3Storage(client=client, enable_multipart=False)

    storage.upload_bytes(b"sidecar", "s3://bucket/sidecar.json", metadata={"sha256": "abc"})
    assert client.body == b"sidecar"
    assert client.put == {
        "Bucket": "bucket",
        "Key": "sidecar.json",
        "ContentLength": 7,
        "Metadata": {"sha256": "abc"},
    }

    source = tmp_path / "shard.tar"
    source.write_bytes(b"payload")
    storage.upload_file(source, "s3://bucket/shard.tar", metadata={"sha256": "def"})
    assert client.put["Metadata"] == {"sha256": "def"}


def test_head_returns_normalized_metadata_and_etag():
    storage = S3Storage(client=Client())

    head = storage.head("s3://bucket/object")

    assert head == {
        "size_bytes": 7,
        "metadata": {"sha256": "abc"},
        "etag": "etag-value",
    }
