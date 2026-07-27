import io
import json
import tarfile
from contextlib import contextmanager

from neko_data.build.source_hf_webdataset import HFWebDatasetSource


def _make_tar(path, image):
    with tarfile.open(path, "w") as archive:
        for name, data in {
            "42.jpg": image,
            "42.json": json.dumps({"tags": "one, two", "regular_summary": "a scene", "width": 32}).encode(),
            "42.txt": b"default caption\n",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_hf_webdataset_source_reads_local_tar_as_stream(sample_image_bytes, tmp_path):
    tar_path = tmp_path / "input.tar"
    _make_tar(tar_path, sample_image_bytes)
    source = HFWebDatasetSource(input_files=[str(tar_path)], split="train", source_id="fixture")
    samples = list(source)
    assert len(samples) == 1
    assert samples[0].sample_key == "42"
    assert samples[0].select_caption("tags") == "one, two"
    assert samples[0].select_caption("regular_summary") == "a scene"
    assert samples[0].caption == "default caption"
    assert samples[0].width == 32

def test_hf_webdataset_source_accepts_non_seekable_hub_stream(sample_image_bytes, tmp_path):
    tar_path = tmp_path / "input.tar"
    _make_tar(tar_path, sample_image_bytes)
    tar_bytes = tar_path.read_bytes()

    class NonSeekableStream(io.BytesIO):
        def seekable(self):
            return False

        def seek(self, *args, **kwargs):
            raise io.UnsupportedOperation("stream is not seekable")

    class StreamingSource(HFWebDatasetSource):
        @contextmanager
        def open_file(self, path):
            yield NonSeekableStream(tar_bytes)

    samples = list(StreamingSource(input_files=["hub/sample.tar"], split="train", source_id="hub"))
    assert len(samples) == 1
    assert samples[0].sample_key == "42"
    assert samples[0].caption == "default caption"


def test_hf_webdataset_source_retries_transient_stream(sample_image_bytes, tmp_path):
    tar_path = tmp_path / "input.tar"
    _make_tar(tar_path, sample_image_bytes)

    class FailingStream(io.BytesIO):
        def read(self, *args, **kwargs):
            raise OSError("temporary stream failure")

    class NonSeekableStream(io.BytesIO):
        def seekable(self):
            return False

        def seek(self, *args, **kwargs):
            raise io.UnsupportedOperation("stream is not seekable")

    class RetryingSource(HFWebDatasetSource):
        opens = 0
        retry_tar = tar_path.read_bytes()


        @contextmanager
        def open_file(self, path):
            self.opens += 1
            if self.opens == 1:
                yield FailingStream()
            else:
                yield NonSeekableStream(self.retry_tar)

    source = RetryingSource(input_files=["hub/sample.tar"], max_retries=1, retry_backoff=0)
    samples = list(source)
    assert len(samples) == 1
    assert source.opens == 2
    assert samples[0].sample_key == "42"
    assert samples[0].caption == "default caption"
