# Build, Publish, and Train Operations

Install the extras and configure credentials through the environment or a
boto3 profile. Never put tokens or signed URLs in YAML or Git. Start with a
small tar fixture and a small shard size before running the full build.

`HFWebDatasetSource` lists matching tars and processes each with one streaming
HTTP request and `tarfile.open(mode="r|*")`. A normal HF source uses
`load_dataset(..., streaming=True)`. A `DuckDBMetadataProvider` joins by the
sample key during build; it is not opened by the training reader.

Use one cache directory visible to every rank on a node, preferably on local
NVMe. Configure approximately 420 GiB maximum and 350 GiB eviction target on
a 500 GiB machine. The cache verifies downloaded size and SHA-256, atomically
renames completed files, and uses leases so active shards are not evicted.

Use `required` mode for normal training, `preferred` mode if direct streaming
is an acceptable outage fallback, and `disabled` only for debugging. A
manifest is immutable once published; make a new dataset version for changed
captions, filters, or images.

## Resume and error handling

Set `build.resume: true` only when the input stream is stable and replayable.
The build journal contains finalized shards; resume replays the source and
skips the samples already represented by that journal. Do not resume against a
source whose ordering or filtering changed.

`skip_invalid_samples: true` records `TypeError`, `ValueError`, and
`UnicodeError` from sample normalization in `reports/errors.jsonl` and keeps
building. Publisher failures and storage/network I/O errors remain fatal. Use
`overwrite: true` for an intentional rebuild instead of mixing old records
with a changed source.

## HTTP fallback

Public `http://` and `https://` manifests and shards are supported through
`HTTPStorage`; it is selected automatically when no storage object is passed.
HTTP reads are streamed, and the normal cache still downloads whole assigned
tar shards before training. This is useful for public mirrors; for private R2,
use `s3://` with the `r2` extra and credentials from the environment.
