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

