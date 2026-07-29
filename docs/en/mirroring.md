# Mirroring Prebuilt WebDataset Shards

Use `neko-data mirror` when a Hub repository already contains suitably sized
WebDataset tar shards. The mirror pins the source revision, derives sample
counts and SHA-256 values from sibling JSON sidecars, and copies each tar
without decoding or repacking its members. It publishes the final manifest
only after every tar, sidecar, and auxiliary metadata object is verified.

Run a read-only plan first, then execute or inspect a resumable job:

```bash
neko-data mirror --config mirror.yaml
neko-data mirror --config mirror.yaml --workers 8 --execute --resume
neko-data mirror --config mirror.yaml --status
```

Each worker stages at most one source shard. Partial HTTP downloads use Range
requests, completed files are checked against the source SHA-256, and R2 HEAD
metadata prevents verified objects from being downloaded again. Credentials
must remain in the environment rather than the YAML file.

For RainbowNeko image-only training, use
`RainbowWebDatasetImageSource.from_manifest(...)` with
`RuntimeContext.from_env()`. It returns the same `id` and PIL `image` fields as
RainbowNeko's `WebDatasetImageSource`, while using manifest-based distributed
shard assignment, the shared node cache, and bounded prefetching.
