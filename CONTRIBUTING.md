# Contributing

Install development dependencies with `pip install -e '.[dev,hf,r2]'`.

Before opening a change, run:

```bash
ruff check src tests
python -m compileall -q src
pytest
```

Keep data contracts backward compatible within a schema version. A change to
manifest or tar member semantics requires a schema version decision and a
fixture covering old and new data. Do not add credentials, real Hub tokens,
R2 endpoints containing secrets, or generated dataset shards to Git.

