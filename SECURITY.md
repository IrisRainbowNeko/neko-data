# Security

Please do not report credentials, access keys, private dataset URLs, or sample
content in a public issue. Send a minimal reproduction without secrets to the
repository owner through a private channel.

The library never writes cloud credentials to manifests. Configure boto3 using
the standard environment, profile, or instance role mechanisms. Treat cached
shards as sensitive copies of the source dataset and set filesystem
permissions accordingly.

