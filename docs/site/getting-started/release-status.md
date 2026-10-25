# 0.1.0 package status

Capicove 0.1.0 is the first public source release, dated October 25, 2026. The
core and five companion distributions share version 0.1.0, have built wheel and
source archives, and passed the recorded package checks. The packages have **not** been published to PyPI.
Install by package name only after they appear in an approved package index.

Read the [known limitations](../reference/known-limitations.md) before enabling
preview features. Source publication does not resolve the reported defects or
complete production qualification.

| Distribution | Capability |
| --- | --- |
| `capicove` | Core CLI, libraries and optional extras |
| `capicove-execution-spark` | Local Spark execution plugin |
| `capicove-processor-paddleocr` | PDF and Book parser plugin |
| `capicove-retriever-elasticsearch` | Elasticsearch retriever plugin |
| `capicove-retriever-opensearch` | OpenSearch retriever plugin |
| `capicove-retriever-faiss` | FAISS retriever plugin |

## Install after publication

When all six distributions are available in the package index selected for your
environment, follow the [installation guide](installation.md). A package extra
installs Python dependencies; model files, credentials, native programs and
external services remain explicit selections.

## Install a reviewed local candidate

Developers and operators with access to the repository's local release artifacts
can prepare a hash-locked offline wheel bundle using the
[deployment guide](../guides/deployment.md). A local candidate is tied to its
reviewed wheel hashes and target OS/Python version; it is not a substitute for a
published immutable release.

The root [CHANGELOG](https://github.com/happyBayes/capicove/blob/main/CHANGELOG.md)
describes the release changes. See [Releasing](../development/releasing.md) for
account configuration and publication gates.
