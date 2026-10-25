# Changelog

Notable changes are recorded by version and date. Source releases and package-index
publication are tracked separately.

## [0.1.0] - 2026-10-25

First public source release of Capicove: local-first knowledge, multimodal
processing and persistent context for humans and AI agents.

### Added

- Native knowledge workspaces with imports, hybrid search, source-linked answers,
  source updates and revocation. The qualified platform is Linux.
- A standalone Markdown Wiki with preserved sources and rebuildable search, plus
  page-aware PDF book parsing and agent-assisted summaries with validated citations.
- Temporal Memory for scoped episodes, time-aware facts and evidence-backed recall.
- Supervised skill evolution with version history, candidate evaluation, explicit
  approval and rollback.
- Image filtering, deduplication, OCR and visual retrieval; audio import,
  transcription and sample-accurate clipping; video splitting, filtering and
  deduplication through capability-specific runtimes.
- Resumable directory batches, operator jobs and opt-in agent trace export.
- CLI and Python interfaces, local MCP stdio integration and a loopback workspace
  HTTP host with explicit permissions and source roots.
- Configurable model services for embeddings, reranking, OCR, PDF parsing,
  transcription and compatible generation, including VLM tool conversations and
  explicit multimodal token counting.
- Companion plugins for PaddleOCR, FAISS, Elasticsearch, OpenSearch and local Spark
  execution, all versioned 0.1.0.
- Hash-locked offline deployment bundles, package inventories and SHA-256 checksums,
  installed-wheel checks, and a gated Trusted Publishing workflow.

### Fixed

- Empty VLM completions, request deadlines and HTTP framing validation; large
  parsing and serialization workloads no longer block health endpoints.
- Windows readiness diagnostics and portable delivery fixtures.
- Elasticsearch vector mapping stability through explicit HNSW parameters, and
  OpenSearch alias operations through configured generation-prefix restrictions.
- Implicit EasyOCR detector downloads; missing assets now require explicit local
  provisioning. RAM++ also uses explicitly provisioned vocabulary and tokenizer assets.

### Security

- Updated MCP, Pillow, certifi, sentence-transformers, aiohttp, urllib3 and development
  dependency baselines; resolved installation profiles are audited separately.
- Explicit network opt-in, verified model assets and capability-specific access
  controls. Model weights, private data and credentials are excluded from packages.

### Compatibility

- Requires Python 3.11 or later. Platform, model and native-runtime requirements
  vary by workflow; see the [capability matrix](docs/site/getting-started/capabilities.md).
- Rebuild BGE indexes created with preprocessing identity `st5.2-v1`; the updated
  identity is `st5.6-v2`.
- Rebuild Elasticsearch vector generations created before the HNSW mapping correction.

### Release scope

- This entry records the source release. PyPI publication remains a separate step;
  see [package status](docs/site/getting-started/release-status.md).
- Code search, unified multimodal retrieval, native audio search, Remote MCP and
  Wiki v3 maintenance remain preview or experimental work. See
  [Coming soon](ROADMAP.md#coming-soon) and [known limitations](docs/site/reference/known-limitations.md)
  before trying them; source availability does not establish production readiness.
- Model assets, system runtimes and external services must be provisioned separately.
  See [installation](docs/site/getting-started/installation.md) and
  [deployment](docs/site/guides/deployment.md).
