# Choose a capability

This matrix describes the implemented 0.1.0 interfaces. A registered command does not
provision its providers, models, or services. Platform and provider qualification
remain explicit release gates. Record `git rev-parse HEAD` with local test results
and note any uncommitted changes.

| Task and guide | Entry point | Requirements and boundary |
| --- | --- | --- |
| [Installation check](installation.md) | `capicove doctor` | Base package; environment diagnostics. A prepared 0.1.0 candidate is available locally; index publication is a separate step. |
| [Image processing](../guides/image/overview.md) | `capicove image run`, [API](../api/image.md) | Six default local tools; other tools need host-configured providers. |
| [Video](../guides/video/overview.md) | `capicove video`, [API](../api/video.md) | Decoder/FFmpeg or model/provider according to operation; exact-file deduplication needs the `video-dedup` extra, not an embedding model. |
| [Audio](../guides/audio/overview.md) | `capicove audio`, [API](../api/audio.md) | Local import and sample-accurate clipping with FFmpeg; transcription needs local Whisper.cpp runtime and model. |
| [Directory batches](../guides/directory-batches.md) | `capicove batch run`, [API](../api/batch.md) | Explicit roots and supported canonical Tool. |
| [Agent traces](../guides/agent-trace-export.md) | `capicove traces export`, [API](../api/export.md) | Existing trace store and explicit export/redaction configuration. |
| [PDF books](../guides/pdf-book-fast-reading.md) | `capicove book` | Optional parser/runtime/local models and Agent-supplied judgments; profile exclusions apply. |
| [Model installation](../guides/pdf-book-fast-reading.md#install-the-optional-parser) | `capicove models` | Explicit network opt-in; inspect dry-run first. Image models also have a separate download executable. |
| [Offline workspace search](../guides/offline-retrieval.md) | `capicove source import`, `capicove index build`, `capicove search --runtime-config`, `capicove get-evidence --runtime-config` | Version-2 workspace config with `capabilities.retrieval`; SQLite FTS5; Markdown text and image evidence; qualified on Linux, native macOS qualification pending. |
| [Workspace video and audio Tools](../guides/workspace-media.md) | `capicove video … --runtime-config`, `capicove audio import`, `capicove audio transcribe --runtime-config`, MCP | Version-2 config with `capabilities.video` or `capabilities.audio`; configured FFmpeg/FFprobe; Whisper.cpp and pinned models for transcription; Linux (deduplicate also on other POSIX). |
| [Operator jobs, traces, and models](../guides/operator-jobs.md) | `capicove jobs`, `capicove traces … --runtime-config`, `capicove models plan\|verify\|install --runtime-config`, MCP, HTTP | Version-2 config with `capabilities.batch`, `trace`, or `model`; an operator-started `capicove jobs worker`; installation needs an admin network profile and an allowlisted plan digest; Linux/macOS (batch video on Linux). |
| [Loopback HTTP](../cli/http.md) | `capicove http serve` | `api` extra; serves the selected Tools on 127.0.0.1 with a bearer token; Linux only. |
| [Experimental video retrieval](../guides/video/retrieval.md) | `capicove search`, `capicove get-evidence`, [API](../api/retrieval.md) | Explicit index, encoder, authorization and evidence wiring; not an unconfigured end-to-end service. |
| [Local Wiki](../guides/local-wiki.md) | `capicove wiki`, [API](../api/wiki.md) | `wiki` extra; offline workspace separate from Temporal Memory. |
| [Native knowledge](../guides/native-knowledge.md) | `capicove kb`, query-only MCP/HTTP host | Linux; explicit model configuration for dense retrieval and generated answers; text-only mode returns evidence. |
| [Temporal Memory](../guides/temporal-memory.md) | `capicove memory`, [API](../api/memory.md) | Accepted local SQLite contracts; provide a private runtime and explicit workspace/scope authorization. Recall is lexical by default, not general question answering. |
| [Data plans](../guides/data-processing.md) | [Python API](../api/processing.md) | Snapshots, operator catalog and execution context; no processing CLI. |
| [Spark](../guides/spark-execution.md) | Python plugin | Separate plugin and Java 17; reviewed local operators only. |
| [Document graphs](../guides/document-graph-pipeline.md) | Python composition; `capicove setup check-llm` probes a selected compatible chat endpoint | Graph builds use host authorization, profiles, tokenizer and assets. The probe sends an authored readiness request; it does not build a graph. There is no graph-build CLI/MCP command. |
| [Skill evolution](../api/evolution.md) | `capicove evolution`, Python API | POSIX ownership support (Linux or macOS); explicit evaluation and admission; no automatic promotion authority. |

Select only the [optional dependencies](optional-dependencies.md) you need and
read [Privacy and network behavior](../reference/privacy-and-network.md).

## Preview and experimental work

Code search, unified multimodal and native audio retrieval, Remote MCP and Wiki v3
maintenance are tracked in the
[roadmap](https://github.com/happyBayes/capicove/blob/main/ROADMAP.md#coming-soon).
Read [known limitations](../reference/known-limitations.md) before enabling them;
their presence in the source checkout does not establish production readiness.
