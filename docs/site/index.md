# Capicove developer documentation

Start with a [five-minute offline task](getting-started/quickstart.md), then use
the [capability matrix](getting-started/capabilities.md) to select a workflow and
check its dependencies. Select documentation matching the installed version;
local previews can include uncommitted changes. Record the package version and
checkout revision when reporting a problem.

Capicove is local-first multimodal processing infrastructure for humans and AI
agents. This site documents the implemented capabilities in the 0.1.0 source
release, including their package, model, and external-service requirements.

!!! note "Source release and package availability"

    Version 0.1.0 package archives have been built and checked but are not yet
    published to PyPI. See [0.1.0 package status](getting-started/release-status.md)
    and [installation](getting-started/installation.md) for published-index and
    local-candidate options. Package publication and runtime asset setup are
    separate steps; see the [release procedure](development/releasing.md) and
    [deployment guide](guides/deployment.md).

## What you can build

- Import sources, search a native knowledge workspace and ask questions with
  source citations through the [native knowledge workflow](guides/native-knowledge.md).
- Evaluate, approve and roll back versioned skills through
  [supervised skill evolution](guides/skill-evolution.md).
- Split and filter local video while preserving traceable result metadata.
- Filter, map, deduplicate, compare, and OCR local images.
- Import immutable audio evidence and transcribe it with a local Whisper.cpp
  backend.
- Run resumable directory batches and export model-neutral Agent operation traces.
- Build deterministic, lineage-preserving data plans and execute their reviewed
  operators with the optional local Spark plugin.
- Build and search an inspectable standalone Wiki from immutable local Markdown
  snapshots without connecting to Temporal Memory.
- Store scoped Episodes and time-aware facts in a private local Temporal Memory,
  then recall cited context across world time and observation time.
- Use backend-neutral retrieval contracts and evaluate or cluster retrieval results.
- Compose document graph builds with local models or an explicitly selected
  OpenAI-compatible chat service.
- Invoke supported canonical Tool contracts through Python, CLI, and MCP
  boundaries; capability-specific commands may adapt a public service directly.

Start with [installation](getting-started/installation.md) or go directly to a
capability guide.

Code search, unified multimodal retrieval, Remote MCP and Wiki v3 maintenance
remain preview or experimental capabilities. See the
[roadmap](https://github.com/happyBayes/capicove/blob/main/ROADMAP.md#coming-soon)
and [known limitations](reference/known-limitations.md) for their current boundaries.
