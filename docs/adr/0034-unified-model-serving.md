# ADR-0034: Compose model HTTP routes and reuse VLM for graph generation

- Status: Accepted under RFC-0030
- Date: 2026-10-21

Reuse the existing model registry, profiles, typed providers and capability HTTP
applications. A bounded dispatcher selects configured aliases and owns application
lifespans, startup rollback and independent model admission. It is not a second
model inference framework. GPU dependencies remain optional and locally installed.

Whisper.cpp continues to own normalization, VAD and native inference. Add an
optional decoded-duration cap to its existing adapter so uploaded compressed audio
cannot expand without limit. Preserve existing audio Tool defaults. The HTTP
boundary stages anonymous files and never trusts client filenames or media URLs.

Graph generation references a registered VLM profile and uses its transport and
token counting. Keep structured schema validation and access policy in the graph
workflow. Preserve the old configuration adapter for migration. Chunk token counts
include chat framing (conservative packing); complete inference counts use the
exact generation messages and template. Upstream revisions remain operator
attestations, not cryptographic remote-weight verification.

Legacy service commands remain compatible. The configured entry point is the
recommended path for selection, discovery, authentication and multi-model hosting.
