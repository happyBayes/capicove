# ADR-0006: Provide One Unified Local Video Filter

- Status: Accepted
- Date: 2026-08-11
- Owners: Capicove maintainers

## Context

Video quality screening needs metadata, motion, OCR-area, visual quality, safety,
watermark, tag, and frame-text metrics. Independent Tools would repeatedly accept,
probe, decode, and sample the same immutable source while exposing inconsistent
aggregation and error semantics. Model-backed criteria also create a network and
remote-code boundary that must remain explicit.

## Decision

Capicove exposes one versioned `capicove.video.filter` Tool with Python, CLI, and MCP
translations. A request contains one or more immutable video items and one or more
tagged criteria. Criteria aggregate with `all` by default; each criterion separately
chooses whether any or all requested videos must pass.

A transport-neutral pipeline probes each accepted source once and reuses bounded
sampled frames by source hash and normalized sampling configuration. Criterion
implementations depend on capability-owned ports. ffprobe, PyAV, OpenCV, EasyOCR,
PyTorch, Transformers, Pillow, and RAM+ remain private, lazy optional adapters.

Model resolution accepts only local paths or entries already present in an approved
local cache. The Tool declares `network_access=False`, never enables remote code,
and never downloads a package, model, vocabulary, or weight. Missing local assets
return an actionable unavailable failure.

If any required video or evaluator fails, the operation fails with `keep=None`.
Diagnostics completed before failure remain visible, but incomplete criterion or
final decisions are not emitted. The operation is read-only and never deletes,
moves, quarantines, or rewrites source evidence.

## Consequences

- Shared source, metadata, and frame work reduces repeated decoding and inference.
- Every decision carries source, sampling, configuration, adapter, and model
  lineage through all transports.
- Optional media and model packages do not affect base-package imports.
- Callers must install providers and make model assets local before invocation.
- Thresholds remain configuration-specific and are not assumed portable across
  backends, sampling rates, model revisions, or score scales.
- One failed item cannot be hidden by permissive `any` aggregation.

## Alternatives Considered

- One Tool per filter: rejected because it duplicates source work and fragments
  aggregation, failure, and transport contracts.
- One monolithic implementation: rejected because it couples optional dependencies
  and prevents evaluator replacement behind public ports.
- Automatic Hugging Face downloads: rejected because runtime network access must be
  explicit and the Tool is designed for reproducible local execution.
- Partial decisions after evaluator failure: rejected because `keep` would be based
  on incomplete evidence.

## References

- Video filter design (archived development reference)
- Video filter implementation plan (archived development reference)
- [Module boundaries](../architecture/module-boundaries.md)
- [ADR-0004: Declare Tool resource requirements](0004-tool-resource-requirements.md)
