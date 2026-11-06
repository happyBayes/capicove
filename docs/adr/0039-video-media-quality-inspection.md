# ADR-0039: Keep video media quality inspection evidence-based and offline

- Status: Accepted by the maintainer on 2026-10-29 for S1 implementation
- Date: 2026-10-29
- Owners: Capicove maintainers

## Context

Sampled frame filtering does not establish whole-file decoding, audio loudness,
or temporal media integrity. Teaching-video review needs reproducible technical
measurements, while still images, dark layouts and pauses may be intentional.
Model-produced or reviewer-produced grades must not masquerade as measurements.

## Decision

Provide one read-only `capicove.video.quality` Tool, owned by video processing,
with shared Python, CLI and MCP contracts. Acceptance of
[RFC-0035](../rfcs/0035-video-media-quality-inspection.md) precedes public
implementation. The maintainer explicitly accepted both documents and authorized
S1 in the originating conversation on 2026-10-29. The maintainer authorized S2
workspace-host integration on 2026-10-30.

Measure local selected streams using FFprobe and full-file FFmpeg analysis. Keep
provider processes private and optional, schemas independent of provider SDKs,
and runtime network access disabled and enforced. Preserve source identity,
stream selection, normalized configuration, backend versions, timing coverage,
typed measurements, temporal events and incomplete diagnostics.

Separate execution success, detector observations, explicit policy verdicts and
human acceptance. Do not emit a default 1–10 total score. Black/static/silent
intervals are review observations unless explicit delivery policy says otherwise.
Metadata timestamp differences do not establish semantic synchronization.

Retain existing video filter v1 behavior. Its sampled scalar criteria remain
separate from this full-file inspection contract. Reuse existing verified media
handoff and executable authority for the workspace host, including its current
Linux restriction. Qualify standalone support and host support separately.

## Consequences

- Full decoding costs more than sampled filtering and needs bounded execution,
  output draining, cancellation and honest incomplete results.
- Missing FFmpeg filters or measurements cannot silently pass or become zeros.
- Explicit policies can evaluate delivery requirements without claiming that a
  lecture slide is a playback freeze or that a black background is defective.
- A calibrated numerical score, automatic repair, semantic synchronization or
  educational grading requires a subsequent reviewed proposal and validation.
- No network model inference, model downloads or new persistent store is needed.

## Alternatives considered

- Extend filter criteria: rejected for this proposal because temporal coverage,
  stream metadata, event intervals and measured units do not fit scalar sampled
  quality scores without changing the existing contract.
- Add a Tool per detector: rejected because it duplicates intake and decoding and
  fragments coverage/failure semantics.
- Ask a VLM for a media quality grade: rejected as the primary detector because
  grades do not measure loudness, decode integrity or exact event intervals.
- Assign fixed penalty weights to produce a total score: deferred because no
  validated teaching-video calibration or audience-specific policy exists.
- Normalize/transcode while checking: rejected because it changes the measured
  signal and conflates inspection with source transformation.

## References

- [RFC-0035](../rfcs/0035-video-media-quality-inspection.md).
- [Public usage and qualification guide](../site/guides/video/quality.md).
- [ADR-0006](0006-unified-local-video-filter.md).
- [ADR-0026](0026-host-media-handoff.md).


## S2 integration amendment (2026-10-30)

The maintainer explicitly requested completing integration into the Capicove
chain. `capicove.video.quality` is selectable in the version-2 workspace host's
existing video capability schema 1. It requires both configured FFmpeg and
FFprobe, `video:read` and `subprocess:execute`; no output root is required.
Optional `quality_limits` is a closed object with the public `QualityLimits`
fields. Operator values may only narrow default ceilings and the host's input
byte ceiling. Invocation arguments and canonical contract version remain v1.

Runtime adapts the existing verified media handoff through `QualityIntake`.
Composition revalidates the staged snapshot's size, identity, SHA-256 and container
references, holds a Linux descriptor across all passes, and preserves executable
descriptor locators without resolving them to mutable filesystem paths. It does
not make a second standalone temporary copy. Reports retain original source
identity and omit private locators. Existing Linux-only host support remains;
standalone macOS support does not bypass this boundary.

Catalog/readiness, MCP stdio and configured CLI reuse the existing host binding.
Configured tracing belongs to the host configuration; the CLI rejects a second
standalone trace store. This amendment grants no remote operation qualification,
network access, semantic quality scoring, or release acceptance.
