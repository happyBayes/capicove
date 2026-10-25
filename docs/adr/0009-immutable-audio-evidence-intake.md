# ADR-0009: Validate and Preserve Immutable Audio Evidence

- Status: Accepted
- Date: 2026-08-11
- Owners: Capicove maintainers

## Context

Issue #12 (archived development review) defines the first audio
hypothesis: one local audio file can become immutable, content-addressed evidence
without modifying or deleting the caller's source. The first implementation was
closed after review because it introduced a new top-level `capicove.source`
capability instead of following the existing media boundary under
`capicove.processing`.

Real-sample validation also showed that a file can have a supported extension and a
matching leading signature while still being undecodable. Such a file must not be
reported as accepted audio evidence. The broader audio acceptance standard requires
invalid input, evidence integrity, privacy, and capability boundaries to be explicit
before model processing begins.

## Decision

The capability lives under `capicove.processing.audio` and exposes an
`AudioProcessor` application operation. It does not create a new top-level package or
change the repository's architecture conventions.

`AudioProcessor` accepts an explicit local source path, an explicit local evidence
store root, an optional zero-based audio-stream ordinal, and injected stream
inventory and validation ports. The ordinal is the audio-stream position used by
FFmpeg `0:a:N`, not the container-global stream index.

A private FFprobe adapter counts audio streams with bounded, strictly parsed output
before evidence storage. Zero streams fail. One stream may be selected implicitly at
ordinal zero. Multiple streams require an explicit ordinal; Capicove does not guess
from default disposition or language metadata. Invalid, ambiguous, and out-of-range
selections fail before the evidence store changes.

A private FFmpeg adapter validates only the selected stream from start to finish
using a fixed argument array without a shell. Decoder absence, timeout, a missing
stream, corruption in the selected stream, or decode failure prevents evidence
storage. Capicove does not fall back to another stream. Damage in an unselected
stream does not block an explicitly selected valid stream when the container
inventory remains readable, and lineage states that validation covered only the
selected stream.

Accepted bytes use SHA-256 identity and are stored at:

```text
<store-root>/sha256/<first-two-hex>/<full-sha256>/blob
```

The returned immutable `AudioEvidence` contains the evidence identity, SHA-256,
byte size, media type, original file name, and absolute stored path. It does not
retain the full source path. Failures expose stable `AudioFailureCode` values and
bounded messages without decoder diagnostics or private source paths.

Stream selection is request lineage rather than evidence identity. The intake
operation can return `AudioEvidence` together with a versioned selection value that
records the selected ordinal, zero-based audio-stream semantics, selection method,
preflight stream count, and `selected_audio_stream_only` validation scope. The same
original container selected at two ordinals therefore shares one evidence identity
while retaining distinct processing lineage.

Imports are idempotent. Existing content is hash-verified before reuse. New content
is copied into a synchronized temporary file and atomically placed. The operation
compares source device, inode, size, modification time, and change time before and
after validation and copying. A changed source or corrupted existing blob fails
without overwriting accepted evidence. The caller's source is never modified,
moved, or deleted.

The initial boundary is local file intake and CPU media decoding. It does not declare
ASR, TTS, VAD, diarization, denoising, streaming, GPU execution, network access,
hardware ingestion, retrieval, or user-facing deletion. Those capabilities require
their own reviewed contracts and bound acceptance profiles.

## Consequences

- Later ASR and retrieval work can reference stable original bytes and lineage.
- Every accepted item has passed an inexpensive extension/signature guard, bounded
  stream inventory, and complete decoder validation of the selected stream. It does
  not claim validation of unselected streams.
- The base package keeps no Python media dependency; FFprobe and FFmpeg remain
  selected private runtime adapters, and a missing executable is a stable failure.
- Identical bytes share one stored blob, while corrupt existing storage is never
  silently repaired or overwritten.
- Runtime state and real validation media remain outside Git.
- There is deliberately no deletion API. Any future retention change requires
  explicit user confirmation, authorization, and audit behavior.

## Alternatives Considered

- Introduce `capicove.source`: rejected by architecture review because existing
  media capabilities belong under `capicove.processing`.
- Accept matching extensions and leading signatures without decoding: rejected
  because malformed containers can be misclassified as usable audio.
- Start with ASR: rejected because transcripts cannot be audited or rebuilt without
  stable original evidence and retention semantics.
- Commit real audio fixtures: rejected because tests can generate tiny synthetic WAV
  data, while supplementary real samples belong in ignored local datasets.
- Automatically select the default or language-matching stream: rejected because
  metadata can be missing, incorrect, or ambiguous and no product policy exists.
- Validate every audio stream: rejected for the MVP because a damaged unselected
  stream would block an explicitly valid selection. Selected-only scope is recorded
  instead.

## References

- Issue #12 (archived development review)
- [RFC-0004: Explicit audio stream selection and lineage](../rfcs/0004-explicit-audio-stream-selection.md)
- Issue #49 (archived development review)
- [Audio Tool Development and Acceptance Standard](../development/audio-tool-acceptance.md)
- [Module boundaries](../architecture/module-boundaries.md)
- [Testing strategy](../development/testing.md)
