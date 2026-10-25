# ADR-0011: Provide Sample-Accurate Local Audio Clipping

- Status: Accepted
- Date: 2026-08-17
- Owners: Capicove maintainers

## Context

Capicove can preserve local audio as immutable evidence and derive an offline
transcript, but it has no public operation for extracting a bounded audio range.
A clipping contract must distinguish container, codec, sample rate, channel count,
bit depth, and bit rate. Treating those properties as interchangeable would make
duration and quality claims misleading. Compressed stream copying also cannot
promise arbitrary sample-accurate boundaries because AAC, MP3, Opus, and similar
codecs operate on packet or frame boundaries.

The operation writes a derived artifact, so it must preserve the original evidence,
reject ambiguous media, avoid silent overwrite, and return metadata derived from the
bytes that were actually published.

## Decision

The 2026-10-07 implementation integrates the existing clipping contract into the
version-2 workspace host through audio configuration version 2, adding a private
`output_root`. Its host path retains ADR-0026's Linux-only staging and executable
pinning, checks the invocation deadline while staging, and uses RFC-0005's fixed
`fd:` media commands. Standalone processing retains the declared Linux/macOS
profile. Version-1 audio host configurations continue to serve import/transcription.

Add the versioned Tool `capicove.audio.clip`, the CLI command
`capicove audio clip`, and the MCP name `capicove_audio_clip`. Version 1 extracts
exactly one half-open range `[start_seconds, end_seconds)` from an authorized local
file. It does not concatenate ranges, edit volume, add fades, or expose arbitrary
FFmpeg arguments.

The declared input extensions remain AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and
WebM. An extension is only an intake hint. FFprobe supplies the selected audio
stream's actual codec, container family, sample rate, channel count, bit depth, and
observed stream bit rate. A missing stream bit rate remains `null`; container bit
rate is never relabeled as audio bit rate. Media with zero or multiple audio streams
is rejected in version 1. RFC-0004 is present in `dev`, but clipping version 1 has
no stream-selector property. Issue #49 may proceed independently and does not block
clipping; adding a selector later requires a separately reviewed compatibility
decision and normally a Tool contract version increase.

Version 1 has one output profile: lowercase `.wav` containing PCM signed 16-bit
little-endian audio. The caller may select an integer sample rate from 8,000 through
96,000 Hz and either one or two channels. Omitting sample rate preserves the decoded
input rate when it is within that range. Omitting channels preserves mono or stereo;
inputs with more channels require an explicit mono or stereo conversion. PCM bit
rate is derived exactly as `sample_rate_hz * channels * 16 bits` and is not a free
parameter.

The requested continuous-time interval is mapped to the decoded source grid with
`S = ceil(Decimal(str(start)) * Rs)` and
`E = ceil(Decimal(str(end)) * Rs)`. With `N = E - S`, resampling computes
`(q, r) = divmod(N * Ro, Rs)` and
`M_expected = q + (1 if 2 * r >= Rs else 0)`: integer round-to-nearest with exact
halves rounded toward the larger frame count. Same-rate output must contain exactly
`N` frames; resampled output must contain a positive parser-derived `M_actual`
within one frame of `M_expected`. The returned `actual_range.end_seconds` is the
source-grid boundary `E / Rs`, while output duration is separately
`M_actual / Ro`; neither is relabeled as the other. A resampled range must contain
at least 20 milliseconds. Estimated and actual PCM payloads are limited to 512 MiB,
and publication retains a 16 MiB free-space reserve.

The service first imports and verifies immutable evidence. FFprobe and FFmpeg consume
the same open source descriptor. FFmpeg uses its seekable `fd:` protocol and writes
only to a service-created `O_EXCL | O_NOFOLLOW` descriptor. The service strictly
parses classic PCM or WAVE_FORMAT_EXTENSIBLE PCM RIFF bytes and verifies sizes,
chunks, format geometry, frame alignment, sample rate, channels, bit depth, frame
count, byte rate, duration, and SHA-256.

The destination parent is opened root-to-leaf with directory descriptors and
`O_NOFOLLOW`. Publication uses a same-filesystem hard link and never overwrites an
existing file or symlink. The final inode and hash are checked before and after
partial cleanup. Original evidence is rehashed before success. Source and
destination paths are redacted from persisted Tool arguments and canonical results.

The Capicove distribution remains portable, but clipping is conditionally
available only on CPython 3.11 or newer on Linux or macOS with descriptor-relative
no-follow operations, stable device/inode/link-count semantics, same-directory hard
links, directory `fsync`, explicit descriptor inheritance, and FFprobe/FFmpeg `fd:`
protocol directions. Capability discovery establishes host/backend readiness only.
Each invocation behaviorally admits its actual evidence and destination roots,
including a bounded create-link-verify-unlink-fsync probe with no media content;
filesystem names are not treated as proof. Managed filesystems remain subject to
deployment policy and the same-UID threat-model exclusion. The implementation never
falls back to mutable pathnames, pipes, copy-and-replace, or overwrite. Python raises
`BACKEND_UNAVAILABLE` for missing secure prerequisites; Tool, CLI, and MCP map that
to non-retryable `unavailable`.

`AUDIO_CLIP_SPEC` exclusively owns the closed schemas and registry key
`("capicove.audio.clip", 1)`, confidential write classification, required
permissions, path redaction, 300-second deadline, and filesystem plus exactly
`ffmpeg`/`ffprobe` resources. It is deterministic for fixed bytes, configuration,
and backend version, but non-idempotent. There is no replay key or existing-output
adoption. Every failure has `retryable: false`, including timeouts before or after
the no-overwrite hard-link attempt, ambiguous publication, lost acknowledgement,
collision, verification, and cleanup. A caller may create a new logical invocation
with a new request id only after reconciling the authorized destination; transports
never replay automatically.

Version 1 is synchronous and supports an absolute deadline, but no active
cancellation or job id; capability discovery reports
`cancellation: "not_supported"`, and `CANCELLED` is reserved rather than emitted.
The implementation adds one mandatory, non-skipped Linux CI job with a pinned
FFmpeg/FFprobe build. It proves both `fd:` input directions, seekable FFmpeg `fd:`
output, all eight declared formats, and descriptor-bound WAV publication. Direct
acceptance also covers canonical schema and Tool registration, capability reporting,
deadline cleanup, CLI/MCP parity, and clean-wheel exposure.

The local threat model covers accidental concurrency, path and parent replacement,
and untrusted media bytes. A malicious process running under the same operating-
system user can still race a name between an inode check and `unlink`; POSIX has no
portable compare-and-unlink primitive. Random partial names reduce that window, but
same-UID adversarial unlink races remain deferred and must be resolved before a
multi-tenant or hostile-local-process claim.

## Consequences

- Callers get a small, auditable clipping contract instead of arbitrary transcoding.
- Successful results report requested and quantized ranges, input and output media
  properties, source and output hashes, frame count, backend version, and a
  configuration hash.
- PCM output is larger than compressed output, but it has deterministic geometry and
  does not blur codec-frame boundaries with sample-accuracy claims.
- FFmpeg and FFprobe remain external executables. Descriptor-bound clipping requires
  an FFmpeg build with the seekable `fd:` protocol.
- Windows can install Capicove but cannot advertise or execute clipping version 1.
  Other filesystem roots require behavioral admission per invocation. Deployment
  changes, not request retries, resolve an unavailable runtime; no insecure
  compatibility fallback is permitted.
- Resampling uses integer round-to-nearest with exact halves rounded upward. Source
  grid end and output duration are separate result facts.
- The Tool is non-idempotent and every failure is non-retryable. Callers reconcile
  the destination before deliberately creating a new logical invocation; transports
  never replay automatically.
- The required real-FFmpeg CI lane adds installation and execution cost but prevents
  protocol, encoder, and fixture skips from masquerading as support.
- Stream copying, compressed output codecs, caller-selected bit rate, overwrite,
  multiple audio streams, arbitrary channel layouts, and sub-20-ms resampling are
  not part of version 1.
- Generated integration fixtures stay in temporary directories. No source or
  derived audio is added to Git.
- This ADR requires RFC-0005 to be Accepted, approved by one non-author
  maintainer on the final head, and merged into `dev` before implementation is
  submitted. It adds no supplemental two-approval gate.

## Alternatives considered

- Default to `-c:a copy`: rejected because compressed packet boundaries cannot honor
  arbitrary sample-accurate cuts.
- Offer every FFmpeg codec and flag: rejected because it creates an unsafe,
  deployment-dependent shell-shaped contract.
- Overwrite an existing destination: rejected because failed validation could
  destroy caller-owned data.
- Infer codec and bit rate from the filename: rejected because containers can carry
  different codecs and variable-rate streams may not expose one fixed bit rate.
- Preserve every input channel layout: rejected in version 1 because output geometry
  and downmix policy would be ambiguous.

## References

- [RFC-0005: Sample-accurate local audio clipping](../rfcs/0005-sample-accurate-local-audio-clipping.md)
- Issue #52: RFC discussion (archived development review)
- Issue #53: implementation (archived development review)
- [RFC-0004: Explicit audio stream selection](../rfcs/0004-explicit-audio-stream-selection.md)
- [ADR-0009: Immutable audio evidence intake](0009-immutable-audio-evidence-intake.md)
- [ADR-0010: Local offline audio transcription](0010-local-offline-audio-transcription.md)
- [Audio Tool Development and Acceptance Standard](../development/audio-tool-acceptance.md)
