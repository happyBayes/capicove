# RFC-0005: Sample-Accurate Local Audio Clipping

- Status: Accepted
- Date: 2026-08-17
- Owners: Capicove maintainers

## Summary

Capicove will expose one local, sample-accurate audio clipping operation across
Python, Tool, CLI, and MCP surfaces. It extracts one half-open time range from an
authorized local audio source and publishes a new, verified PCM16 WAV artifact
without modifying the source or overwriting an existing destination.

This RFC distinguishes sample rate, channel count, codec, container, bit depth, bit
rate, continuous time, and the discrete sample grid. It defines the contract and
trust boundary required by Issue #53. It does not authorize implementation or
release until the governance acceptance gate is complete.

## Scope and surfaces

Version 1 exposes:

- Python service `capicove.processing.audio.AudioClippingService`;
- Tool `capicove.audio.clip` contract version 1;
- CLI command `capicove audio clip`; and
- MCP name `capicove_audio_clip`.

One request selects exactly one range. Multiple ranges, concatenation, mixing,
fades, gain changes, compressed stream copy, overwrite, arbitrary FFmpeg arguments,
and compressed output codecs are excluded.

The canonical request is equivalent to:

```json
{
  "source": {"type": "path", "path": "/authorized/source.m4a"},
  "destination": {"type": "path", "path": "/authorized/clip.wav"},
  "start_seconds": 12.5,
  "end_seconds": 27.75,
  "profile": "pcm16-wav",
  "sample_rate_hz": 16000,
  "channels": 1
}
```

`profile`, `sample_rate_hz`, and `channels` are optional. Explicit JSON null is not
a substitute for omission. Booleans are rejected wherever a number or integer is
required.

## Input and audio-stream boundary

The declared input extensions are AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and WebM.
An extension is only an intake hint. FFprobe supplies the actual container family,
audio codec, sample rate, channel count, bit depth, duration, and observed
audio-stream bit rate. A missing audio-stream bit rate remains null; aggregate
container bit rate is never relabeled as audio bit rate.

Version 1 requires exactly one audio stream. Zero streams fail. Multiple streams
fail closed without selecting, mixing, or guessing from default disposition,
language, title, or container-global index. RFC-0004 is present in `dev` and Issue
#49 tracks its transcription implementation. Neither blocks clipping contract
version 1: clipping deliberately has no `audio_stream_index` property and does not
silently inherit the transcription selector. Adding explicit stream selection to
clipping later requires a separately reviewed compatibility decision and normally
a clipping Tool contract version increase.

Range validation uses the selected audio stream's duration. Container duration is
never used as an exact audio upper bound. Missing or untrustworthy selected-stream
duration fails before clipping rather than accepting an unverifiable range.

## Output profile and bit rate

Version 1 has one output profile:

- lowercase `.wav` container;
- `pcm_s16le`, signed 16-bit little-endian PCM;
- integer sample rate from 8,000 through 96,000 Hz; and
- one or two channels.

Omitting sample rate preserves the decoded input rate when it is inside the
supported range. Omitting channels preserves mono or stereo. Inputs with more than
two channels require an explicit mono or stereo conversion.

PCM bit rate is not caller-configurable. It is derived exactly as:

```text
sample_rate_hz * channels * 16 bits
```

For example, 48 kHz stereo PCM16 is 1,536,000 bits per second. Observed compressed
input bit rate remains diagnostic metadata and may be null.

## Time and sample-grid contract

The requested range is half-open: start is inclusive and end is exclusive. Let
`Rs` be the verified decoded source sample rate, `Ro` the resolved output sample
rate, and `Ts` and `Te` be `Decimal(str(start_seconds))` and
`Decimal(str(end_seconds))`. Boundary calculation does not use binary
floating-point multiplication or a language `round()` function:

```text
S = ceil(Ts * Rs)
E = ceil(Te * Rs)
N = E - S
P = N * Ro
(q, r) = divmod(P, Rs)       # 0 <= r < Rs
M_expected = q + (1 if 2 * r >= Rs else 0)
```

The selected source frames are `S, E)`. `M_expected` is the nearest integer to
`N * Ro / Rs`, with an exact half rounded upward toward the larger frame count.
Products, division, and comparison use non-negative integer arithmetic. Start must
be finite and non-negative. End must be finite, greater than start, and no later
than the selected audio stream's verified duration.

When `Ro == Rs`, the verified WAV must contain exactly `N` frames. When rates
differ, its strict RIFF parser derives `M_actual` from the data-chunk byte count and
requires `M_actual > 0` and `abs(M_actual - M_expected) <= 1`; any larger difference
is corrupted output. The quantized source range must contain at least 20
milliseconds when rates differ.

The canonical `actual_range` remains source-referenced. It returns
`start_frame = S`, `end_frame = E`, `start_seconds = S / Rs`, and
`end_seconds = E / Rs`. It never infers the returned source end from resampled
output. The separate output object returns `frame_count = M_actual` and
`duration_seconds = M_actual / Ro`; a caller that needs the output-derived terminal
position computes `S / Rs + M_actual / Ro`. These rational formulas are normative;
JSON numbers are their finite floating representations.

Both estimated and parser-verified PCM payloads are limited to 512 MiB.
Publication also requires enough free space to retain a 16 MiB reserve.

## Processing, evidence, and publication order

The operation order is fixed:

1. validate request types, profile, range, output extension, rate, and channels;
2. authorize source read and destination write paths;
3. preflight the validated operating-system, filesystem, descriptor-inheritance,
   FFprobe, and FFmpeg profile;
4. import and verify the complete source as immutable content-addressed evidence;
5. open the evidence blob without following symlinks and verify regular-file,
   link-count, size, and SHA-256 identity;
6. inspect exactly one audio stream through FFprobe using the same open source
   descriptor;
7. quantize the requested range and calculate bounded output geometry;
8. create a random same-directory output with `O_EXCL | O_NOFOLLOW`;
9. decode, clip, resample, and downmix through FFmpeg using fixed arguments and open
   source/output descriptors;
10. parse and verify the resulting RIFF/WAVE bytes, frame count, duration, geometry,
   and SHA-256;
11. publish atomically without replacing an existing path;
12. remove the private partial name and verify the final inode, link count, bytes,
    hash, source evidence, and destination-parent binding; and
13. return the canonical result.

Failures publish no successful result. Cleanup is identity-checked so the service
does not intentionally remove a path it no longer owns.

## Descriptor and subprocess contract

FFprobe and FFmpeg consume the same verified source file descriptor. FFmpeg writes
to the service-created output descriptor. Version 1 requires an FFmpeg build that
supports its seekable `fd:` protocol; `pipe:` and `/dev/fd` output are not accepted
because they cannot reliably rewrite the final WAV header.

Subprocesses use fixed argument arrays, `shell=False`, no caller-supplied flags, the
active request deadline, and bounded diagnostic capture. The capability performs no
network access, executable download, telemetry, model execution, or model upload.

## Validated runtime and filesystem profile

Audio clipping version 1 is available only on CPython 3.11 or newer on Linux or
macOS. Filesystem support is behavioral rather than inferred from a volume label.
The evidence store and destination parent must each provide POSIX regular-file,
device/inode, and link-count semantics plus `O_DIRECTORY`, enforced `O_NOFOLLOW`,
descriptor-relative `open`, `mkdir`, `stat`, `link`, and `unlink`, `pread` or
`lseek`, `fstatvfs`, and directory `fsync`. The subprocess runtime must support
inheriting only explicitly listed descriptors through `pass_fds`.

Windows and other operating systems are outside the version 1 profile. A network,
FUSE, cloud-synchronized, removable, or other managed filesystem is not rejected by
name, because the runtime cannot reliably identify every such deployment. Instead,
it is supported only if its observed behavior passes the admission sequence below
and only against this RFC's local threat model; Capicove makes no claim against a
same-UID synchronization agent or administrator that mutates names concurrently.
Operators requiring that stronger boundary must reject such roots in deployment
policy. Capicove never falls back to mutable pathnames, `/dev/fd`, `pipe:`,
copy-and-replace, rename-overwrite, or overwrite mode.

Capability discovery performs only host/backend readiness checks: Python and OS
primitive presence, descriptor inheritance in an implementation-owned temporary
directory, both executables, FFprobe `fd:` input, and FFmpeg `fd:` input and
seekable output. Its `available: true` does not certify an arbitrary future evidence
or destination root.

After path authorization and before evidence import or media processing, each
invocation admits the actual evidence-store root and destination parent. It opens
each directory root-to-leaf by descriptor without following symlinks and verifies
stable device/inode binding. In the destination parent it then creates a private
empty `0600` probe with `O_EXCL | O_NOFOLLOW`, hard-links it to a second private name
using source and destination directory descriptors, verifies both names bind the
same inode with link count two and that no existing name was replaced, fsyncs,
identity-removes both names, verifies absence, and fsyncs the directory again. The
probe is bounded, contains no media, and must leave no name behind. Equivalent
secure-store admission covers the evidence root. Missing primitives,
`ENOSYS`/`ENOTSUP`/`EOPNOTSUPP`/same-directory `EXDEV`, or a semantic mismatch maps
to `BACKEND_UNAVAILABLE`. Path-policy authorization denial remains
`permission_denied`; ordinary filesystem permission, capacity, I/O, or cleanup
failure maps to output/storage `internal`. Protocol-probe output is bounded and
never returned.

## RIFF/WAVE validation

Successful publication requires a strict, bounded RIFF parser. It accepts classic
PCM or WAVE_FORMAT_EXTENSIBLE with the PCM subtype and verifies:

- RIFF and WAVE identifiers and declared file size;
- chunk boundaries, padding, uniqueness, and complete payload;
- format tag and extensible PCM GUID;
- 16-bit sample width;
- sample rate, channels, block alignment, and byte rate;
- data size alignment to complete frames;
- expected non-zero frame count and duration; and
- complete-file SHA-256.

Trailing bytes, truncated chunks, malformed extension sizes, partial sample frames,
unsupported subtypes, zero-frame output, or geometry mismatches fail as corrupted
output.

## Canonical result and lineage

`AUDIO_CLIP_SPEC` is the sole owner of the Tool id, version, description, closed
input and output JSON Schemas, permissions, privacy, side effects, deadline,
redaction, and runtime resources. The registry key is exactly
`("capicove.audio.clip", 1)`. The specification declares modalities `("audio",)`,
permissions `audio:read`, `artifact:write`, and `subprocess:execute`, confidential
privacy, a write side effect, no network, deterministic execution for the same
backend version and inputs, `idempotent = false`, a 300-second maximum deadline,
redaction of `source.path` and `destination.path`, filesystem read/write, and
exactly the `ffmpeg` and `ffprobe` subprocess resources.

The closed input schema accepts only `source`, `destination`, `start_seconds`,
`end_seconds`, and optional `profile`, `sample_rate_hz`, and `channels`. Source and
destination are authorized local path references. Version 1 registers no remote
URI fetcher and no separate MCP Resource endpoint. The closed output schema owns
the requested range, source-referenced actual range, input/output metadata,
evidence, artifact, and lineage described below. Python, CLI, and MCP do not add or
rename transport-specific result fields.

Every successful result contains closed objects for:

- requested start and end seconds;
- actual sample-grid start and end seconds and frames;
- input and output duration, sample rate, channels, codec, container, observed bit
  rate, bit depth, audio ordinal, and container-global stream index;
- immutable source evidence id, SHA-256, byte size, media type, and source basename;
- output artifact id, SHA-256, byte size, media type, and destination basename; and
- backend, backend version, profile, configuration SHA-256, and source audio ordinal.

The Tool result never contains absolute source, evidence-store, partial, or
destination paths. Tool traces redact both request paths. CLI and MCP preserve the
same canonical Tool envelope and do not invent transport-specific result fields.

## Capability discovery, deadline, and cancellation

The existing `capicove audio capabilities` surface gains `--component clip` and
does not introduce another Tool contract. Omitting `--component` preserves the
existing transcription behavior. Selecting `clip` requires no model or VAD-model
arguments; transcription-only arguments neither become required nor influence the
clip result. Its closed clip report contains the
component id and version, overall availability, bounded FFmpeg/FFprobe versions and
their `fd:` protocol directions, the supported platform profile, secure filesystem
primitive checks, declared input formats and output geometry, exactly-one-stream
policy, no-overwrite policy, `network_access: false`, synchronous execution, a
300-second maximum timeout, and `cancellation: "not_supported"`. Executable and
private filesystem paths are never returned. `available: true` means host/backend
readiness only; the actual evidence store and destination remain subject to
per-invocation admission.

Unavailable reports include only these stable reason codes:
`unsupported_os`, `directory_fd_unavailable`, `nofollow_unavailable`,
`hard_link_unavailable`, `fd_inheritance_unavailable`, `ffprobe_unavailable`,
`ffprobe_fd_input_unavailable`, `ffmpeg_unavailable`,
`ffmpeg_fd_input_unavailable`, `ffmpeg_fd_output_unavailable`, and
`filesystem_profile_unavailable`. The Tool remains registered when unavailable.

Version 1 is synchronous. It creates no job id and registers no cancel-by-id or
active-invocation cancellation operation. `CANCELLED` is reserved and is not an
emitted version 1 outcome. Each invocation does carry one absolute timezone-aware
deadline. Remaining time is recomputed before authorization-dependent hashing,
probe, transform, validation, publication, and final verification, and every
subprocess receives the remaining bound. An already-expired deadline starts no
backend work. Expiry during FFprobe or FFmpeg terminates the child, performs bounded
reap then kill if necessary, and removes only identity-owned unacknowledged output.

## Stable failure semantics

The audio domain adds stable codes for invalid range, unsupported audio profile,
backend unavailability, output collision, output failure, and output corruption.
Existing invalid-audio, unsupported-format, decoder-unavailable, source-changed,
timeout, cancellation, and evidence-store enum values remain defined; as specified
above, `CANCELLED` is reserved and is not emitted by clipping version 1.

Invalid input, range, profile, source changes, and destination collisions map to a
failed `invalid_input` Tool result. Failure of any validated runtime prerequisite,
including directory-FD support, enforced no-follow behavior, same-directory hard
links, descriptor inheritance, an executable, or a required `fd:` protocol
direction, raises `BACKEND_UNAVAILABLE`. Its outer Tool result is failed
`unavailable` with `retryable: false`, no result or output evidence reference, and
the bounded path-free message `Secure audio clipping is unavailable in this
runtime.` Machine consumers use the capability reason codes, not this message.
Request-specific capacity, permission, I/O, validation, or cleanup failure after a
successful preflight remains an output or storage failure mapped to `internal`.
Path-policy rejection remains a separate rejected `permission_denied` result.

Version 1 has no replay key, result cache, receipt lookup, or rule that adopts an
existing destination. `OUTPUT_EXISTS` is always permanent `invalid_input`; an
existing file or symlink is never replaced or converted into success. Internally,
execution progresses through `PREPARE`, `PUBLISH_ATTEMPTED`, `PUBLISHED`, and
`ACKNOWLEDGED`. `PUBLISH_ATTEMPTED` begins immediately before the no-overwrite
hard-link syscall, `PUBLISHED` only after the final name is proven to bind the
candidate inode, and `ACKNOWLEDGED` only after final verification and construction
of the canonical success result.

Every version 1 failure has `retryable: false`, including a timeout in `PREPARE`
after successful cleanup, collision, unavailable backend, an ambiguous publication
syscall failure, final verification failure, lost acknowledgement, and cleanup
failure. The public retry bit does not ask a transport
to infer a hidden phase or filesystem state. Transports must never automatically
replay this non-idempotent Tool. After any failure, the caller first reconciles the
authorized destination: if absent, it may deliberately make a new logical
invocation with a new request id; if present, it preserves and independently
inspects it or chooses a new destination. Existence or byte equality alone is not a
replay receipt.

Messages are bounded and contain no raw subprocess diagnostics or private paths.
Failed and rejected results contain no successful artifact and no output evidence
reference.

## Security, privacy, and operational boundary

Source content, generated audio, model files, credentials, and `tests/datasets/`
content never enter Git. Integration tests create tiny synthetic media only in
temporary directories.

The destination parent is opened root-to-leaf with directory descriptors and
`O_NOFOLLOW`. Publication uses a same-filesystem hard link so an existing file or
symlink is never replaced. The final inode and hash are verified before and after
partial cleanup, and the source evidence is rehashed before success.

The local threat model covers accidental concurrency, untrusted media bytes, source
replacement, destination replacement, and parent-directory swaps. A malicious
process under the same operating-system user can still race a name between an inode
check and `unlink`; POSIX has no portable compare-and-unlink primitive. Random
unpredictable partial names reduce the window, but same-UID adversarial unlink races
remain deferred. A multi-tenant or hostile-local-process profile requires a
service-owned private staging directory or a stronger platform primitive.

## Compatibility and migration

This is a new contract and does not change existing evidence or transcription
schemas. It adds no Python runtime dependency and requires no persistent-data
migration. Removing the capability before release removes only its Tool/CLI/MCP
registration; immutable evidence and caller-owned outputs remain valid files.

Future compressed output, stream copy, multi-range editing, or explicit
multi-stream clipping requires a new contract version or accepted compatibility
decision. Version 1 clients must not infer those behaviors.

## Mandatory FFmpeg integration CI profile

The implementation PR must add an unconditional `audio-ffmpeg-fd` CI job on the
declared Linux profile. It uses Python 3.11 and a checksum- or digest-pinned
FFmpeg/FFprobe build, records both executable versions and build configuration, and
fails unless both executables, FFprobe `fd:` input, FFmpeg `fd:` input and seekable
output, and the fixture encoders for AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and WebM
are present.

The job sets `CAPICOVE_REQUIRE_AUDIO_FFMPEG=1` and runs the dedicated real-backend
module. Under that flag any missing executable, encoder, format, or protocol is a
failure, never a skip, and the module must finish with zero skips. Runtime-generated
tiny synthetic fixtures exercise all eight extensions, non-silent sample-grid
clipping, resampling, downmixing, 96 kHz extensible PCM, selected-stream duration
versus container duration, and descriptor-bound no-overwrite WAV publication. The
job has no path filter, `continue-on-error`, or optional runner condition.

## Validation

Acceptance requires automated coverage for:

- numeric and bool/type validation, finite ranges, half-open grid quantization, and
  output frame arithmetic;
- sample-rate, channel, output-extension, profile, payload, duration, and disk-space
  boundaries;
- zero, one, and multiple audio streams; audio ordinal versus container-global
  stream index; and selected-stream duration versus container duration;
- fixed FFprobe/FFmpeg arguments, `shell=False`, descriptor passing, deadline,
  missing executables, timeout, malformed JSON, and bounded diagnostics;
- classic PCM and 96 kHz WAVE_FORMAT_EXTENSIBLE PCM;
- malformed RIFF sizes, chunks, extension length, block alignment, byte rate,
  partial frames, truncation, trailing bytes, and zero-frame output;
- real source replacement, destination collision, destination symlink,
  parent-directory swap, post-publish replacement, cleanup failure, rollback, and
  source/output hash mismatch;
- the exact closed schemas, `ToolRegistry.resolve("capicove.audio.clip", 1)`,
  declared permissions/resources/privacy/side effects/deadline/redaction,
  representative success/failed/rejected envelope validation, and rejection of
  undeclared properties including `audio_stream_index`;
- MCP `tools/list` exposure as `capicove_audio_clip`, exact schema parity, local-path
  authorization, no remote URI or separate Resource endpoint, and CLI/MCP envelope
  parity;
- capability discovery for every available/unavailable platform reason, no private
  paths, destination-specific hard-link recheck, and registered-but-unavailable
  behavior;
- deadline expiry before every major stage and during both subprocesses, bounded
  termination/reap/kill and cleanup, `cancellation: "not_supported"`, and no
  version 1 `CANCELLED` outcome;
- `retryable: false` for every failure, including pre-publication and
  publication-attempted timeout, collision, ambiguous hard-link error,
  verification/cleanup failure, and lost acknowledgement, plus rejection of every
  automatic replay path;
- zero or multiple streams without invoking a fixed `0:a:0` fallback, even when
  default, language, or title metadata exists;
- every declared input extension through real FFmpeg with runtime-generated tiny
  synthetic fixtures; and
- the non-skipped mandatory `audio-ffmpeg-fd` CI job described above.

The base wheel must contain the clipping service, canonical Tool specification and
registration, CLI command, MCP mapper, and FFmpeg/FFprobe adapters without bundling
the executables or importing the optional MCP package during base-package import.
An installed-wheel test inspects those modules, installs into a clean environment,
imports the Python service and Tool contract, runs `capicove audio clip --help`,
and executes
`capicove audio capabilities --component clip --ffmpeg /__capicove_missing__/ffmpeg
--ffprobe /__capicove_missing__/ffprobe --output json`. That command must return
exit code 1 and a closed unavailable JSON report with no traceback, download, or
network access. After installing the `mcp` extra, the test verifies
`capicove_audio_clip` and the exact canonical schemas. Tests and local media must
not be present in the wheel.

The implementation PR must pass repository quality, full tests with coverage,
strict documentation, build, security diff scan, secret scan, staged-path,
staged-object-size, and CI gates. No real audio or local dataset is accepted as Git
evidence.

## Alternatives considered

- **Default to `-c:a copy`.** Rejected because compressed packet boundaries cannot
  honor arbitrary sample-accurate cuts.
- **Expose every FFmpeg codec and flag.** Rejected because it creates an unsafe,
  deployment-dependent shell-shaped contract.
- **Overwrite an existing destination.** Rejected because failed validation could
  destroy caller-owned data.
- **Infer codec or bit rate from the filename.** Rejected because containers can
  carry different codecs and variable-rate streams may not expose one fixed rate.
- **Preserve arbitrary channel layouts.** Rejected in version 1 because downmix and
  result-geometry policy would be ambiguous.
- **Use container duration when stream duration is missing.** Rejected because a
  longer video or sibling stream could authorize a range beyond the selected audio.

## Governance gate

This RFC adds a canonical Tool contract, CLI/MCP surface, and media-processing
trust boundary. Under the current `GOVERNANCE.md` and development workflow, it
requires an accepted RFC and one approving review from a non-author maintainer on
the final head. That single approval satisfies both the RFC risk-level requirement
and the `dev` workflow's non-author review requirement; this RFC imposes no
supplemental two-approval gate. RFC-0005 and ADR-0011 must then be merged into
`dev` before the runtime implementation is submitted.

## References

- [Issue #52: RFC discussion (archived development review)
- Issue #53: implementation (archived development review)
- [ADR-0011: Sample-accurate local audio clipping](../adr/0011-sample-accurate-local-audio-clipping.md)
- [RFC-0004: Explicit audio stream selection](0004-explicit-audio-stream-selection.md)
- [ADR-0009: Immutable audio evidence intake](../adr/0009-immutable-audio-evidence-intake.md)
- [Audio Tool Development and Acceptance Standard](../development/audio-tool-acceptance.md)
