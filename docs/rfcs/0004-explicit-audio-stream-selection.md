# RFC-0004: Explicit Audio Stream Selection and Lineage

- Status: Accepted
- Date: 2026-08-16
- Owners: Capicove maintainers

## Summary

Capicove will select one audio stream explicitly and consistently across Python,
Tool, CLI, and MCP transcription surfaces. The optional `audio_stream_index` is the
zero-based audio-stream ordinal used by FFmpeg `0:a:N`; it is not the
container-global stream index reported by FFprobe.

Single-stream input may omit the selector and implicitly uses ordinal zero.
Multi-stream input without a selector fails closed. Capicove does not automatically
mix streams, select a default disposition, guess by language, or fall back after a
selected-stream failure.

This RFC defines the accepted contract required by Issue #49 and coordinates the
decisions in ADR-0009 and ADR-0010. Implementation is authorized; public release
remains a separate gate.

## Evidence and problem

The current audio validator and Whisper normalization both use `0:a:0`. Synthetic
compatibility probes showed that later audio streams are silently ignored even when
a later stream is marked default. A silent first stream can be selected instead of
an audible second stream, while damage in the first stream can block an otherwise
usable later stream.

Existing mono, stereo, and 5.1 tests cover channel layouts within one audio stream.
They do not establish compatibility with containers that have multiple independent
audio streams.

## Selection contract

`audio_stream_index` is optional, but an explicitly supplied null value is invalid.
Valid values are integers greater than or equal to zero. Python booleans, JSON
booleans, floats, strings, and negative values are rejected before a probe, evidence
write, or model invocation.

The Python service uses an implementation-owned omission sentinel as the keyword
default. Normal callers omit the keyword; the sentinel is not public input and is
never serialized. Explicit Python `None` is therefore distinguishable from omission
and is rejected. Tool and MCP requests represent omission only by an absent property
and reject JSON `null`. CLI omission means the flag is absent; the CLI does not pass
`None` or serialize a null property.

The selection rules are:

- zero streams fail with `NO_AUDIO_STREAM`;
- one stream with the selector omitted selects ordinal zero with
  `sole_stream_implicit`;
- a valid explicit selector uses `explicit_index`;
- multiple streams with the selector omitted fail with
  `AUDIO_STREAM_SELECTION_REQUIRED`;
- an invalid selector fails with `INVALID_AUDIO_STREAM_INDEX`; and
- a non-negative selector outside the observed range fails with
  `AUDIO_STREAM_INDEX_OUT_OF_RANGE`.

FFprobe is used only to count audio streams. Array order becomes the audio-stream
ordinal; container-global `stream.index`, title, language, and default-disposition
metadata do not select a stream. Probe output is strictly parsed, limited to 64 KiB,
and limited to at most 64 audio streams.

## Processing and evidence order

The operation order is fixed:

1. validate the selector type and non-negative range;
2. verify the supported extension and leading content signature;
3. inspect the audio-stream count;
4. resolve one selection using the rules above;
5. fully decode only that stream through FFmpeg `0:a:N`;
6. hash and preserve the complete original container as immutable evidence;
7. normalize and transcribe the same selected ordinal; and
8. re-verify evidence and return versioned selection lineage.

All selection, inventory, and selected-stream decode failures occur before the
evidence store is changed and before ASR or model hashing begins. The same immutable
selection value is passed to validation and normalization so the two stages cannot
select different streams.

Only the selected stream is decoded for admission. A damaged unselected stream does
not block a valid selected stream when the container inventory remains readable. A
damaged selected stream fails with `SELECTED_AUDIO_STREAM_UNDECODABLE`; Capicove does
not fall back. Results must state `selected_audio_stream_only` and never claim that
every stream in the preserved container was validated.

The evidence SHA-256, identity, and blob path continue to describe the complete
original container. Selecting a different stream does not create a different source
evidence identity.

## Result and lineage schema

Every successful transcription records a closed selection object at
`lineage.audio_stream_selection`:

```json
{
  "schema_version": 1,
  "selected_audio_stream_index": 1,
  "index_semantics": "zero_based_audio_stream_ordinal",
  "selection_method": "explicit_index",
  "preflight_audio_stream_count": 2,
  "validation_scope": "selected_audio_stream_only"
}
```

The Python result owns the same immutable value. Model and VAD identity remain in
backend lineage; source-selection policy is not embedded in model identity.

Successful explicit selection from a multi-stream container adds
`UNSELECTED_AUDIO_STREAMS_IGNORED` exactly once. Single-stream requests do not add
that warning.

Tool version 2 exposes a closed domain outcome containing the exact audio failure
code and bounded message. The outer `ToolErrorCode` stays transport-neutral:
`AUDIO_STREAM_SELECTION_REQUIRED` is a policy rejection, while invalid, no-stream,
out-of-range, and undecodable selections are failed invalid input. Permission and
path-policy rejections remain separate and contain no audio-domain result. All
selection rejections and failures return no evidence references.

The Tool version 2 output schema is a `oneOf` over three closed result objects. The
success branch requires `status: "succeeded"` plus the existing transcript,
warnings, evidence, and backend lineage fields and the required stream-selection
lineage above. The structured ambiguity-rejection branch is exactly:

```json
{
  "status": "rejected",
  "failure": {
    "code": "AUDIO_STREAM_SELECTION_REQUIRED",
    "message": "Multiple audio streams require audio_stream_index.",
    "retryable": false,
    "preflight_audio_stream_count": 2
  }
}
```

For this branch, `status` is the discriminator and must equal `rejected`, `code` is
the constant `AUDIO_STREAM_SELECTION_REQUIRED`, the outer `ToolResult` uses
`ToolStatus.REJECTED`, and its outer error code is `policy_rejected`. The failed
domain branch has the same shape but requires `status: "failed"`, excludes
`AUDIO_STREAM_SELECTION_REQUIRED` from its code enum, and uses
`ToolStatus.FAILED`.

For both domain branches, `additionalProperties` is false at both object levels;
`code`, `message`, and `retryable` are required. `message` is non-empty, at most 256
characters, and never contains source paths or subprocess diagnostics.
`preflight_audio_stream_count` is optional, is an integer from zero through 64, and
appears only when inventory completed. The allowed domain codes are the existing
`AudioFailureCode` values plus the five selection codes defined by this RFC. The
inner and outer retryability decisions are identical, and `evidence_refs` is empty.
Permission and path-policy rejection instead has `ToolStatus.REJECTED`, no domain
result, and no evidence references.

The exhaustive outer mapping is:

| Audio-domain condition or code | Outer status | Outer `ToolErrorCode` | Retryable |
| --- | --- | --- | --- |
| `AUDIO_STREAM_SELECTION_REQUIRED` | `rejected` | `policy_rejected` | false |
| `INVALID_AUDIO`, `UNSUPPORTED_FORMAT`, `UNSUPPORTED_LANGUAGE`, `SOURCE_CHANGED` | `failed` | `invalid_input` | false |
| `INVALID_AUDIO_STREAM_INDEX`, `NO_AUDIO_STREAM`, `AUDIO_STREAM_INDEX_OUT_OF_RANGE`, `SELECTED_AUDIO_STREAM_UNDECODABLE` | `failed` | `invalid_input` | false |
| `DECODER_UNAVAILABLE`, `MODEL_UNAVAILABLE`, `MODEL_LOAD_FAILED` | `failed` | `unavailable` | false |
| `TIMEOUT` | `failed` | `timeout` | true |
| `CANCELLED`, `STORE_CORRUPTED`, `STORE_FAILED` | `failed` | `internal` | false |

A missing FFprobe executable maps to `DECODER_UNAVAILABLE`; probe timeout maps to
`TIMEOUT`; probe failure, malformed JSON, output beyond 64 KiB, or inventory beyond
64 audio streams maps to `INVALID_AUDIO`. These failures occur before evidence or
model work. No raw probe output is returned.

## Runtime surfaces and resources

- `AudioProcessor` performs inventory and selected-stream validation before storage
  and can return evidence together with selection lineage.
- `AudioTranscriptionService.transcribe_file` uses the omission sentinel described
  above and passes the resolved ordinal to the ASR port.
- `capicove.audio.transcribe` advances to Tool contract version 2.
- CLI adds `--audio-stream-index` and an explicit `--ffprobe` executable setting.
  Omitting the selector does not serialize explicit null.
- MCP exposes the exact Tool version 2 input and output schemas.
- Tool resources and capability discovery declare `ffprobe`, `ffmpeg`, the explicit
  selector, and the ambiguity policy.

Supported formats remain AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and WebM. This RFC
does not add MP4, MKV, or another media-container boundary.

## Compatibility and migration

Single-stream Python and CLI calls may continue to omit the selector. Their Tool and
MCP results advance to version 2 and contain required selection lineage.

Multi-stream calls that previously succeeded by silently selecting stream zero now
fail until the caller supplies an explicit ordinal. This is an intentional safety
correction. Tool clients must request or accept version 2 and its closed
success/failure result union. Version 1 is not retained as an unsafe multi-stream
alias.

Evidence stores need no migration because original bytes, hashes, and paths do not
change. Rollback may restore the version 1 first-stream-only surface, but that state
must retain a blocked multi-stream release gate and cannot be described as
multi-stream compatible.

## Security, privacy, and operational boundaries

FFprobe and FFmpeg use fixed argument arrays, no shell, the active request deadline,
bounded probe output, and bounded stream count. Caller-provided flags, raw FFprobe
metadata, decoder diagnostics, track titles, and private source paths are not
returned or persisted.

All processing remains local, offline, CPU-only, and explicitly authorized.
Original-container hashing, source immutability checks, content-addressed storage,
path authorization, model hashing, trace redaction, and no-network behavior remain
unchanged.

No automatic mixing, default- or language-based selection, fallback, model
download, telemetry, source deletion, or format expansion is introduced.

## Validation

Acceptance requires automated coverage for:

- zero, one, two, and over-limit audio-stream inventories;
- bool, float, string, explicit-null, negative, and out-of-range selectors;
- audio ordinal versus container-global stream index;
- fixed FFprobe and FFmpeg argument arrays, `shell=False`, active-deadline
  propagation, missing executables, timeout, malformed JSON, and oversized output;
- two valid streams and a default stream not at ordinal zero;
- silent and audible selections;
- damaged selected and unselected streams;
- multilingual and commentary-tagged streams without metadata-based guessing;
- no evidence-store change and no ASR/model invocation for every failed selection;
- one shared evidence identity with distinct selection lineage for two ordinals;
- the ignored-stream warning and selected-only validation scope;
- Python, Tool, CLI, and MCP success and domain-failure parity; and
- mono, stereo, and 5.1 single-stream regressions, kept distinct from multi-stream
  behavior.

Real FFprobe/FFmpeg integration fixtures generate tiny multi-stream M4A and WebM
containers at runtime in temporary directories. Damaged-stream cases use controlled,
argument-sensitive subprocess doubles. No real audio, model, or `tests/datasets/`
content enters Git, and real ASR accuracy is not claimed.

The implementation PR must pass repository quality, test, build, documentation,
secret, object-size, and CI gates. Audio release readiness remains blocked until the
implementation matrix passes and receives the required review.

## Alternatives considered

- **Keep first-stream-only behavior.** Rejected because it silently omits content.
- **Choose by default disposition or language.** Rejected because metadata can be
  missing, incorrect, or ambiguous and no product policy has been approved.
- **Mix streams.** Rejected because it changes meaning and harms ASR quality and
  lineage.
- **Validate every stream.** Rejected for this MVP because an unselected damaged
  stream would block an explicitly valid selection.
- **Use container-global stream indexes.** Rejected because the public contract is
  the audio ordinal used by `0:a:N`.
- **Add MP4/MKV support now.** Rejected as unrelated format and trust-boundary scope.

## Governance gate

This RFC changes a canonical Tool contract, serialized lineage, CLI/MCP behavior,
and a media-validation trust boundary. Under `GOVERNANCE.md`, it requires two
approving reviews, including at least one maintainer, before its status becomes
Accepted or implementation begins.

## References

- Issue #50: RFC discussion (archived development review)
- Issue #49: Multi-stream implementation (archived development review)
- PR #43 reviewer decision (archived development review)
- [ADR-0009: Immutable audio evidence intake](../adr/0009-immutable-audio-evidence-intake.md)
- [ADR-0010: Local offline audio transcription](../adr/0010-local-offline-audio-transcription.md)
