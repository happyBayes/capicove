# ADR-0010: Provide Local Offline Audio Transcription

- Status: Accepted
- Date: 2026-08-12
- Owners: Capicove maintainers

## Context

The weekly delivery plan requires one complete audio vertical slice after immutable
evidence intake: an authorized local WAV or MP3 must become an ordered timestamped
transcript through Python, CLI, and MCP-compatible boundaries. The broader audio
acceptance standard also requires explicit capabilities, stable failures, source
integrity, model lineage, Mandarin, English, declared code-switching behavior, and
no hallucinated text for silence or pure noise.

Real local smoke tests exposed two material backend behaviors. Running Whisper on a
whole file can produce hallucinated text during long silence or noise, and one
automatic language decision for a complete recording can omit the second language
in Mandarin-English code-switching. These behaviors make a direct one-command
Whisper wrapper insufficient for the declared boundary.

## Decision

Add the versioned Tool `capicove.audio.transcribe` under the existing
`capicove.processing.audio` capability. Tool version 2 accepts only an explicitly
authorized local path, an optional supported language, and an optional
`audio_stream_index`. The index is the zero-based audio-stream ordinal used by
FFmpeg `0:a:N`, not a container-global stream index.

Stream inventory and selection complete before evidence storage or model
invocation. Single-stream input may omit the index; Capicove then selects ordinal
zero implicitly. Multi-stream input without an index returns a structured
selection-required policy rejection. The Tool never guesses by default disposition
or language, mixes streams, or falls back to another stream.

The Tool preserves and verifies the complete original container through the
content-addressed evidence intake before deriving a transcript from the selected
stream. The Tool has no network access and never downloads a model.

The private adapter uses explicit local paths for a multilingual whisper.cpp model
and a Silero VAD model. It requires FFmpeg, `whisper-cli`, and
`whisper-vad-speech-segments`. Automatic-language requests first detect speech
segments, transcribe each segment independently, and map the segment-relative
timestamps back to the original normalized audio timeline. Explicit-language
requests use the selected language with VAD enabled. Temporary normalized and
segmented audio is removed when the request ends.

The result contains complete text, ordered millisecond-precision segments,
structured warnings, immutable evidence identity, ASR/VAD model hashes, and a
required versioned `lineage.audio_stream_selection` object. That object records the
selected ordinal, index semantics, explicit or sole-stream selection method,
preflight stream count, and selected-only validation scope. Explicit selection from
a multi-stream container adds `UNSELECTED_AUDIO_STREAMS_IGNORED` exactly once.

Stable selection failures retain their audio-domain code in a closed Tool version 2
failure result while the outer Tool error remains transport-neutral. Selection
failures return no evidence references. The internal `ToolResult` also records input
duration, segment count, total duration, and end-to-end real-time factor without
logging raw audio or transcript text.

Capabilities report the actual executable and model availability. Unsupported
audio operations remain explicitly listed rather than being simulated.

The initial deployment profile is offline CPU file transcription. It declares no
streaming, GPU, remote inference, automatic model acquisition, TTS, standalone VAD,
diarization, denoising, asynchronous job, cancellation, or retrieval support.

## Consequences

- Mandarin, English, and Mandarin-English code-switching can use one local Tool
  without sending source content to a service.
- Silence and pure-noise inputs return an empty transcript with
  `NO_SPEECH_DETECTED` instead of fabricated text in the validated profile.
- Every transcript identifies both the ASR and VAD model artifacts and retains an
  immutable evidence reference plus exact audio-stream selection lineage.
- Containers with multiple audio streams cannot silently omit later streams. A
  caller must select one explicit ordinal, and ignored streams are disclosed.
- Existing single-stream calls may omit the selector, while Tool and MCP consumers
  migrate from contract version 1 to version 2 because the closed result and lineage
  schemas change.
- Automatic language detection incurs one local ASR invocation per detected speech
  region. This favors correctness and mixed-language coverage over minimum latency.
- Model weights, real audio, derived evidence stores, and local smoke outputs stay
  outside Git.
- A shared `SearchableUnit` mapping remains blocked on the repository-wide shared
  interface freeze; this ADR does not invent a private cross-module replacement.
- This ADR does not modify or supersede the immutable intake decision in ADR-0009.

## Alternatives considered

- Transcribe the complete recording once with `language=auto`: rejected because the
  backend can lock to the first language and omit a later English or Mandarin span.
- Run ASR without VAD and filter suspicious text afterward: rejected because text
  heuristics cannot reliably distinguish real repetitive speech from hallucination.
- Run cloud ASR as a fallback: rejected because it expands the trust boundary and
  conflicts with the offline default.
- Merge separate forced-Mandarin and forced-English transcripts: rejected because it
  duplicates content and cannot provide an honest unified timeline.
- Select a default- or language-tagged stream automatically: rejected because stream
  metadata is not a reviewed product policy and may be absent or incorrect.
- Mix or fall back across streams: rejected because it changes source meaning and
  makes transcript lineage ambiguous.

## References

- [Audio Tool Development and Acceptance Standard](../development/audio-tool-acceptance.md)
- [Audio transcription usage and boundary](../development/audio-transcription.md)
- [RFC-0004: Explicit audio stream selection and lineage](../rfcs/0004-explicit-audio-stream-selection.md)
- Issue #49 (archived development review)
- [ADR-0009: Validate and Preserve Immutable Audio Evidence](0009-immutable-audio-evidence-intake.md)
- Two-person weekly delivery plan (archived development reference)
