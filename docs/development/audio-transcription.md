# Local Audio Transcription

This document describes the first Capicove audio vertical slice. It implements
offline file transcription through the canonical `capicove.audio.transcribe` Tool,
the CLI, and an MCP mapper. It follows the applicable ASR, evidence, capability,
privacy, and smoke-test requirements in
[Audio Tool Development and Acceptance Standard](audio-tool-acceptance.md).

## Implemented boundary

- authorized local file paths only;
- explicit zero-based audio-stream selection for multi-stream containers;
- immutable, content-addressed preservation of the original audio;
- local FFmpeg normalization to 16 kHz mono PCM for inference only;
- local CPU whisper.cpp ASR with explicit model artifacts;
- local Silero VAD before ASR;
- independent speech-region language detection for Mandarin-English code-switching;
- complete text, ordered millisecond-precision segments, warnings, evidence hashes,
  and ASR/VAD lineage;
- Python Tool handler, CLI JSON/human output, and MCP-compatible result parity;
- no network access, automatic download, telemetry, or source deletion.

The caller keeps the original source. Capicove also preserves an immutable evidence
copy under the configured evidence store. Temporary normalized and segmented files
are deleted automatically at the end of the request.

The following operations are not implemented by this vertical slice and are
reported as unavailable: TTS, standalone VAD, diarization, denoising, asynchronous
jobs, job cancellation, streaming, GPU execution, and retrieval indexing. Mapping
transcript segments to the planned shared `SearchableUnit` remains dependent on the
repository-wide shared interface freeze.

## Local prerequisites

Install FFprobe, FFmpeg, and whisper.cpp outside the Python package. Obtain a compatible
multilingual whisper.cpp model and Silero VAD model from their official sources.
Capicove does not download or redistribute model weights.

Verify the actual deployment before transcription:

```bash
capicove audio capabilities \
  --model /absolute/path/to/ggml-base.bin \
  --vad-model /absolute/path/to/ggml-silero-v6.2.0.bin \
  --whisper-cli /absolute/path/to/whisper-cli \
  --whisper-vad-cli /absolute/path/to/whisper-vad-speech-segments \
  --ffprobe /absolute/path/to/ffprobe \
  --ffmpeg /absolute/path/to/ffmpeg \
  --output json
```

`available` is true only when all four executables and both model files are
present. The response includes model hashes, declared formats and languages, the
offline CPU boundary, the explicit stream-selection policy, and every unimplemented
operation.

## Transcribe a local file

```bash
capicove audio transcribe \
  --input /absolute/path/to/meeting.wav \
  --model /absolute/path/to/ggml-base.bin \
  --vad-model /absolute/path/to/ggml-silero-v6.2.0.bin \
  --store /absolute/path/to/private-evidence-store \
  --language auto \
  --whisper-cli /absolute/path/to/whisper-cli \
  --whisper-vad-cli /absolute/path/to/whisper-vad-speech-segments \
  --ffprobe /absolute/path/to/ffprobe \
  --ffmpeg /absolute/path/to/ffmpeg \
  --timeout 300 \
  --output json
```

Supported language arguments are `auto`, `zh`, `en`, `zh-CN`, `en-US`, and
`en-GB`. Locale variants are mapped to the backend language while the result reports
the detected transcript language. Automatic mode returns `multilingual` only when
more than one language has at least one second of recognized speech. Silence and
pure noise return empty text and `NO_SPEECH_DETECTED`.

For a container with multiple independent audio streams, pass the zero-based audio
ordinal used by FFmpeg `0:a:N`:

```bash
capicove audio transcribe \
  --input /absolute/path/to/multistream.m4a \
  --audio-stream-index 1 \
  --model /absolute/path/to/ggml-base.bin \
  --vad-model /absolute/path/to/ggml-silero-v6.2.0.bin \
  --output json
```

This index is not the container-global stream index. A single audio stream may omit
the flag and selects ordinal zero implicitly. Multiple streams without the flag are
rejected; Capicove never guesses from default disposition, language, or title,
never mixes streams, and never falls back after a selected-stream failure. A
successful explicit multi-stream request reports
`UNSELECTED_AUDIO_STREAMS_IGNORED` once.

The JSON result does not expose the server-side evidence storage path. It returns
the evidence SHA-256, source file name, byte size, media type, timestamped segments,
warnings, ASR/VAD model lineage, and required
`lineage.audio_stream_selection` fields. Full paths are redacted from trace arguments.
The evidence and both model artifacts are hash-verified after inference; a concurrent
change fails instead of returning a transcript with misleading lineage.

## Developer smoke test

Real or generated audio, evidence stores, model weights, and per-sample result files
must stay in an ignored local directory such as `/private/tmp`. Do not commit them.
At minimum, exercise clean Mandarin, clean English, code-switching, numbers and
dates, speaker turns, office noise, music interference, reverberation, silence,
pure noise, WAV, and MP3. Record source hashes before and after each run and verify
the evidence blob has the same hash.

The local smoke evidence for this implementation is recorded in
audio-transcription-smoke-report-2026-08-12.md (internal record; public support status is documented in the capability guide).
It is development evidence, not a release benchmark or a substitute for the frozen,
license-cleared business evaluation set required before release.
