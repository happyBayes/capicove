# Transcribe audio

Transcription requires local FFprobe, FFmpeg, Whisper.cpp, and VAD executables plus
explicit model paths. VAD is required to reduce non-speech hallucinations.

```bash
capicove audio transcribe \
  --input ./speech.wav \
  --model ./models/ggml-model.bin \
  --vad-model ./models/ggml-silero-vad.bin \
  --store ./.capicove/audio \
  --language auto \
  --output json
```

For a multi-stream M4A or WebM container, add `--audio-stream-index N`, where `N`
is the zero-based audio-stream ordinal used by FFmpeg `0:a:N`. A single stream may
omit the flag. Multiple streams may not: Capicove rejects ambiguity instead of
guessing from default or language metadata, mixing streams, or falling back.

Successful output includes the selected ordinal, selection method, preflight stream
count, index semantics, and `selected_audio_stream_only` validation scope under
`lineage.audio_stream_selection`.

Use `capicove audio capabilities` with the same model paths to report the actual
local deployment and executable versions. Timeouts, source changes, incompatible
models, and backend failures are represented explicitly.

The Linux CPU acceptance profile uses the multilingual `ggml-tiny.bin` model and
`ggml-silero-v6.2.0.bin` with a pinned Whisper.cpp `v1.9.5` source build. These are
separately provisioned assets, not files included in the Python wheel. Record
their upstream revisions and SHA-256 values, then pass absolute model and
executable paths when the binaries are not on `PATH`. The model's multilingual
capability does not establish application-specific transcription accuracy.

The opt-in native suite verifies actual speech, silence, damaged model rejection
and multi-stream isolation. See [models and service profiles](../../getting-started/runtime-profiles.md#repeat-the-added-runtime-acceptance)
for the hash-pinned qualification configuration and command.

See the [audio CLI reference](../../cli/audio.md) and
[Python API](../../api/audio.md).
