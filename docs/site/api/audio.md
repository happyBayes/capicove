# Audio processing API

`AudioClipRequest` and `AudioClippingService` expose the same sample-accurate
PCM16 WAV operation as the CLI and MCP Tool. Construct the service with the
existing `AudioProcessor` and an `AudioClipBackend`; the built-in adapter is
`FfmpegAudioClipBackend`. Calls require an absolute timezone-aware deadline and
return path-free evidence, output metadata, and lineage. See
[audio commands](../cli/audio.md) for constraints and failure reconciliation.

::: capicove.processing.audio
