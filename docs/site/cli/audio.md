# Audio commands

## `capicove audio clip`

Extract one half-open time range to a new PCM16 WAV file:

```bash
capicove audio clip --input meeting.m4a --destination excerpt.wav \
  --start 12.5 --end 27.75 --output json
```

Inputs may use `.aac`, `.flac`, `.m4a`, `.mp3`, `.ogg`, `.opus`, `.wav`, or
`.webm`; the container signature and actual decoding must also pass validation.
Clipping requires exactly one audio stream. The source is preserved byte-for-byte
in the immutable evidence store (default `.capicove/audio-evidence`).

Optional flags are `--sample-rate-hz` (8,000–96,000), `--channels` (1 or 2),
`--profile pcm16-wav`, `--store`, `--ffmpeg`, `--ffprobe`, and `--timeout`
(greater than zero, at most 300 seconds). Without explicit conversion, a supported
source rate and mono/stereo geometry are preserved. Multichannel sources require
an explicit channel count. The destination parent must exist and the destination
must use lowercase `.wav`. Existing files or symlinks are never overwritten.

The canonical result reports requested and actual source ranges, verified output
geometry, source evidence and artifact hashes, and backend/configuration lineage.
It contains basenames rather than absolute paths. PCM16 output can be larger than
compressed input and reduces the precision of higher-bit-depth sources.

Workspace execution uses the same Tool and envelope:

```bash
capicove audio clip --runtime-config host.json \
  --input /home/me/recordings/meeting.m4a \
  --destination /home/me/cove/audio-clips/excerpt.wav \
  --start 12.5 --end 27.75 --output json
```

The configured mode rejects `--store`, `--ffmpeg`, `--ffprobe`, and `--timeout`;
the host owns those settings. See [workspace media](../guides/workspace-media.md).
Clipping is synchronous and non-idempotent. After any failure, inspect the
authorized destination before issuing a new request; transports must not replay
the operation automatically. Source evidence may remain after a failed request.

## Clipping capability discovery

```bash
capicove audio capabilities --component clip --output json
```

No Whisper or VAD model is required. Discovery checks the local backend and
seekable descriptor protocols; each actual evidence/output filesystem is admitted
again during a call. A successful discovery report does not certify an arbitrary
destination. The standalone profile requires CPython 3.11+, Linux or macOS,
secure POSIX filesystem primitives, and FFmpeg/FFprobe with working `fd:` input
and seekable FFmpeg output. Workspace media execution currently requires Linux.

## `capicove audio transcribe`

```text
capicove audio transcribe --input PATH --model PATH --vad-model PATH
  [--store PATH]
  [--language {auto,zh,en,zh-CN,en-US,en-GB}]
  [--whisper-cli PATH]
  [--whisper-vad-cli PATH]
  [--ffprobe PATH]
  [--ffmpeg PATH]
  [--audio-stream-index NON_NEGATIVE_INTEGER]
  [--timeout SECONDS]
  [--output {human,json}]
```

The input, model, VAD model, store, and executable paths are local. A timeout must
be positive and bounded by the CLI policy.

`--audio-stream-index` is the zero-based audio-stream ordinal used by FFmpeg
`0:a:N`. Omit it only for a container with exactly one audio stream. Multi-stream
input without an explicit ordinal is rejected; default disposition and language
metadata are not selection rules.

## Workspace host mode

`capicove audio transcribe --runtime-config CONFIG --input PATH` and
`capicove audio import --runtime-config CONFIG --input PATH [--audio-stream-index N]`
use the version-2 host's `capabilities.audio` authority. In this mode the model,
VAD model, store, and executable options are rejected. See
[workspace video and audio Tools](../guides/workspace-media.md).

## `capicove audio capabilities`

```text
capicove audio capabilities --model PATH --vad-model PATH
  [--whisper-cli PATH]
  [--whisper-vad-cli PATH]
  [--ffprobe PATH]
  [--ffmpeg PATH]
  [--output {human,json}]
```

This command reports the configured model hashes and actual executable versions.
Use it in deployment diagnostics and acceptance records.
Omitting `--component` preserves the transcription behavior and requires both
`--model` and `--vad-model`.
