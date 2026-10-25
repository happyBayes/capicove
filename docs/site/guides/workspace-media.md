# Workspace video and audio Tools

## Audio clipping

Select `capicove.audio.clip` in `enabled_tools` and use audio configuration version
2 with an `output_root`. Existing audio configuration version 1 remains supported
for import/transcription. Clipping needs no Whisper executables or model assets.

```json
{
  "schema_version": 2,
  "read_roots": ["/home/me/recordings"],
  "staging_root": "/home/me/cove/audio-staging",
  "evidence_root": "/home/me/cove/audio-evidence",
  "output_root": "/home/me/cove/audio-clips",
  "ffmpeg": "/usr/bin/ffmpeg",
  "ffprobe": "/usr/bin/ffprobe",
  "max_input_bytes": 1073741824,
  "max_duration_seconds": 3600
}
```

This object belongs under `capabilities.audio` in the complete version-2 host
configuration. Select the `write` or `admin` profile and permissions `audio:read`,
`artifact:write`, and `subprocess:execute`. The owner-private output root must be
inside `workspace_root` and disjoint from read, staging, and evidence roots.
It is created only during an explicit call; nested destination parents must exist.

The MCP name is `capicove_audio_clip`, with the following canonical arguments:

```json
{
  "source": {"type": "path", "path": "/home/me/recordings/meeting.m4a"},
  "destination": {"type": "path", "path": "/home/me/cove/audio-clips/excerpt.wav"},
  "start_seconds": 12.5,
  "end_seconds": 27.75
}
```

An agent can translate a request such as “keep 12.5–27.75 seconds” into this call.
Content-based requests first use transcription timestamps to locate the range.
The Tool accepts one audio stream and produces one PCM16 WAV; it never overwrites
a destination. Source/destination paths are redacted from traces. The result
returns basenames and verified hashes rather than absolute paths. After a failed
call, reconcile the destination before a new invocation; automatic replay is unsafe.

Clipping inherits the Linux-only host handoff and executable pinning. Its fixed
transform commands additionally use inherited `fd:` media descriptors rather than
mutable input/output paths. No protocol other than the selected local descriptor
is permitted for these commands. Host startup and catalog discovery run no media
subprocesses and create no directories.

A version-2 workspace configuration can select local video and audio Tools. The
CLI (`--runtime-config`) and MCP (`capicove mcp serve`) then use the same
bindings, roots, and permissions. Nothing is downloaded, and startup, doctor, and
catalog never run FFmpeg, FFprobe, or Whisper.

!!! note "Current status"
    These capabilities implement the accepted
    [RFC-0021](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0021-workspace-video-capability.md)
    (video) and
    [RFC-0022](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0022-workspace-audio-capability.md)
    (audio). Split, filter, import, and transcription run on Linux only, because
    they execute pinned executables and write through descriptor-bound paths;
    they fail closed elsewhere. Deduplicate also runs on other POSIX systems.
    Whisper.cpp 1.9.5-dev with pinned multilingual tiny and Silero VAD v6.2.0
    passed seven installed-wheel CPU cases covering speech, silence, corrupt
    assets, and stream selection/isolation. This functional check does not measure
    Chinese production transcription accuracy; evaluate representative recordings
    for the language and domain you deploy. See
    [models and service profiles](../getting-started/runtime-profiles.md).

## How media reaches a backend

Every call copies the authorized source into a private, read-only file under
`staging_root` before any backend runs
([ADR-0026](https://github.com/happyBayes/capicove/blob/main/docs/adr/0026-host-media-handoff.md)).
The copy is opened through no-follow handles and checked for size, container
type, and an unchanged file identity. If the source changes during the copy, the
call fails with no output. If it changes afterwards, the result still describes
the verified copy. Results always report the original path and SHA-256; the
staging directory is removed when the call ends.

## Executables

FFmpeg, FFprobe, and Whisper paths are checked once at startup (identity and
SHA-256). Every call reopens them, confirms they are unchanged, and runs exactly
that verified file. If an executable is replaced or upgraded, calls fail with
`unavailable` until the host is restarted. Executables are never looked up on
`PATH`.

## Video

```json
"capabilities": {
  "video": {
    "schema_version": 1,
    "read_roots": ["/home/me/videos"],
    "staging_root": "/home/me/cove/video-staging",
    "output_root": "/home/me/cove/clips",
    "backend": "ffmpeg",
    "ffmpeg": "/usr/bin/ffmpeg",
    "ffprobe": "/usr/bin/ffprobe",
    "max_input_bytes": 4294967296
  }
}
```

| Tool | Profile | Permissions |
| --- | --- | --- |
| `capicove.video.split` | write, admin | `video:read`, `artifact:write`, `subprocess:execute` |
| `capicove.video.filter` | read, write, admin | `video:read`, `subprocess:execute` |
| `capicove.video.deduplicate` | read, write, admin | `video:read` |

- Split writes segments only under `output_root`, which is required when split
  is selected, together with `ffmpeg` (every backend, including `av`, writes clips
  with FFmpeg). Outputs are written through a directory handle pinned at call
  time, so replacing `output_root` cannot redirect them.
- Filter accepts only the `metadata` and `motion` criteria. Model-based criteria
  (`ocr_area`, `aesthetics`, `nsfw`, `watermark`, `tags`,
  `frame_text_similarity`) are rejected before any media is read.
- Deduplicate takes up to 1,000 absolute file paths, hashes each once with
  xxHash3-128 and SHA-256, and returns groups of byte-identical files. It never
  moves or deletes anything.
- Selecting only deduplicate needs neither FFmpeg nor PyAV; it needs the
  `video-dedup` extra.

```bash
capicove video split --runtime-config /home/me/cove/host.json \
  --input /home/me/videos/talk.mp4 --strategy duration --split-duration 30
capicove video deduplicate --runtime-config /home/me/cove/host.json \
  /home/me/videos/a.mp4 /home/me/videos/b.mp4
```

With `--runtime-config`, split refuses `--output-directory` and stdin input, and
deduplicate refuses the legacy scanning and action options.

## Audio

```json
"capabilities": {
  "audio": {
    "schema_version": 1,
    "read_roots": ["/home/me/recordings"],
    "staging_root": "/home/me/cove/audio-staging",
    "evidence_root": "/home/me/cove/audio-evidence",
    "ffmpeg": "/usr/bin/ffmpeg",
    "ffprobe": "/usr/bin/ffprobe",
    "whisper_cli": "/opt/whisper/bin/whisper-cli",
    "whisper_vad_cli": "/opt/whisper/bin/whisper-vad-speech-segments",
    "model": {"path": "/opt/whisper/models/ggml-small.bin", "sha256": "<64 hex>"},
    "vad_model": {"path": "/opt/whisper/models/ggml-silero.bin", "sha256": "<64 hex>"},
    "max_input_bytes": 1073741824,
    "max_duration_seconds": 7200
  }
}
```

| Tool | Profile | Permissions |
| --- | --- | --- |
| `capicove.audio.import` | write, admin | `audio:read`, `audio:write`, `artifact:write`, `subprocess:execute` |
| `capicove.audio.transcribe` | write, admin | `audio:read`, `artifact:write`, `subprocess:execute` |

- Import preserves the original bytes in the content-addressed `evidence_root`.
  Importing identical bytes again returns the same evidence ID with
  `"reused": true` and writes nothing new.
- A file with several audio streams needs `audio_stream_index`; otherwise the
  call is rejected with `AUDIO_STREAM_SELECTION_REQUIRED` and the stream count.
- Startup hashes each configured model. A mismatch reports `asset_integrity` in
  `setup doctor` and disables transcription. Each transcription copies the models
  into staging while checking their hashes, so Whisper always reads verified bytes.
- Import-only selection needs no Whisper executables or models.
- Each file must use a supported extension (`.wav`, `.mp3`, `.flac`, `.ogg`,
  `.opus`, `.m4a`, `.aac`, `.webm`) and start with that container's signature.
  FFmpeg and FFprobe always run with a forced demuxer and local-file access only,
  so playlists and other indirect formats cannot pull in other files.

```bash
capicove audio import --runtime-config /home/me/cove/host.json \
  --input /home/me/recordings/meeting.m4a --audio-stream-index 0
capicove audio transcribe --runtime-config /home/me/cove/host.json \
  --input /home/me/recordings/meeting.m4a --audio-stream-index 0 --output json
```

With `--runtime-config`, transcription refuses `--model`, `--vad-model`,
`--store`, and the executable options; the host config owns them.
