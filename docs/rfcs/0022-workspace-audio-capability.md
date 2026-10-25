# RFC-0022: Audio import and transcription in the workspace host

- Status: Accepted by the maintainer on 2026-10-06 (package WP-3 of the
  next-stage proposal (internal record; public support status is documented in the capability guide)).
- Date: 2026-10-06
- Owners: Capicove maintainers
- Related: [RFC-0017](0017-workspace-cli-mcp-host-v2.md),
  [ADR-0026](../adr/0026-host-media-handoff.md) (shared media handoff),
  [ADR-0009](../adr/0009-immutable-audio-evidence-intake.md),
  [ADR-0010](../adr/0010-local-offline-audio-transcription.md)

## Scope

`capabilities.audio` makes a new `capicove.audio.import` (v1) Tool and the existing
`capicove.audio.transcribe` (v2) selectable in the version-2 host. Transcription
keeps its v2 contract unchanged: path input, `start_seconds`/`end_seconds`,
stream selection semantics, and its write side effect. Evidence-ID input,
millisecond times, caching, and ASR-origin search are deferred.

## Configuration

The 2026-10-07 clipping extension composes the already accepted RFC-0005 Tool.
Audio configuration version 2 adds `output_root`, required when
`capicove.audio.clip` is selected. It is an owner-private child of the workspace,
disjoint from read, staging, and evidence roots. Version 1 remains valid for
import and transcription. The clipping Tool requires `audio:read`,
`artifact:write`, and `subprocess:execute` under write/admin profiles; it requires
no Whisper assets. The canonical clipping contract stays at version 1.

Clipping inputs use the existing verified staging handoff, with invocation
deadline checks during copy. Its source evidence is subsequently read through a
pinned no-follow descriptor and its output through a service-created descriptor.
These clipping-specific FFprobe/FFmpeg commands force the demuxer and permit only
the `fd` protocol, as required by RFC-0005. Import/transcription retain their
documented `file`-only command policy. Source and destination remain authorized
local path references; no network or remote-resource surface is introduced.

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1` |
| `read_roots` | 1–16 unique absolute existing directories without symlink components |
| `staging_root` | Owner-private directory strictly inside `workspace_root`, disjoint from inputs and evidence |
| `evidence_root` | Owner-private content-addressed store strictly inside `workspace_root`, disjoint from inputs and staging |
| `ffmpeg`, `ffprobe` | Absolute executables for validation and stream inspection |
| `whisper_cli`, `whisper_vad_cli` | Absolute executables; required when transcription is selected |
| `model`, `vad_model` | `{"path": absolute, "sha256": hex}`; required when transcription is selected |
| `max_input_bytes` | Integer 1 through 64 GiB, checked before copy or decode |
| `max_duration_seconds` | Integer 1 through 14,400, enforced on the selected stream |

Import-only selection needs no Whisper executables or models. Startup, doctor, and
catalog check paths and executable bits, and hash each configured model through
pinned no-follow handles; they run no subprocess and load no model. A model whose
SHA-256 differs from the configuration fails with the doctor code
`asset_integrity`. Missing executables report `missing_dependency`.

## Tools and authority

| Tool ID | Contract | Profiles | Required permissions |
| --- | --- | --- | --- |
| `capicove.audio.import` | new v1, write | write, admin | `audio:read`, `audio:write`, `artifact:write`, `subprocess:execute` |
| `capicove.audio.transcribe` | existing v2, write | write, admin | `audio:read`, `artifact:write`, `subprocess:execute` |

`audio:read` and `audio:write` are added to the write and admin ceilings only.

### `capicove.audio.import` v1

Input: `{"source": {"type": "path", "path": absolute}, "audio_stream_index"?: int}`.
The source is staged through ADR-0026, validated and inspected with the
configured FFmpeg/FFprobe, and preserved with the existing content-addressed
intake. Output:

```json
{
  "evidence": {"evidence_id": "sha256:...", "sha256": "...", "size_bytes": 1, "media_type": "audio/wav", "source_name": "a.wav"},
  "audio_stream_selection": {"selected_audio_stream_index": 0, "selection_method": "...", "...": "..."},
  "reused": false
}
```

Identical bytes return the same evidence ID with `reused: true` and write no new
evidence payload. A multi-stream source without `audio_stream_index` fails with
the existing `AUDIO_STREAM_SELECTION_REQUIRED` code and stream count.
Errors reuse the transcription Tool's stable audio failure codes.

### Transcription in the host

Sources are staged, and each model file is copied into the invocation's staging
directory while its SHA-256 is checked against the configuration, so Whisper
reads verified bytes even if the configured file changes later. Results report
the original source name and path, never a staging locator.

## CLI

`capicove audio transcribe` gains `--runtime-config PATH`; the configured mode
rejects `--model`, `--vad-model`, `--whisper-cli`, `--whisper-vad-cli`,
`--ffmpeg`, `--ffprobe`, and `--store`. A new
`capicove audio import --runtime-config PATH --input PATH [--audio-stream-index N]`
runs the import Tool. Legacy standalone transcription keeps its explicit options.

## Qualification note

Real ASR qualification requires locally provisioned, pinned Whisper and VAD
models and all four executables. Tests with fake executables cover authority,
handoff, integrity, and error paths only; they do not qualify ASR quality.

## Review amendment (2026-10-06)

The maintainer review of the first implementation required these corrections, now part of this contract:

- The import result has no `duration_seconds` field; duration is enforced, not reported.
- Each source must use a supported extension whose leading bytes match that container
  (for example `.wav` requires `RIFF…WAVE`). Anything else is rejected before any
  FFprobe or FFmpeg process runs.
- Every FFmpeg and FFprobe invocation in the host forces the demuxer for that container
  and allows only the local `file` protocol (`-protocol_whitelist file -f <demuxer>`),
  so playlist, concat, or network demuxers can never read secondary inputs.
- Configured executables are snapshotted (identity and SHA-256) at startup and
  executed per call through `/proc/<pid>/fd/<n>` paths bound to the verified open
  file. A replaced or modified executable fails the call with `unavailable` and
  is never run. This requires Linux; other platforms fail closed.
- Configured CLI calls validate the complete selected catalog, as MCP startup does.
