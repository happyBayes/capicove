# Inspect video media quality

`capicove video quality` inspects an explicitly selected local MP4/MOV or MKV/WebM
file using local FFmpeg and FFprobe. It needs no model weights or model framework,
never downloads media, and does not rewrite the source. Install FFmpeg and
FFprobe separately and make them available to your standalone composition.

```bash
capicove video quality --input /absolute/lecture.mp4 --output json
```

The report contains nine check groups: stream inventory, video format, audio
format, full selected-stream decoding, black intervals, static intervals, quiet
intervals, loudness/true peak, and audio/video timing differences. Each metric
has a value, unit and explicit reason when unavailable. Source SHA-256, selected
stream indexes, configuration hash, backend versions and observed coverage make
results traceable.

## Review observations and delivery policies

A blackboard background, static slide or pause may be intentional. Black/static/
quiet events yield `review` observations by default, not an automatic failure.
The Tool does not generate a 1–10 quality grade or judge educational, scientific,
aesthetic or legal correctness.

Save explicit delivery requirements as a policy object:

```json
{
  "require_audio": true,
  "min_display_width": 1280,
  "min_display_height": 720,
  "allowed_video_codecs": ["h264", "hevc"],
  "max_true_peak_dbtp": -1.0,
  "max_abs_end_delta_seconds": 0.1
}
```

```bash
capicove video quality --input /absolute/lecture.mp4 \
  --policy /absolute/policy.json --output json
```

These are example requirements, not universal educational-media or publishing
standards. Policy violations produce a completed inspection with `verdict=fail`.
Missing a measurement required by policy instead produces an incomplete failed
inspection. Omitted requirements make no quality claim. Select all checks needed
by a policy; a policy cannot silently enable an unrequested check.

## Detector configuration

Defaults are a 2-second minimum interval, near-black picture ratio 0.98, black
pixel threshold 0.10, static noise tolerance 0.001, and quiet threshold -50 dBFS.
Use `--detector-options` with a closed JSON object to change detection settings:

```json
{
  "black_min_seconds": 2.0,
  "black_picture_ratio": 0.98,
  "black_pixel_threshold": 0.10,
  "freeze_min_seconds": 3.0,
  "freeze_noise": 0.001,
  "silence_min_seconds": 2.0,
  "silence_threshold_dbfs": -50.0,
  "silence_per_channel": false
}
```

Per-channel quiet mode retains channel indexes; its overall interval fraction is
union time with any detected quiet channel, not an average over channels.
Detector transitions are quantized to video/audio frame boundaries. Static
interval discovery may need the next frame to reach its minimum elapsed time;
intervals exactly at that boundary may not be emitted by the backend.

Loudness reports integrated LUFS, loudness range in LU, and true peak in dBTP.
Entirely silent or below-gate audio retains null measurements with a reason,
never NaN/infinity or fabricated ordinary voice loudness. No gain normalization
is applied. Timing differences use a shared source-relative presentation origin;
they do not establish lip synchronization or narration/animation alignment.

## Selection, coverage and limitations

Multiple eligible video or audio tracks require explicit absolute container stream
indexes via `--video-stream-index` / `--audio-stream-index`. Attached cover images
are excluded from eligible video tracks. Only selected tracks are decoded; an
unselected alternate soundtrack is not audited. A missing independent subtitle
track does not prove that burned-in captions are absent.

Use `--checks streams video_format audio_format` for metadata-only inspection.
Unrequested checks are `not_run`; metadata-only success is not whole-file quality
acceptance. Timing and counts use a structured FFprobe frame pass per selected track,
followed by one shared FFmpeg detector/loudness pass. These passes share the
verified input and deadline and are listed in `coverage.passes`. This avoids
using interleaved human-readable logs as frame-count or timestamp evidence.

A full scan retains `progress=end`, decoder diagnostics and observed
frames/samples; missing filters, decoding errors, deadlines and unknown required
measurements cannot silently pass. Interrupted intervals are not closed using
container duration metadata. Event details are bounded and truncation is marked;
exact totals continue to be collected within the execution budget.

Standalone defaults are 4 GiB per input, 7,200 seconds of media, a 300-second Tool
deadline, two FFmpeg threads, 1,000 retained events per detector, and bounded
provider output. Library composition may select different limits through
`QualityLimits`; Tool callers cannot enlarge them. No-follow input paths reject
symlink components. Only supported self-contained containers are accepted;
playlists and external media references are rejected.

## Configured workspace host (Linux)

The version-2 host selects `capicove.video.quality` in `enabled_tools` and exposes
`capicove_video_quality` in its catalog, MCP stdio server, and configured CLI.
This route requires Linux procfs (ADR-0026); standalone inspection remains
available on macOS. Use this complete configuration with your actual absolute
paths and an existing owner-only workspace directory:

```json
{
  "schema_version": 2,
  "server_name": "capicove-local",
  "workspace_id": "quality-review",
  "actor_id": "reviewer",
  "environment_id": "local",
  "workspace_root": "/srv/capicove-quality",
  "profile": "read",
  "permissions": ["video:read", "subprocess:execute"],
  "enabled_tools": ["capicove.video.quality"],
  "trace_store": null,
  "capabilities": {
    "video": {
      "schema_version": 1,
      "read_roots": ["/srv/lesson-videos"],
      "staging_root": "/srv/capicove-quality/staging",
      "backend": "ffmpeg",
      "ffmpeg": "/usr/bin/ffmpeg",
      "ffprobe": "/usr/bin/ffprobe",
      "max_input_bytes": 1073741824,
      "quality_limits": {"threads": 1, "max_events": 500}
    }
  },
  "network": {"enabled": false, "source_hosts": []}
}
```

Protect `host.json` with mode 0600 and its workspace with mode 0700. Read roots
must be disjoint from staging. Both executable paths must be absolute regular
files without symlink components; resolve your installation paths when writing
the configuration. `output_root` is unnecessary for quality-only inspection.

```bash
capicove setup doctor --config /srv/capicove-quality/host.json
capicove mcp catalog --config /srv/capicove-quality/host.json
capicove mcp serve --config /srv/capicove-quality/host.json
capicove video quality --runtime-config /srv/capicove-quality/host.json \
  --input /srv/lesson-videos/lecture.mp4 --output json
```

Catalog/readiness never decode media. Calls use the host's authorized staging
handoff, revalidate source identity and content, and keep staged files and
configured executables pinned through all provider passes. Reports retain the
original authorized path, SHA-256 and size; private descriptor/staging locators
are removed. Invocation scratch is cleaned after success or failure.

Optional `quality_limits` accepts `max_input_bytes`, `max_duration_seconds`,
`max_events`, `max_records`, `max_diagnostic_bytes`, `max_probe_bytes`, and
`threads`. Each must be positive (integers except duration) and may only narrow
the standalone defaults above; input bytes also cannot exceed the host's
`max_input_bytes`. Limits appear in the report's configuration and digest.
Callers cannot override roots, executables or resource budgets.

For configured calls, opt into tracing through the host's `trace_store`; combining
`--runtime-config` with the standalone global `--trace-store` is rejected before
opening a local store. Remote service qualification is separate from this local
workspace integration. Media observations do not establish teacher, listener or
release acceptance.

See the [CLI reference](../../cli/video-quality.md) and
[Python/MCP API](../../api/video.md#media-quality-inspection).
