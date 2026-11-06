# `capicove video quality`

```text
capicove video quality --input PATH
  [--checks streams video_format audio_format decode black freeze silence loudness av_timing]
  [--video-stream-index INTEGER] [--audio-stream-index INTEGER]
  [--detector-options JSON_FILE] [--policy JSON_FILE]
  [--output human|json] [--runtime-config HOST_JSON]
```

Paths are explicit local files. CLI relative paths become absolute without
following symlinks; input intake rejects symlink components. Configuration files
are bounded JSON objects. The canonical Tool accepts their validated object
contents, not paths to executable configurations or arbitrary filter strings.

Exit codes:

- `0`: complete observation, pass, or review-needed result;
- `1`: complete inspection violating an explicit delivery policy;
- `2`: invalid/rejected input or incomplete inspection.

With `--output json`, inspect the standard Tool envelope's `status`,
`result.verdict`, per-check `status`, `coverage`, `failures` and `warnings`.
`review` is not human acceptance. Raw measurements retain their units; no overall
numerical score is calculated. `--runtime-config` invokes the selected Tool through
the Linux version-2 workspace host with the same authority as MCP. See the guide
for a complete configuration. Startup failures return 2; no standalone fallback
is attempted.

`--trace-store` is the existing global opt-in trace option and appears before
`video quality`. It records the canonical operation through the shared trace
boundary. Configured calls use the host configuration's `trace_store` instead;
combining both trace authorities is rejected. The source is not copied into a permanent evidence store by this Tool.

The [guide](../guides/video/quality.md) lists checkpoint meanings, detector defaults,
policy examples, stream selection and resource limits.
