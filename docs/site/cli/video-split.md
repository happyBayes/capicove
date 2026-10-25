# `capicove video split`

Split a video by fixed duration, keyframe boundaries, or detected scenes.

```text
capicove video split (--input PATH | --artifact-ref REF)
  --strategy {duration,keyframe,scene}
  [--artifact-sha256 SHA256]
  [--split-duration SECONDS]
  [--overlap-duration SECONDS]
  [--min-last-split-duration SECONDS]
  [--min-keyframe-segment-duration SECONDS]
  [--detector {content,adaptive,threshold}]
  [--scene-option NAME=VALUE]
  [--output-format {path,bytes}]
  [--output-directory PATH]
  [--backend {auto,ffmpeg,av}]
  [--write-mode {accurate,copy}]
  [--output {human,json}]
```

Duration splitting requires `--split-duration`. Scene-specific options are
allowlisted and repeatable. Path output requires an authorized output directory.
With `--runtime-config CONFIG`, the split runs through the version-2 host's
`capabilities.video` authority: output goes to its `output_root`, and
`--output-directory`, `--artifact-ref`, and stdin input are rejected.
`--output-format` defaults to `path` when `--output-directory` is given and to
`bytes` otherwise; combining `--output-format bytes` with `--output-directory` is a
usage error.
`accurate` is the default write mode; `copy` is faster but can shift boundaries to
keyframes.
