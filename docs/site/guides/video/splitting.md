# Split video

`capicove video split` supports duration, keyframe, and scene strategies. Accurate
transcoding is the default write mode; stream copying is an explicit performance
trade-off because copied boundaries can move to nearby keyframes.

```bash
capicove video split \
  --input ./input.mp4 \
  --strategy duration \
  --split-duration 30 \
  --output-directory ./clips \
  --output json
```

Use `--artifact-ref` with `--artifact-sha256` instead of `--input` when a configured
artifact resolver supplies immutable evidence. Output paths must pass the active
path policy. Byte output is intended for transport callers and may be unsuitable
for large media.

See the complete [CLI options](../../cli/video-split.md) and
[Python API](../../api/video.md).
