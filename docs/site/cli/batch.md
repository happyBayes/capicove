# Batch

Run a resumable image or video operation over a local directory:

```text
capicove batch run \
  --input DIRECTORY \
  --operation OPERATION \
  --parameters FILE \
  --state FILE \
  --provider-fingerprint SHA256 \
  [--artifact-root DIRECTORY] \
  [--workers 1..64] \
  [--modality-workers 1..64] \
  [--max-retries 0..10] \
  [--initial-backoff SECONDS] \
  [--max-backoff SECONDS] \
  [--output human|json]
```

Supported unified operations are `image_aspect_ratio_filter`,
`image_shape_filter`, `image_size_filter`, `image_subplot_filter`, `video_filter`,
and `video_split`.

`--parameters` must contain one JSON object. `video_split` additionally requires an
absolute or resolvable `--artifact-root` outside the input tree. Resume and content
cache reuse are automatic when the state path and all cache-affecting identities are
unchanged.

Use the global `--trace-store FILE` option before `batch run` to record each actual
Tool attempt for later multimodal trajectory export. A cache hit does not create a
synthetic Tool Episode.

See [Resumable image and video batches](../guides/directory-batches.md) for complete
examples and parameter shapes.
