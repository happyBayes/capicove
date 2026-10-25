# Video deduplication CLI

Exact-file video deduplication groups byte-identical files using the optional
xxHash backend. It does not decode video and does not classify visually similar
encodes as duplicates.

```bash
capicove video deduplicate videos --print-unique
capicove video deduplicate videos --write-manifest upload.json
```

With `--runtime-config CONFIG`, the command groups explicit file paths through
the host's read-only `capicove.video.deduplicate` Tool and prints its canonical
result; scanning, manifest, and move or delete options are rejected. See
[workspace video and audio Tools](../guides/workspace-media.md).

Install `capicove[video-dedup]` first. Inputs can be files or directories. Use
`--help` to inspect traversal, action, and manifest options before applying any
filesystem action.
