# RFC-0021: Local video Tools in the workspace host

- Status: Accepted by the maintainer on 2026-10-06 (package WP-2 of the
  next-stage proposal (internal record; public support status is documented in the capability guide)).
- Date: 2026-10-06
- Owners: Capicove maintainers
- Related: [RFC-0017](0017-workspace-cli-mcp-host-v2.md),
  [ADR-0026](../adr/0026-host-media-handoff.md) (shared media handoff)

## Scope

`capabilities.video` makes three local operations selectable in the version-2
host: the existing `capicove.video.split` (v1) and `capicove.video.filter` (v1),
and a new read-only `capicove.video.deduplicate` (v1). Filter accepts exactly the
`metadata` and `motion` criteria. `ocr_area`, `aesthetics`, `nsfw`, `watermark`,
`tags`, and `frame_text_similarity` remain unavailable until a host provider and
asset contract is accepted. Perceptual deduplication, scanning, deletion, and
Video Search are out of scope.

## Configuration

`capabilities.video` is a closed version-1 object:

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1` |
| `read_roots` | 1–16 unique absolute existing directories without symlink components |
| `staging_root` | Owner-private directory strictly inside `workspace_root`, disjoint from every read root |
| `output_root` | Owner-private directory inside `workspace_root`, disjoint from read roots and `staging_root`; required when split is selected |
| `backend` | `auto`, `ffmpeg`, or `av`; `auto` uses FFmpeg when configured, otherwise PyAV |
| `ffmpeg`, `ffprobe` | Absolute executable paths or `null`; there is no `PATH` lookup |
| `max_input_bytes` | Integer from 1 through 64 GiB, checked before copy or decode |

Requirements depend on the selection:

| Selected Tool | Requires |
| --- | --- |
| split | `output_root`; `ffmpeg` and `ffprobe` for `ffmpeg`, PyAV for `av`, either for `auto` |
| filter | `ffprobe` (metadata) and PyAV with OpenCV (motion sampling) |
| deduplicate | the `xxhash` package only |

Startup, `setup doctor`, and `mcp catalog` check paths, executable files, and
importable modules only. They run no subprocess and decode no media.

## Tools and authority

| Tool ID | Contract | Profiles | Required permissions |
| --- | --- | --- | --- |
| `capicove.video.split` | existing v1, write | write, admin | `video:read`, `artifact:write`, `subprocess:execute` |
| `capicove.video.filter` | existing v1, read | read, write, admin | `video:read`, `subprocess:execute` |
| `capicove.video.deduplicate` | new v1, read | read, write, admin | `video:read` |

The profile ceilings add `video:read` and `subprocess:execute` to every profile
and `artifact:write` to write and admin. Read profiles may use invocation scratch
under `staging_root` but cannot select split, so they never publish output.

Host bindings accept only `path` sources below a read root. Inline bytes,
artifact references, and caller-selected models or executables are rejected
before intake. Media is staged through the shared handoff (ADR-0026); backends
read only the verified staged copy, and results report the original path and
SHA-256, never a staging locator. Split writes only under `output_root`.

### `capicove.video.deduplicate` v1

Input: `{"paths": [absolute path, ...]}` with 1 to 1,000 unique entries. Each
file is read once through pinned no-follow handles, with its identity checked
before and after the read; xxHash3-128 candidates are confirmed by SHA-256.
The total bytes read are bounded by 1,000 × `max_input_bytes` and the Tool
deadline. Output:

```json
{
  "file_count": 3,
  "groups": [{"sha256": "...", "xxh3_128": "...", "size_bytes": 1024, "paths": ["/a.mp4", "/b.mp4"]}],
  "failures": [{"path": "/c.mp4", "code": "unreadable"}],
  "lineage": {"fast_hash": "xxh3_128", "xxhash_version": "...", "confirmation": "sha256"}
}
```

Groups contain two or more paths sorted by string; groups are sorted by their
first path; failures are sorted by path and capped at 1,000. Nothing is moved,
deleted, or written.

## CLI

`capicove video split`, `video filter`, and `video deduplicate` gain
`--runtime-config PATH`. Configured calls use the same bindings, identity,
permissions, and roots as MCP. In that mode split rejects `--output-directory`
and `--artifact-ref` (output goes to `output_root`), and deduplicate takes
explicit file paths and rejects the legacy scan and action options. Without
`--runtime-config`, the standalone commands behave as before.

## Review amendment (2026-10-06)

The maintainer review of the first implementation required these corrections, now part of this contract:

- Every split backend writes clips with FFmpeg, so `ffmpeg` is required whenever
  split is selected, including `backend: "av"`; the PyAV writer receives the
  configured executable and never searches `PATH`.
- Executables are identity-pinned at startup and run per call through
  `/proc/<pid>/fd/<n>` paths; replacements are never executed.
- Staging and split outputs are allocated, written, finalized, and discarded only
  through directory descriptors pinned at call time. Backends receive
  descriptor-bound `/proc/<pid>/fd/<n>/…` paths, so replacing `staging_root` or
  `output_root` cannot redirect reads or writes. Results are rejected if the
  reported source SHA-256 differs from the verified staged copy.
- Split and filter therefore require Linux in this release; deduplicate (no
  subprocess) remains available on other POSIX systems. Configured CLI calls
  validate the complete selected catalog.
