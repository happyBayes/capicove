# Video processing API

::: capicove.processing.video

## Exact-file deduplication

Install with `pip install 'capicove[video-dedup]'`, then run a read-only scan:

```python
from pathlib import Path
from capicove.processing.video import deduplicate_video_files

result = deduplicate_video_files((Path("videos"),))
```

Use `capicove video deduplicate videos --print-unique` for upload-ready paths,
`--print-duplicates` for groups, or `--write-manifest upload.json` for a JSON/CSV
manifest. Size and XXH3-128 narrow candidates; SHA-256 confirms every group. No
file changes unless `--move-duplicates DIR` or
`--delete-duplicates --force` is supplied.

## Media quality inspection

The accepted canonical Tool is `capicove.video.quality` version 1. Its
standalone CLI is `capicove video quality`; its standalone MCP name is
`capicove_video_quality`. The Linux version-2 workspace host also selects this
Tool through `capabilities.video` and `enabled_tools`. See the
[workspace configuration](../guides/video/quality.md#configured-workspace-host-linux).

```python
from pathlib import Path

from capicove.tools import build_video_quality_tool_handler
from capicove.mcp import create_video_quality_mcp_server

handler = build_video_quality_tool_handler(
    "operator-policy",
    (Path("/absolute/authorized-videos"),),
)
server = create_video_quality_mcp_server(handler)
```

Pass the server to your existing MCP stdio runner. Construction registers schemas
without invoking FFmpeg, decoding a source, or downloading packages. The
composition owns authorized roots, executables and `QualityLimits`; invocation
arguments cannot widen them. A normal Tool invocation uses the existing
`InvocationContext` with `video:read` and `subprocess:execute` permissions.

For direct capability composition, `capicove.processing.video.quality` exports
`QualityRequest`, `QualityLimits`, `QualityInspection`, `QualityError`, and
`VideoQualityPipeline`. The pipeline consumes `QualityIntake` and `QualityBackend`
ports; provider implementations remain private. `QualityInspection.to_dict()`
returns detached finite JSON, while its underlying report is immutable.

See [media quality inspection](../guides/video/quality.md) for policies and
interpretation, and [the CLI reference](../cli/video-quality.md) for exit codes.
