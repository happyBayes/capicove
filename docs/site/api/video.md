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
