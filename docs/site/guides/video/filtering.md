# Filter video

Unified video filtering evaluates one or more local files against a non-empty JSON
criterion array. Criteria can inspect metadata, sampled frames, motion, OCR text,
text similarity, tags, or injected local model scores.

For example, `criteria.json` can enforce duration and display width:

```json
[
  {
    "type": "metadata",
    "criterion_id": "delivery-shape",
    "min_duration_seconds": 2,
    "min_display_width": 1280
  }
]
```

```bash
capicove video filter \
  --input ./candidate.mp4 \
  --criteria ./criteria.json \
  --aggregation all \
  --output json
```

Model-backed criteria require explicit local model references. Capicove does not
download model weights or call hosted inference APIs. A failed criterion or media
decode is represented in the structured result; callers should inspect status and
failure details rather than only checking whether a process returned output.

See the [CLI reference](../../cli/video-filter.md) and
[Python API](../../api/video.md).
