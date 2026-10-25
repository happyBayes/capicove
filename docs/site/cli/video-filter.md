# `capicove video filter`

Evaluate one or more videos using a UTF-8 JSON criterion array.

```text
capicove video filter --input PATH [--input PATH ...]
  --criteria FILE
  [--text TEXT ...]
  [--aggregation {all,any}]
  [--output {human,json}]
  [--runtime-config CONFIG]
  [--ocr-provider {paddleocr,none}]
  [--ocr-model-root ABSOLUTE_PATH]
  [--model-config JSON_OR_TOML --model-binding video.ocr]
```

With `--runtime-config`, the filter runs through the version-2 host's
`capabilities.video` authority and accepts only `metadata` and `motion` criteria.
See [workspace video and audio Tools](../guides/workspace-media.md).

Each `--text` value is paired by position with an input and is required by
text-similarity criteria. `all` requires every criterion to pass; `any` requires at
least one. Criteria that use OCR, models, or tags need their corresponding optional
dependencies and explicit local resources.

Standalone OCR-area filtering defaults to PaddleOCR. Install the optional
`capicove-processor-paddleocr` plugin and provide its verified `book-balanced`
model root with `--ocr-model-root`. `--ocr-provider none` disables OCR. No fallback
downloads or selects another engine. The metric is the union area of recognized
line boxes. These flags do not enable OCR in `--runtime-config` host mode.
See [local BGE-M3 and Paddle setup](../guides/local-bge-paddle.md).

Alternatively select the standalone OCR provider through the
[model factory configuration](../guides/model-factory.md) using `--model-config`.
Its default binding is `video.ocr`; the factory owns the provider and model root.
It cannot be combined with conflicting OCR flags or `--runtime-config`.
