# Image processing

The image capability provides evidence-aware filters, mappers, OCR, public pHash,
invocation-local LSH search, perceptual
deduplication, and comparison metrics. Public value objects retain source hashes,
derived-output hashes, model lineage, and measurements needed to explain a result.

The base package includes Pillow, NumPy, and headless OpenCV. Learned scorers and
specialized adapters remain optional and are injected or loaded only when used.
Use Python, canonical Tool handlers, or [`capicove image run`](../../cli/image.md).
The default CLI supports the aspect-ratio, shape, size, and subplot filters,
pHash, and LSH search. Other registered tools require an explicitly configured
provider through Python or MCP; appearing in `--help` does not provision it.
The separate `capicove-download-image-models` command performs explicit model intake.

Start with the [offline Quickstart](../../getting-started/quickstart.md) for a
complete task using a synthetic image and no model downloads.

Choose a task:

- [filter images](filters.md)
- [derive or compare images](mapping.md)
- [find visual near-duplicates](deduplication.md)
- [run OCR](ocr.md)
- [prepare local models](local-models.md)
