# Image Processing

Capicove image processing lives under `capicove.processing.image`. The capability
preserves input bytes and emits immutable hashes, metrics, decisions, derived
artifacts, difference regions, captions, structured OCR, or duplicate groups.

Pillow, NumPy, and headless OpenCV are core dependencies. rembg/ONNX Runtime,
Transformers/PyTorch, imagededup, and Ray are lazy optional adapters. Optional model
adapters accept only `VerifiedModelInstallation` objects, set
`local_files_only=True` and `trust_remote_code=False`, derive lineage from the
trusted lock entry and externally anchored file-manifest digest, and never enable
network access implicitly.

Recommended Hugging Face revisions are pinned in a packaged lock manifest. Model
acquisition remains a separate explicit operation documented in
[`docs/development/image-models.md`](../development/image-models.md); Tool handlers
and adapters never invoke the downloader.

## Capability catalog

| Reference name | Public class | Canonical Tool ID | Default/provider dependency |
| --- | --- | --- | --- |
| `image_remove_background_mapper` | `ImageRemoveBackgroundMapper` | `capicove.image.remove-background` | Injected provider; `capicove[image-rembg]` adapter |
| `imgdiff_difference_area_generator_mapper` | `ImgDiffDifferenceAreaGeneratorMapper` | `capicove.image.diff.areas` | Core OpenCV |
| `imgdiff_difference_caption_generator_mapper` | `ImgDiffDifferenceCaptionGeneratorMapper` | `capicove.image.diff.caption` | Injected provider; `capicove[models-dev]` PyTorch adapter |
| `image_aesthetics_filter` | `ImageAestheticsFilter` | `capicove.image.filter.aesthetics` | Injected local scorer |
| `image_aspect_ratio_filter` | `ImageAspectRatioFilter` | `capicove.image.filter.aspect-ratio` | Core Pillow |
| `image_nsfw_filter` | `ImageNsfwFilter` | `capicove.image.filter.nsfw` | Injected local classifier |
| `image_shape_filter` | `ImageShapeFilter` | `capicove.image.filter.shape` | Core Pillow |
| `image_size_filter` | `ImageSizeFilter` | `capicove.image.filter.size` | Core filesystem/bytes |
| `image_subplot_filter` | `ImageSubplotFilter` | `capicove.image.filter.subplot` | Core OpenCV |
| `image_text_matching_filter` | `ImageTextMatchingFilter` | `capicove.image.filter.text-matching` | Injected local image-text scorer |
| `image_text_similarity_filter` | `ImageTextSimilarityFilter` | `capicove.image.filter.text-similarity` | Injected local image-text scorer |
| `image_watermark_filter` | `ImageWatermarkFilter` | `capicove.image.filter.watermark` | Injected local classifier |
| `image_pair_similarity_filter` | `ImagePairSimilarityFilter` | `capicove.image.filter.pair-similarity` | Injected local embedding scorer |
| `image_phash` | `ImagePhash` | `capicove.image.phash` | Core OpenCV pHash; optional invocation-local LSH acceleration |
| `image_lsh_search` | `ImageLshSearch` | `capicove.image.lsh-search` | Core deterministic binary LSH with exact Hamming verification |
| `image_perceptual_deduplicator` | `ImagePerceptualDeduplicator` | `capicove.image.perceptual-deduplicate` | Core OpenCV pHash; optional `capicove[image-dedup]` adapter |
| `ray_image_perceptual_deduplicator` | `RayImagePerceptualDeduplicator` | `capicove.image.perceptual-deduplicate-ray` | `capicove[image-ray]`; explicitly initialized Ray runtime |
| `image_ocr` | `ImageOcr` | `capicove.image.ocr` | Explicitly installed Tesseract CLI with `chi_sim` and `eng` trained data |

## Structured OCR

`ImageOcr` preserves ordered source hashes and returns full text plus word-level
pixel boxes, normalized confidence, and page/block/paragraph/line/word hierarchy.
Its default ordered languages are Simplified Chinese and English
(`chi_sim+eng`). Callers may provide a validated ordered language list, but they
cannot provide arbitrary Tesseract flags, configuration paths, or output paths.

The capability depends on an `ImageOcrProvider` port. The private Tesseract adapter
sends normalized PNG bytes to `tesseract` over standard input and parses bounded TSV
from standard output. It declares `tesseract` as a subprocess resource, uses the
active Tool deadline to terminate the process group on timeout, never accesses the
network, and never downloads an executable or trained data. Missing Tesseract or
language data is reported as unavailable.

## Filter semantics

Range filters use inclusive lower and upper bounds, except watermark probability,
which keeps scores strictly below its threshold. `Aggregation.ANY` keeps a sample
when any requested image passes; `Aggregation.ALL` requires every requested image.
Subplot filtering uses the same enum to select whether any or all images must
contain a detected grid before the sample is rejected. Empty collections are kept
as not applicable. Pair operations require exactly two readable images.

Scorers are ports. A metric records its provider, provider version, model ID, model
revision, preprocessing revision, and execution device independently of the filter
decision. Classification adapters select semantic labels rather than assuming a
fixed class index. The aesthetics adapter supports either a single-logit regressor
or an explicitly selected classification label. Image-text matching uses the BLIP
ITM head and returns the matched-class probability; CLIP remains a separate cosine
similarity adapter. Image-pair similarity uses pooled image embeddings.

## Difference processing

Version 1 requires matching dimensions so returned pixel-space boxes have one
unambiguous coordinate system. The area mapper calculates a maximum CIELAB-channel
difference, applies thresholding and morphological closing, extracts bounded
contours, and applies deterministic non-maximum suppression. It uses no random
temporary crop files. Resizing or registration is an explicit upstream operation
whose transform must be retained in lineage.

The caption mapper consumes the resulting boxes, crops both images in memory, and
calls a structured caption provider. It never parses presentation strings such as
`ASSISTANT:`. The Hugging Face adapter uses structured `generated_text` fields and
the `image-text-to-text` task. Its difference model must natively accept both images.
A single-image VQA model is not a valid difference-model substitute.

## Perceptual deduplication

Perceptual deduplication is intended to filter visually similar images from image
training datasets. It is not byte-exact deduplication. Version 1 computes a 64-bit
pHash from low-frequency DCT coefficients. A Hamming threshold of zero groups equal
pHashes; a positive threshold enables visual near-duplicate grouping.

Near-duplicate edges are converted to connected components. The lexicographically
lowest stable record ID is retained, so input ordering and Ray scheduling cannot
change the winner. The driver authorizes local paths and materializes them to
immutable bytes before Ray distributes decoding and pHash preparation. Workers
receive the same configured hash provider, while the same component reduction is
used locally and in distributed mode. Ray must be initialized explicitly by the
caller. Each wait uses the smaller of the configured timeout and active Tool
deadline. Workers receive the absolute request deadline. When submission or waiting
fails, the coordinator force-cancels every submitted reference independently and
waits for all of them to terminate before it returns.

### Public pHash and invocation-local LSH

`ImagePhash` exposes each record's versioned 64-bit pHash and source SHA-256 values.
It also returns the same connected-component grouping used by
`ImagePerceptualDeduplicator`. `enable_lsh=True` changes candidate generation only:
the implementation creates deterministic `radius + 1` bit bands and then verifies
every candidate with exact Hamming distance. It does not accelerate image decoding
or the DCT calculation itself.

`ImageLshSearch` accepts already-produced compatible hashes and returns matches in
ascending `(distance, record_id)` order. Its table exists for one invocation only;
it neither writes an index nor participates in retrieval-owned generation and
activation lifecycles. Persistent or cross-request image search remains a retrieval
capability.

## Filesystem and resource boundaries

Tool path inputs are authorized and materialized as bounded regular-file bytes
through no-follow, descriptor-relative traversal before a capability starts. The
capability and Ray workers never reopen the mutable user pathname. Named pipes,
symlinks, path swaps, and growth beyond the request byte budget are rejected.

Background output directories are similarly opened below an authorized root before
model work. Artifact writes remain relative to that descriptor and are published by
atomic create-if-absent linking. An existing digest target is verified and never
overwritten. Platforms without descriptor-relative filesystem operations fail
path-backed image inputs and background artifact outputs as unavailable; inline
image bytes remain supported instead of falling back to mutable pathname access.

One invocation may decode at most 100 million pixels or 300 million RGB bytes and
may spend at most 256 model calls. Difference-caption requests accept at most 64
distinct regions; exact duplicate boxes are removed before model work. These
cumulative limits complement the per-file encoded-byte and per-image pixel limits.

## Deadlines

Image capabilities enforce Tool deadlines cooperatively at decoding, model-call,
iteration, scheduling, and artifact-commit boundaries. Provider calls run in the
calling process and are checked immediately before and after inference; providers
requiring hard interruption must supply a killable worker boundary. Timeout threads
are prohibited because they cannot stop GPU work and could commit artifacts after a
timeout result.

## Tests and local fixtures

Default tests use authored in-memory images generated by test helpers. Heavy model
and Ray behavior is exercised through deterministic provider fakes, while lazy
import, local-only configuration, class-label resolution, structured responses, and
local/Ray parity are covered in the default suite. Real provider executions require
the corresponding optional extra and local model assets.
