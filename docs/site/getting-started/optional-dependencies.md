# Optional dependencies

Install only the execution paths you need.

See [models and service profiles](runtime-profiles.md) for the complete explicit
asset inventory, measured runtime coverage and optional-install decisions.

The `video-scene`, `video`, and `video-all` extras use PySceneDetect's
`scenedetect-headless` 0.7 distribution with `opencv-python-headless`. Do not
co-install the GUI `scenedetect` or `opencv-python` distributions in the same
environment. The adapter uses the 0.7 timestamp API; recreate older environments
when upgrading to avoid overlapping packages.

| Extra | Purpose |
| --- | --- |
| `knowledge` | Native dense retrieval, graph publication, compatible generation and PDF rendering; OCR still needs its explicit Paddle plugin/runtime and weights |
| `api` | Local HTTP API and schema validation |
| `model-serving` | Explicit local model HTTP serving |
| `video-all` | Combined video Python dependencies; local models are still required |
| `wiki` | YAML frontmatter and ontology parsing for the standalone local Wiki |
| `graph-models` | Verified local causal LLM and tokenizer adapters for document graph generation |
| `graph-remote` | Verified local chat tokenization and templates for an explicitly configured compatible LLM server; no Torch requirement |
| `text-embedding` | Verified local BGE batch text embedding adapter |
| `graph-community` | NetworkX 3.6 Louvain community adapter |
| `graph-data` | PyArrow 25 typed Parquet graph publication |
| `video` | PyAV, OpenCV, and scene detection for complete video splitting |
| `video-av` | PyAV video backend |
| `video-scene` | SceneDetect strategies |
| `video-filter` | Local frame sampling and OpenCV video filters |
| `video-filter-ocr` | OCR-backed video criteria |
| `video-filter-models` | Learned video scorers |
| `video-filter-tags` | Local video tagging |
| `video-dedup` | Exact-file video deduplication with xxHash |
| `image-rembg` | Background removal adapter |
| `image-dedup` | imagededup adapter |
| `image-ray` | Distributed perceptual deduplication |
| `models-cpu` | ONNX Runtime CPU model execution |
| `retrieval-cpu` | CPU-only ONNX Runtime for personal reverse-image search |
| `models-gpu` | GPU model execution where supported |
| `models-dev` | Model export, quantization, and development |
| `mcp` | MCP transport integration |

Example:

```bash
python -m pip install "capicove[retrieval-cpu,mcp]"
```

For a broader local Wiki, video, image-processing, and MCP environment:

```bash
python -m pip install "capicove[wiki,video,image-rembg,mcp]"
```

Execution backends are separate plugin distributions rather than Capicove extras.
For local Spark processing, install the matching release after publication:

```bash
python -m pip install "capicove-execution-spark==0.1.0"
```

`capicove-execution-spark` requires PySpark 4.1–4.2 and an executable Java 17
runtime. Plugin discovery itself is lazy and does not start Spark.

The PDF book parser has two installation steps. First install a PaddlePaddle
runtime compatible with the target CPU or accelerator by following PaddlePaddle's
platform instructions. Confirm that runtime in the Capicove environment:

```bash
python -c "import paddle; print(paddle.__version__)"
```

Then install the separate parser plugin into the same environment as the Capicove
CLI. From a repository checkout:

```bash
uv pip install -e ./plugins/processor-paddleocr
```

This second step registers the `paddleocr` document-parser entry point and installs
the PP-StructureV3 parser, pypdf, and model-download dependencies. The plugin does
not install the platform-specific PaddlePaddle runtime. Its model plan exposes the
supported languages, runtime compatibility ranges, fixed pipeline configuration,
and per-file model SHA-256 values. Parsing requires a matching schema 2 receipt and
rehashes the closed model file tree before source processing. It also validates the
exact installed PaddleOCR, PaddlePaddle, PaddleX, pypdf, and PDFium versions before
source access and binds them into the parser fingerprints.

The current `book-balanced` profile rejects pages that contain raster images and
pages without a valid, nonblank Unicode text layer. This excludes covers, logos,
charts, photos, scan-only PDFs, and hybrid native/OCR pages until an independent
source-image script gate is qualified.

Audio transcription uses user-provided `ffmpeg`, Whisper.cpp, Whisper.cpp VAD, and
local model files. Those executables and model artifacts are not Python extras and
are never downloaded automatically.

Reverse-image search also requires two already-local, verified ONNX image-encoder
bundles: one for instance similarity and one for concept similarity. Installing the
extra never downloads model weights, creates an index, or activates a generation.

Document graph extras also never download model assets or enable a network
fallback. The host must explicitly construct verified `LocalModelAssets`, select
the accepted model/tokenizer/BGE identities and device, and opt in to each adapter.
See the [document graph pipeline](../guides/document-graph-pipeline.md).
