# Explicit models and service profiles

The core package runs without a model server. SQLite workspace search, local Wiki,
evidence storage, traces and deterministic processing do not require a cloud API
key. Installing an extra installs Python dependencies; it does not provision model
weights, native executables, credentials or services.

This inventory describes implemented capabilities and their deployment inputs.
An unqualified row means runtime evidence is missing, not that its feature is a
future roadmap item. Qualification below is Linux x86-64 evidence from 2026-10-19;
it does not establish Windows/macOS compatibility or production model accuracy.

## Model inventory

| Capability | Explicit asset selection | Configuration entry point | Local acceptance |
| --- | --- | --- | --- |
| Dense text retrieval and graph embeddings | `BAAI/bge-m3`; tokenizer and closed safetensors inventory | Model factory `bge`, root, trusted manifest digest, device, token limit | Real inference; 6/6 expected Top-1 results |
| Joint image/text embeddings and video search | `google/siglip2-base-patch16-224` | Model factory `siglip2`, or explicitly started local model API | Real image/text inference and Elasticsearch video integration |
| Image/video NSFW scores | `Falconsai/nsfw_image_detection` | Verified image-model installation or video criterion model root/digest | Real image and video inference |
| Image/video watermark scores | `prithivMLmods/Watermark-Detection-SigLIP2` | Verified installation or criterion model root/digest | Real inference; threshold quality still needs application data |
| Image/video aesthetics | `cafeai/cafe_aesthetic` | Explicit restricted-model selection | Real inference; packaged manifest records AGPL-3.0; exclude from an automatic default bundle |
| Image pairs and frame/text similarity | `openai/clip-vit-base-patch32` | Verified installation or criterion model root/digest | Real inference |
| Image/text matching | `Salesforce/blip-itm-base-coco` | Verified installation | Real inference |
| Region captions | `Salesforce/blip-image-captioning-base` | Verified installation | Real inference |
| Image difference descriptions | `HuggingFaceTB/SmolVLM-256M-Instruct` plus region-caption model | Two verified installations | Real inference; completion is not a semantic-quality guarantee |
| Paddle OCR, PDF, Book and video OCR | `PP-DocLayout-L`, `PP-OCRv5_server_det`, `PP-OCRv5_server_rec` | `paddleocr` installer, `book-balanced` version 2; factory `paddleocr` / `paddle-document` | Actual OCR, Book preparation and video OCR passed |
| Audio transcription | Multilingual Whisper tiny GGML plus Silero VAD v6.2.0 GGML, both revision/hash pinned | `--model`, `--vad-model`, executable paths, or audio host configuration | 7 installed-wheel CPU acceptance cases passed: speech, both silence paths, corrupt assets and CLI stream isolation |
| Reverse-image instance and concept retrieval | Two verified ONNX image-encoder bundles with distinct embedding spaces | Bundle paths, trusted manifest digests, preprocessing and CPU thread limits | Verified local CLIP and SigLIP2 ONNX exports; query/target parity, build/activation/search and CLI/MCP reopening passed |
| ONNX image target embeddings | Verified ONNX bundle and tensor/preprocessing contract | Local target encoder construction | Real CLIP/SigLIP2 ONNX batch and query/target parity passed |
| Background removal | A selected rembg model and already provisioned session | Inject session plus model identity into `RembgBackgroundRemovalProvider` | No selected weights/session available |
| Tesseract OCR | Native executable and selected language traineddata | `TesseractCliOcrProvider`, languages and executable path | Native language assets not provisioned in this run |
| EasyOCR text-area detection | Local CRAFT detector weights; recognition is disabled | Existing EasyOCR model cache or explicitly injected detector | Offline construction regression tested; real detector weights absent |
| RAM++ video tags | RAM++ checkpoint, vocabulary and compatible `ram` runtime | Explicit local weights/tag vocabulary, with local BERT tokenizer vocab.txt beside the tag vocabulary | No qualified checkpoint/runtime selected |
| Document graph generation | User-hosted compatible chat server or local causal LLM; exact local tokenizer and accepted generation profile; BGE for embeddings | Verified `LocalModelAssets`, provider/tokenizer identities, context/output budgets | Real user-hosted compatible LLM extraction, receipt replay, authorization revocation and invalid-credential rejection passed |
| Book semantic judgments | User-selected Agent/model outside the deterministic preparation pipeline | Submit canonical chunk results through the Book workflow | Workflow tested; external Agent/model quality is deployment-specific |

The eight image-model identities, upstream licenses and immutable revisions are
already pinned in `src/capicove/model/image_models.toml`. BGE and Paddle have
separate closed allowlists. Reuse those manifests instead of maintaining a second
list of floating model names. An image model marked `default` in a manifest is not
an instruction to download it at runtime.

Whisper tiny multilingual passed functional acceptance on an English speech
fixture, silence and adversarial inputs. This is not a claim that it meets
Chinese/English production transcription quality. Choose the production
model using representative audio. The model and VAD hashes belong in the local
receipt and inference lineage.

## Native runtimes and services

| Component | Required configuration | Installation boundary | Qualification |
| --- | --- | --- | --- |
| FFmpeg / FFprobe | Absolute executable paths; matching build with the required `fd:` input support | Native executable installation; not a pip extra | FFmpeg 7.0.2 used for local media acceptance |
| Whisper.cpp | `whisper-cli` and `whisper-vad-speech-segments`, compatible models | Native build/install, CPU enabled explicitly | Tag `v1.9.5`, fixed source commit; executable reports `1.9.5-dev`; real CPU inference qualified with tiny and Silero v6.2.0 |
| Elasticsearch | HTTPS hosts with explicit ports, trusted CA, API key, index prefix/alias and timeout | Separate service; `capicove-retriever-elasticsearch` installs only the Python plugin | Server 9.5.5 with client 8.19.3, TLS and API-key authentication |
| OpenSearch | Endpoint, TLS/authentication policy, index prefix/alias | Alternative separate service and `capicove-retriever-opensearch` | Server 3.6.0 native BM25/k-NN with TLS and scoped authentication; invalid credentials and untrusted CA rejected |
| FAISS | Installed native library; local generation directory and vector-space identity | `capicove-retriever-faiss`; no daemon | Persistence and 10,000-vector recall, plus video integration |
| SQLite FTS5 | Workspace and evidence paths | Embedded runtime; no separate service | Workspace search and Wiki acceptance |
| PaddlePaddle | Platform-compatible CPU/GPU wheel and supported PaddleOCR/PaddleX versions | Explicit native Python runtime plus separate Paddle plugin | CPU stack qualified; no claim of a combined Paddle/Torch GPU environment |
| Spark | Java 17 executable, PySpark and local resource budgets | `capicove-execution-spark`; no remote cluster required for the supported local profile | 55 actual integration cases; external dataset case not selected |
| Ray image deduplication | Optional Ray runtime and resource limits | `image-ray`; cluster deployment is optional | Real local Ray 2.57.0 workers matched serial deduplication; multi-node clusters were not qualified |
| SigLIP2 HTTP service | Local model root/digest, device, loopback host/port | Optional `model-serving`; direct factory use avoids this daemon | Actual model endpoint used in Elasticsearch acceptance |
| Capicove HTTP/MCP | Explicit host configuration, allowed roots/capabilities, HTTP authentication where configured | `api` / `mcp`; operator-started process | Existing transport acceptance; no mandatory hosted LLM service |

Elasticsearch's official [client compatibility table](https://www.elastic.co/docs/reference/elasticsearch/clients/python#compatibility)
allows an 8.x client with a 9.x server in compatibility mode. The measured
combination above is the release evidence; other versions still need qualification.
The plugin now pins HNSW parameters in the mapping rather than inheriting
server-dependent defaults. Rebuild pre-fix vector-bearing generations; do not
rewrite their manifests or relax integrity checks. Text-only mapping identities
are unchanged.

## Selectable delivery profiles

The [deployment workflow](../guides/deployment.md) implements a small selector over
existing extras and plugins, a hash-locked offline wheel bundle, private workspace
configuration and post-install acceptance. Native/model acquisition remains a
separate explicit choice; selecting a profile alone does not provision those assets.

| Delivery selection | Dependencies / external inputs | Decision needed |
| --- | --- | --- |
| `core`, `interfaces` | Core, optional Wiki/API/MCP, SQLite storage | Local paths and enabled transports |
| `media` | Video Python dependencies and FFmpeg/FFprobe | Native platform/build source |
| `audio-asr` | Whisper.cpp plus multilingual ASR and VAD assets | Exact model tier, source, revision, hashes and download authorization |
| `semantic-text` | Text embedding dependencies and BGE-M3 | CPU/GPU and asset location |
| `semantic-vision` | SigLIP2; optional image-model selections | Select tasks/models individually; aesthetics requires a separate license decision |
| `ocr-book` | Paddle plugin/runtime and the shared three-model profile | CPU/GPU; current PDF profile excludes scan-only, raster-containing and hybrid pages |
| `elasticsearch`, `opensearch` | One of Elasticsearch or OpenSearch and its plugin | Existing service versus local service; CA/authentication/storage ownership |
| `faiss` | FAISS | Capacity requirement and index storage |
| `graph-generation`, `local-generation` | Compatible remote provider or local LLM, tokenizer and graph extras | Endpoint/model identity, template, budgets and acceptance corpus |
| Advanced adapters | ONNX bundles, rembg, Tesseract, EasyOCR, RAM++, Ray or Spark | Select only the adapter needed; qualify its assets/runtime separately |

Use the existing [model factory](../guides/model-factory.md) for BGE, SigLIP2 and
Paddle bindings. Audio, ONNX, rembg and other adapters currently have their own
explicit configuration; do not write unsupported factory provider names into
`models.toml`. The delivery installer generates the existing audio/video host formats when a
verified native configuration is supplied. Other provider configurations retain
their documented explicit entry points.

The delivery process uses these boundaries:

1. Show a closed plan: packages, native versions, exact files, immutable revisions,
   expected bytes/hashes, licenses, destination and required service resources.
2. Obtain the download/service choice once, then install into a dedicated
   environment. CPU and GPU are distinct selections; a failed GPU installation
   must not silently change the model or execution device.
3. Verify allowlisted files and write a provenance receipt outside the source tree.
   Select an authorized upstream or private repository explicitly; confirm its
   license, immutable revision, file inventory and redistribution rights.
4. Generate local configuration with trusted digests and secret references.
   Credentials must not be embedded in a committed example or command argument.
5. Run a small **real inference/query** acceptance and a missing-asset/authentication
   rejection case. Report `ready`, `missing_configuration`, `unqualified` or
   `failed` per selected capability. Import success alone is not readiness.

Reuse [operator installation plans](../guides/operator-jobs.md) and the existing
Paddle installer where supported. The BGE recipe remains an explicit provisioner.
EasyOCR now disables downloads when constructing its runtime Reader; provision
its detector cache separately. This follows its documented
[`download_enabled` setting](https://www.jaided.ai/easyocr/documentation/).

## Repeat the added runtime acceptance

Use an isolated, disposable Elasticsearch node with a private TLS CA. The tests
create `capicove-*` indexes and aliases. Install the plugin, FAISS and model-serving
dependencies, start the verified local SigLIP2 endpoint, and set these variables
through the local environment or secret manager:

```bash
export CAPICOVE_TEST_REAL_ELASTICSEARCH=1
export CAPICOVE_VIDEO_ES_ENDPOINT=https://127.0.0.1:9200
export CAPICOVE_VIDEO_ES_CA=/absolute/qualification-ca.crt
# Supply CAPICOVE_TEST_ES_API_KEY without putting it in shell history.
export CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT=http://127.0.0.1:8000
python -m pytest -c plugins/retriever-elasticsearch/pyproject.toml \
  plugins/retriever-elasticsearch/tests/integration -q
```

The API key needs access to the disposable `capicove-*` indexes/aliases. The
untrusted-CA test deliberately uses the public trust store, so the test node must
use a private CA. Stop the test services after acceptance.

Whisper acceptance reads an operator-owned JSON file selected by
`CAPICOVE_WHISPER_QUALIFICATION`. It contains `backend_version`, an English speech
fixture's `speech_keywords`, and `files` entries named `model`, `vad_model`,
`speech`, `whisper_cli`, `whisper_vad_cli`, `ffmpeg`, and `ffprobe`. Each entry has an
absolute `path` and trusted `sha256`; the test rehashes every file before use.

```bash
CAPICOVE_WHISPER_QUALIFICATION=/absolute/whisper-qualification.json \
  python -m pytest tests/model/test_local_whisper_cpp.py -q --no-cov
```

These seven cases run fixed-language and automatic-language speech recognition,
both silence VAD paths, timestamp/lineage checks, rejection of truncated ASR/VAD
weights, and actual CLI multistream isolation with ambiguity/out-of-range
rejection. No weights are downloaded by the tests. Without configuration they
report skips, which must not be counted as successful runtime qualification.
