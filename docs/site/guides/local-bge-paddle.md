# Local BGE-M3 and PaddleOCR

After provisioning, use the [model factory](model-factory.md) to select profiles
and share bounded typed leases across local retrieval, graph OCR/PDF preparation,
Book and video OCR. Python imports now use `capicove.model`; the operator removed
the legacy `capicove.model_management` package.

This composition uses BGE-M3 for dense text embedding and the existing Paddle
three-model profile for PDF, Book, graph input preparation, and video OCR. CLIP,
local ASR and local LLM assets are selected separately; see the
[complete runtime inventory](../getting-started/runtime-profiles.md).
External programs can use the
[OpenAI-compatible BGE-M3 HTTP service](bge-m3-http.md) and
[PaddleOCR HTTP service](paddleocr-http.md) after installation.
SigLIP2 remains a separate visual
embedding space; its vectors cannot be mixed with BGE-M3 vectors.

## Model inventory

| Model | Purpose | Pinned revision | CPU |
| --- | --- | --- | --- |
| `BAAI/bge-m3` | 1024-dimensional dense text embedding | `5617a9f61b028005a4858fdac845db406aefb181` | Batch 1, short inputs |
| `PaddlePaddle/PP-DocLayout-L` | PDF layout | `ca760c3336d922ee63cdbff85003181222e3911c` | Supported |
| `PaddlePaddle/PP-OCRv5_server_det` | Text detection | `ca867c897ecbca8873081573a802ad70d499cb94` | Supported |
| `PaddlePaddle/PP-OCRv5_server_rec` | Multilingual recognition | `b26c3587fda8da3c8ec0ce357214b4d661ff1558` | Supported |

The BGE download allowlist is packaged in `bge_m3.lock.toml`. It excludes ONNX
duplicates, sparse/ColBERT heads and SentencePiece weights; the verified fast
tokenizer uses `tokenizer.json`. The operator provisioner checks each upstream
SHA-256 before converting the official PyTorch state dictionary into safetensors.
Runtime never loads `.bin` weights. The receipt records the upstream MIT license,
revision, conversion versions and the derived closed inventory. Paddle acquisition
uses the existing immutable `book_balanced_models.toml` inventory and receipt.

## Isolated CPU setup

Run from the source checkout. Use separate environments so native model stacks
do not change the project's development environment:

```bash
uv venv --python 3.11 /tmp/capicove-bge-m3-env
uv pip install --python /tmp/capicove-bge-m3-env/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
uv pip install --python /tmp/capicove-bge-m3-env/bin/python \
  -e . 'sentence-transformers>=5.6,<5.7' 'transformers>=4.48,<6' \
  'huggingface-hub>=1,<2' safetensors

uv venv --python 3.11 /tmp/capicove-paddle-env
uv pip install --python /tmp/capicove-paddle-env/bin/python \
  -e . -e plugins/processor-paddleocr \
  'paddlepaddle>=3.3,<4' 'pypdfium2>=4,<6' 'av>=14,<19'
```

Choose an absolute model directory outside the source repository. Acquisition is
explicit and network-enabled; inference uses already verified local files.

```bash
export CAPICOVE_MODEL_HOME="$HOME/.local/share/capicove/models"
PATH=/tmp/capicove-bge-m3-env/bin:$PATH \
  /tmp/capicove-bge-m3-env/bin/python scripts/download_bge_m3.py \
  --destination "$CAPICOVE_MODEL_HOME" --confirm-download

PATH=/tmp/capicove-paddle-env/bin:$PATH \
  /tmp/capicove-paddle-env/bin/capicove models install \
  paddleocr --profile book-balanced \
  --destination "$CAPICOVE_MODEL_HOME/paddleocr/book-balanced" --yes
```

Omitting `--confirm-download` from the BGE script prints the pinned plan. Existing
BGE destinations are refused; provision a fresh directory for a new installation.
Retain `bge-m3.receipt.json` next to the `bge-m3` directory and copy its verified
inventory digest into startup configuration. Do not trust a replacement receipt
by automatically accepting a new digest from it.

After acquisition, run inference offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PADDLE_PDX_CACHE_HOME=/tmp/capicove-paddle-cache
```

Use `OMP_NUM_THREADS=1` for Paddle CLI runs; the configured Paddle CPU worker
thread count is two. BGE qualification used four Torch threads.

## Retrieval and graph composition

```python
from pathlib import Path

from capicove.retrieval import TextEmbeddingService
from capicove.retrieval.bge import BgeTextBatchEncoder, load_bge_m3_profile

profile = load_bge_m3_profile(
    Path("/absolute/model-home/bge-m3"),
    expected_manifest_sha256="YOUR_VERIFIED_INVENTORY_SHA256",
    max_tokens=512,
)
service = TextEmbeddingService(
    BgeTextBatchEncoder(profile, device="cpu", batch_size=1), space=profile.space
)
documents = service.encode(
    ("\u592a\u9633\u80fd\u7535\u6c60\u677f\u5c06\u9633\u5149\u8f6c\u5316\u4e3a\u7535\u80fd\u3002",),
    space=profile.space,
    role="document",
)
queries = service.encode(
    ("How can sunlight produce electricity?",), space=profile.space, role="query"
)
```

BGE-M3 uses empty query/document prefixes, normalized CLS vectors, and no silent
truncation. Inputs above the configured token limit are rejected. The existing
graph `RetrievalBatchTextEmbeddingProvider` can delegate to this service.

`DocumentGraphConfiguration(model_root=Path(...))` defaults to the Paddle parser
and image OCR provider. `prepare_pdf(request, document_ref=..., scope=...)` accepts
an already authorized `DocumentParseRequest` and returns the complete parser
result plus graph text. `prepare_image(source, document_ref=..., scope=...)`
prepares image text. Retain the original evidence and PDF parse result for boxes;
graph character spans address derived text. Source hashes, pages, parser
representation and processing identities remain bound to the graph input.
This preparation does not run graph entity extraction without a separately
selected generation provider.

## Book and video OCR

```bash
/tmp/capicove-paddle-env/bin/capicove book prepare \
  /absolute/book-workspace /absolute/document.pdf \
  --model-root "$CAPICOVE_MODEL_HOME/paddleocr/book-balanced" \
  --device cpu --evidence-only

/tmp/capicove-paddle-env/bin/capicove video filter \
  --input /absolute/video.mp4 --criteria /absolute/criteria.json \
  --ocr-provider paddleocr \
  --ocr-model-root "$CAPICOVE_MODEL_HOME/paddleocr/book-balanced" --output json
```

An OCR-area criterion can be `[{"type":"ocr_area","criterion_id":"ocr",
"min_area_ratio":0.01}]`. Its metric is the union area of recognized line boxes;
it is not the old EasyOCR detection-only metric. BGR filtering frames are converted
correctly before image OCR. Video retrieval bootstrap also defaults to Paddle:
add `--ocr-model-root` when using `scripts/run_video_retrieval.py --build-video`.
`--ocr-provider none` explicitly disables OCR. These settings are standalone
composition options; shared-host `--runtime-config` still supports only metadata
and motion video filters.

The CPU image provider uses two threads, recognition batch 1, and detection with
**maximum long edge 736**. It disables orientation and unwarping models. CPU
composition defaults `enable_mkldnn=False`: the first real run on Paddle 3.3.1 hit
a oneDNN attribute-conversion error. The choice is configurable through
`ImageOcrConfiguration` and parser plugin configuration, and enters processing
identity. The three-model profile and its existing PDF restrictions are unchanged:
book-balanced requires a reliable text layer and rejects raster-image PDF pages.

## Historical device qualification

The following measurements predate the 0.1.0 security upgrade. Current BGE
installation requires 5.6.x; see the [release procedure](../development/releasing.md)
for migration and requalification.

The device has four CPU threads, about 7.75 GiB RAM, no swap, no usable GPU, and
ample disk space. Run model stages sequentially. Model files, source fixtures,
generated indexes, workspaces and machine reports stay outside Git.

The offline BGE-M3 smoke used four authored documents and eight English/Chinese
queries, including cross-language queries. All eight expected Top-1 matches passed.
The graph embedding bridge and overlength rejection passed. Peak process RSS was
2060.32 MiB; first document batch including load/verification took 49.67 seconds,
and the eight-query batch took 11.87 seconds. Runtime: Python 3.11.15, CPU Torch
2.13.0, Sentence Transformers 5.2.3, Transformers 5.19.0, safetensors 0.8.0.
The derived inventory digest is
`0b10b750f2b71f73e8dee5d8c1940c20c051cd67fbcc71b9b79d8ac366b237d3`.

The offline Paddle smoke passed authored image recognition, graph image source
hash preservation, a one-page portrait PDF through graph preparation and Book
evidence-only preparation, actual FFmpeg/PyAV video decoding with an OCR text
unit, and standalone OCR-area filtering. The image's two lines were recognized
as `Capicove retrieval test` and `Paddle OCR reads this page`. Video line-box
union area was approximately 0.1750. Image recognition including initialization
took 18.31 seconds; PDF graph preparation took 41.30 seconds; video preparation
took 12.60 seconds. Peak process RSS across sequential stages was 3377.36 MiB.
Runtime: PaddleOCR 3.7.0, PaddlePaddle 3.3.1, PaddleX 3.7.2, PyAV 18.1.0.

Video preparation used an explicitly authored visual encoder stub to isolate
real decoding and OCR. It did not qualify a new SigLIP2/Elasticsearch search run.
A sparse landscape PDF fixture was rejected because recognized lines lacked
layout coverage; the normal portrait fixture passed. That strict validation was
retained. These samples do not establish general PDF acceptance or OCR accuracy.

On this device the verified installations are
`/absolute/model-home/bge-m3` and
`/absolute/model-home/paddleocr/book-balanced`. Paddle model cards
declare Apache-2.0; local `paddleocr/provenance.json` records their pinned sources,
file hashes, model format and runtime versions. Raw fixtures and reports remain
in `/tmp/capicove-bge-paddle-qualification`, outside the source repository.

The final core gate passed **5,941 tests, 25 skips, 85.30% coverage**.
`make quality`, `make docs-check` and `make build` passed; the optional Paddle
plugin's separate unit suite passed **90 tests**. Installed core and plugin
wheels were checked outside the source checkout against the durable model roots.
Real wheel inference passed a BGE cross-language Top-1 query, Book evidence-only
preparation of the authored portrait PDF, and video OCR-area filtering through
the CLI. Book preparation emitted one section and no summary/LLM tasks; video
filtering again reported line-box area approximately 0.1750.

These are small integration samples, not general retrieval/OCR benchmarks, native
cross-platform qualification, or a release approval. Sparse/ColBERT retrieval,
Chinese OCR accuracy, scanned PDFs, real graph LLM extraction, Elasticsearch video
search, GPU execution, and native macOS/Windows qualification remain separate work.
