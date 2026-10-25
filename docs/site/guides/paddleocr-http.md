# Local OCR HTTP service

Expose an installed PaddleOCR profile to external programs using FastAPI. Paths,
network address, port and image limits are selected at startup; no shared client
filesystem is required. The HTTP layer uses the public image OCR provider port,
so applications can also inject another compatible provider factory.

## Install and provision

Use the Python environment in which you will run the service. Install Capicove's
HTTP extra, the Paddle plugin and a platform-compatible Paddle CPU runtime:

```bash
python -m pip install 'capicove[api]' capicove-processor-paddleocr \
  'paddlepaddle>=3.3,<4' 'pypdfium2>=4,<6'
```

For a source checkout, install the checked-out code instead:

```bash
python -m pip install -e '.[api]' -e plugins/processor-paddleocr \
  'paddlepaddle>=3.3,<4' 'pypdfium2>=4,<6'
```

The existing [release deployment script](deployment.md)
can provision an environment with the `ocr-book` and `interfaces` profiles; ensure a compatible Paddle CPU
runtime and PDFium are installed in that same environment before serving. Use a matching core and
plugin version. The `book-balanced` profile also validates PaddleX, pypdf and
PDFium versions, even for image-only requests.

Choose an absolute installation directory for your environment. If the profile
is already installed with a valid receipt, skip the download commands:

```bash
export CAPICOVE_OCR_MODEL_DIRECTORY=/absolute/path/to/paddleocr/book-balanced
capicove models install paddleocr --profile book-balanced \
  --destination "$CAPICOVE_OCR_MODEL_DIRECTORY" --dry-run
# Explicitly authorize this download after reviewing the plan:
capicove models install paddleocr --profile book-balanced \
  --destination "$CAPICOVE_OCR_MODEL_DIRECTORY" --yes
```

Serving checks the installed receipt, model hashes and compatible runtime. It
never downloads missing assets. An arbitrary Paddle cache directory without the
Capicove receipt is insufficient. See [local model provisioning](local-bge-paddle.md).

## Start and call

```bash
capicove models serve paddleocr \
  --model-directory "$CAPICOVE_OCR_MODEL_DIRECTORY" \
  --host 127.0.0.1 --port 8767 \
  --device cpu --max-body-bytes 10485760 --max-pixels 20000000
```

The model warms before Uvicorn installs its signal handlers and reports
readiness, preserving graceful SIGTERM shutdown with the native Paddle runtime. Run the same command with
your service manager or container entrypoint; use its environment's `capicove`
executable. The adapter currently supports CPU. MKL-DNN is off by default;
`--enable-mkldnn` explicitly enables it. The CLI defaults to port 8766 if omitted;
select distinct ports when serving BGE-M3 and OCR together.

```bash
curl --fail http://127.0.0.1:8767/health
curl --fail http://127.0.0.1:8767/v1/ocr \
  -H 'Content-Type: image/png' --data-binary @scan.png
```

Send the image bytes directly, not multipart, JSON, base64, a path or a URL.
Supported Content-Types are `image/png`, `image/jpeg`, `image/webp` and `image/bmp`.
Only single-frame images are accepted. The original bytes are hashed before
image normalization; EXIF orientation is applied during processing.

```python
from pathlib import Path

import httpx

response = httpx.post(
    "http://127.0.0.1:8767/v1/ocr",
    content=Path("scan.png").read_bytes(),
    headers={"Content-Type": "image/png"},
    timeout=120,
)
response.raise_for_status()
result = response.json()
print(result["result"]["full_text"])
print(result["result"]["words"])
```

The response has `result` and `processing_fingerprint`. `result` includes
`source_hash`, `full_text`, `words`, `provider_id`, `provider_version`,
`languages`, `operation_version` and `parameters`. Each Paddle `words` entry is
one recognized **line**, with `text`, `x`, `y`, `width`, `height`, `confidence`
and layout hierarchy. Coordinates are pixels in the EXIF-normalized image;
confidence is between 0 and 1. The fixed multilingual profile uses the existing
`chi_sim`/`eng` language policy; language identifiers do not switch model weights.

Interactive documentation is at `/docs`; the schema is at `/openapi.json`.
The service keeps one provider across requests and releases it on shutdown.
Images remain in memory and are not saved by the service.

## Authentication and remote callers

Default binding is loopback. To allow another machine to connect, explicitly
select a non-loopback address and supply a bearer key through a `CAPICOVE_`
environment variable, populated by your secret manager or shell:

```bash
# CAPICOVE_OCR_API_KEY must already contain your deployment's secret.
capicove models serve paddleocr \
  --model-directory "$CAPICOVE_OCR_MODEL_DIRECTORY" \
  --host 0.0.0.0 --port 8767 --api-key-env CAPICOVE_OCR_API_KEY
```

Clients send `Authorization: Bearer <key>`. Authentication covers every route
except `/health`, including API documentation. You can also enable it on
loopback. Use a trusted TLS reverse proxy for traffic crossing untrusted networks;
the standalone listener provides no tenant isolation or workspace permissions.
Access logging is disabled and inference errors do not echo uploads or internals.

The default upload limit is 10 MiB and decoded limit is 20 million pixels.
`--max-body-bytes` allows up to 100 MiB and `--max-pixels` up to 100 million.
One request is admitted at a time; health remains responsive during inference.
This bounds memory admission rather than cancelling native inference on timeout;
configure upload and response timeouts in your reverse proxy if required.

| HTTP status | Meaning |
| --- | --- |
| 200 | OCR result, including an empty result for an image with no recognized text |
| 400 | Empty, invalid, animated or mismatched image |
| 401 | Missing or incorrect bearer key |
| 413 | Encoded byte or decoded pixel limit exceeded |
| 415 | Unsupported Content-Type |
| 429 | Another upload or OCR inference is running; retry later |
| 500 | Internal inference or evidence failure |
| 503 | Selected model backend unavailable, or service shutting down |

Errors contain `{"error":{"code":"...","message":"..."}}`.

## Embed the app

`capicove.api.ocr.create_app(provider_factory, ...)` accepts a factory implementing
`OcrProvider`: the existing configured public image OCR port plus `close()`.
The factory must verify and warm its assets before returning. `OcrRuntime` owns
the resident provider and releases it exactly once, including startup failures.
Factories used directly by the app run in lifespan off the event loop; the CLI
prewarms the same owner before Uvicorn installs its signal handlers. The
HTTP module has no dependency on PaddleOCR. The packaged CLI supplies the Paddle
composition factory so deployments do not need custom Python glue.

## PDF document parsing

The same `capicove models serve paddleocr` command also enables
`POST /v1/documents/parse`, using the installed `book-balanced` profile and
PP-StructureV3. No additional model download is needed when all three profile
models and their valid receipt are already present. The parser profile version
comes from the installed plugin's declared plan. The document pipeline is
constructed by the existing backend on demand for each invocation.

```bash
curl --fail 'http://127.0.0.1:8767/v1/documents/parse?start_page_index=0&end_page_index=3' \
  -H 'Content-Type: application/pdf' --data-binary @document.pdf
```

Send the original PDF bytes directly. `start_page_index` is zero-based and
inclusive; `end_page_index` is exclusive. The example selects the first three
pages. Omitting both selects the whole PDF, provided it fits the page limit.
Omitting only the end selects from the requested start through the last page.
Unknown or duplicate query parameters and out-of-bounds ranges are rejected.
Authentication uses the same bearer header as image OCR.

```python
from pathlib import Path

import httpx

response = httpx.post(
    "http://127.0.0.1:8767/v1/documents/parse",
    params={"start_page_index": 0, "end_page_index": 1},
    content=Path("document.pdf").read_bytes(),
    headers={"Content-Type": "application/pdf"},
    timeout=300,
)
response.raise_for_status()
result = response.json()["result"]
for page in result["pages"]:
    for block in page["blocks"]:
        print(page["page_index"], block["reading_order"], block["exact_text"])
```

The result uses Capicove's canonical document JSON format:

- `source_sha256` identifies the original uploaded PDF bytes;
  `representation_sha256` identifies the derived representation.
- `pages` preserve original zero-based indices, printed labels, page dimensions,
  status, errors and ordered `blocks`.
- Each block has its ID/type, exact and normalized text, `reading_order`,
  `confidence`, and `box: [left, top, right, bottom]`. These document boxes are
  normalized to `[0, 1]`; image OCR boxes use pixels.
- `outline` preserves the selected pages' embedded PDF outline.
- `receipt` records plugin/profile versions and model, configuration and
  processing hashes; `warnings` remain visible.

HTTP 200 returns parser evidence; inspect each page's status and warnings.
The service checks the returned source hash, page coverage and profile identity.
It stages uploads under a private temporary directory and removes uploads after
success or failure. The parser also uses its existing temporary snapshots. No
Book workspace or persistent output is created.

Configure bounds for your environment:

```bash
capicove models serve paddleocr \
  --model-directory "$CAPICOVE_OCR_MODEL_DIRECTORY" --port 8767 \
  --max-document-bytes 52428800 --max-document-pages 10 \
  --max-document-page-points 2000
```

`--max-document-pages` bounds selected pages, allowing large documents to be
processed in explicit ranges. `--max-document-page-points` limits both media
and crop box side lengths before and after applying UserUnit. Scaled dimensions are physical PDF points
(72 per inch); bounding raw dimensions also protects renderers that ignore
UserUnit. A small UserUnit cannot make an oversized raw page acceptable.
The defaults accept standard A4/Letter pages. Maximum allowed configuration is
100 MiB, 1000 selected pages and 14400 points per side. Image `--max-pixels` is a
separate limit and does not control PDF rendering. Larger PDF geometry limits
increase native rendering memory needs.

Optionally set `--document-temp-directory /absolute/existing/directory` to choose
the upload staging parent. By default, the OS temporary directory is used. The
parent must exist; the service creates private subdirectories under it. Ensure
sufficient temporary disk space for uploads and parser snapshots. Image and
PDF requests share a single admission slot; a concurrent request gets 429, while
health remains responsive. These limits do not interrupt native inference or
isolate it in another process.

The current profile accepts PDFs with a reliable nonblank text layer and no
raster images on selected pages, within its supported scripts. Scanned PDFs,
image-bearing pages and unsupported-script pages remain unsupported. Table,
formula, chart and seal recognition are disabled in this profile. The new route
preserves these backend checks; it does not turn an image-only scan into a
supported document input.

PDF errors use the same envelope as image OCR: 400 invalid/encrypted PDF or
page range, 401 authentication, 413 byte/page/geometry limit, 415 Content-Type,
422 explicit parser rejection, 429 busy, 500 internal parser/staging/evidence
failure, and 503 unavailable model assets/runtime or shutdown. Only a typed
`DocumentRejectedError` becomes 422. An unknown backend `ValueError` is a server
failure, not evidence of bad input. Paddle wraps model verification and runtime
availability failures in `DocumentBackendUnavailableError`. Raw exception text
and private paths are never included in responses.

For embedding, pass a `capicove.api.documents.DocumentParserBinding` as
`document_parser` to `create_app`. The binding uses the public
`DocumentParserBackend` protocol plus profile identity and optional language
hints. Omitting it leaves the document route unavailable. `/health` reports the
configured document parser ID, and `/docs` describes the PDF upload and range
parameters. Other compatible parsers can use the same transport without Paddle
imports in the HTTP module.


## Runtime and extension contracts

`OcrHttpSettings` centralizes image/PDF limits, staging configuration and language
policy for CLI and Python composition. Existing `create_app` keyword arguments
remain supported; use either those arguments or `settings`, without mixing
nondefault legacy limits with a settings object.

```python
from capicove.api.ocr import OcrHttpSettings, create_app

# provider_factory is your verified, warmed public OCR provider factory.
application = create_app(
    provider_factory,
    settings=OcrHttpSettings(max_body_bytes=5 * 1024 * 1024, max_document_pages=5),
)
```

The HTTP routers only translate uploads, parameters and results. Image acceptance
uses `ImageOcr.recognize_upload` and the existing image preparation capability,
without compression or source-byte mutation. `DocumentParsingService` owns PDF
inspection, processing budgets, backend dispatch and source/page/profile result
checks. A `DocumentParserBinding` remains importable from `capicove.api.documents`
and is now defined in `capicove.documents`. It borrows the public document backend;
its creator owns any persistent backend resources. Paddle's backend continues to
own and dispose its per-invocation pipeline and snapshots.

New document backends should raise `DocumentRejectedError` only for explicitly
unsupported documents and `DocumentBackendUnavailableError` for unavailable assets
or runtimes. Other failures become internal processing errors. Document result
serialization preserves the canonical codec, including normalized box arrays.
Both successful results and errors have response schemas in OpenAPI and runtime
response validation; unknown or invalid wire fields fail closed.

Every HTTP response carries a server-generated `X-Request-ID`. Internal failure
logs contain that ID, the stable error code and exception type; they omit request
content, credentials, exception messages and private paths. Use the ID to correlate
an HTTP failure with operator diagnostics.

Client disconnects clean up partial uploads and release admission. If a request
is cancelled while native processing is running, the service drains that work
before deleting its upload or releasing its slot. Shutdown stops admission and
drains work before closing the resident provider. Cancellation does not forcibly
interrupt native inference; configure process supervision separately when needed.
