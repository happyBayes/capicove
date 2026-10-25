# RFC-0027: Provider-independent local image OCR HTTP service

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- Authorization: The operator requested a reusable FastAPI service for installed
  PaddleOCR models across deployment environments. This authorizes local
  implementation, not independent review, publication or release.
- Decision: [ADR-0031](../adr/0031-local-ocr-http.md).

## Contract

`capicove models serve paddleocr --model-directory ABSOLUTE_PATH` starts a
standalone FastAPI service over the public image OCR provider. The operator
selects an existing installation produced by `capicove models install paddleocr
--profile book-balanced --destination PATH`. The plugin verifies the packaged
manifest, installation receipt, file hashes and compatible runtime versions.
Serving does not download models. No host-specific paths or virtual environments
are embedded in the implementation.

`POST /v1/ocr` takes one binary PNG, JPEG, WebP or BMP image as the request body,
with its matching image Content-Type. It accepts no filesystem paths, remote
URLs, runtime flags or model overrides. The JSON response contains `result`
(the existing public `ImageOcrResult`) and `processing_fingerprint` (the startup
provider fingerprint SHA-256). The result preserves original-byte SHA-256,
recognized text, ordered line boxes and confidence, provider identity, languages
and operation metadata. Paddle lines remain lines in the public `words` array;
word segmentation is not fabricated. PNG/JPEG EXIF normalization remains owned
by the image capability. Animated images and format mismatches are rejected.

`GET /health` reports readiness, provider ID/version and processing fingerprint.
`/docs` and `/openapi.json` describe the upload and response contracts. The CLI verifies and warms the provider before Uvicorn installs its signal
handlers, so native Paddle handlers cannot bypass graceful shutdown. The app
claims it during lifespan startup before readiness; shutdown asks the shared runtime owner to release it. The CLI also asks that
owner to clean up if listener startup fails; the underlying provider closes
exactly once. The transport accepts a provider factory conforming to
existing public OCR ports, enabling reuse without Paddle imports in the API.
Provider construction stays in the CLI composition root.

## Resource and security boundaries

Default to loopback, CPU, one worker, one admitted upload/inference at a time,
10 MiB encoded bytes and 20 million decoded pixels. Byte and pixel limits are
operator configurable up to 100 MiB and 100 million pixels. Concurrent requests
receive 429; health remains responsive because decode and inference run in a
thread. Read upload bodies incrementally with a size limit, including chunked
requests. Do not persist uploads, log request bodies or enable access logging.

Non-loopback binding requires an explicit `--api-key-env CAPICOVE_...` bearer
credential; the same option can secure loopback. Every route except health then
requires authentication. Starting the listener is the explicit network opt-in.
This is a single-owner service without TLS, tenant isolation or workspace Tool
permissions. Use a trusted reverse proxy for TLS and request timeouts when
needed. The underlying public adapter currently supports CPU only.

Errors return `error.code` and `error.message`: 400 invalid image, 401 invalid
key, 413 size limit, 415 unsupported Content-Type, 429 busy, 500 inference failed.
Inference exceptions do not expose uploaded content or runtime details. There
are no URL fetches or server-side path reads controlled by requests.

## Validation

Test injected providers, original hashes and structured output, startup failure
and cleanup, malformed/oversized images, media types, authentication, concurrent
admission, health responsiveness, optional import isolation and CLI dispatch.
Qualify verified local models when available. Run repository quality, test and
build gates; distinguish existing failures and real inference qualification from
provider-double contract tests.


The operator-requested service refactoring keeps `create_app` compatible and
introduces `OcrHttpSettings` for shared configuration. Image upload acceptance
uses the public `ImageOcr.recognize_upload` operation and existing preparation
logic in single-frame, no-compression mode. Pixel and invocation budgets and
source hashes remain bound to the original upload. HTTP code does not maintain
its own Pillow validation or error list. Normal responses pass through typed
FastAPI response models. Error schemas, mapping and sanitized request diagnostics
belong to the application rather than a single-function utility module.
