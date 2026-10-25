# RFC-0028: Local document parsing HTTP interface

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- Authorization: The operator explicitly requested document parsing over HTTP
  after the installed PaddleOCR image service. This authorizes implementation,
  not independent maintainer review, publication or release.
- Decision: [ADR-0032](../adr/0032-document-parsing-http.md).

## Public contract

Extend `capicove models serve paddleocr` with `POST /v1/documents/parse` on the
same FastAPI listener. Accept one binary `application/pdf` upload. Optional
`start_page_index` (default zero) and `end_page_index` (default page count) select
a zero-based half-open range. Reject unknown, duplicate, noninteger, reversed
and out-of-bounds query parameters. The caller cannot choose filesystem paths,
URLs, models, profile versions, devices or pipeline settings.

Return `{"result": ...}` with the existing canonical document JSON codec,
including source and representation hashes, pages, block types, exact and
normalized text, normalized boxes, reading order, confidence, printed page
labels, outline, warnings and parser receipt. Page indices remain original PDF
indices. The receipt binds plugin/profile versions, models, configuration and
processing. Check result source hash, page coverage and profile against the
request before serialization. Preserve canonical per-page status and warnings;
HTTP 200 is not a claim that every page succeeded.

The HTTP module consumes the public `DocumentParserBackend` port in an immutable
startup binding. The CLI selects the Paddle parser through public plugin discovery
and derives the profile version from its installation plan. Image OCR and PDF
parsing share authentication and a single admission lock. Backend construction
stays outside canonical contracts and transport translation. No workspace,
index, Book job or durable artifact is created.

## Bounds and private staging

Stream the uploaded bytes into a server-chosen private temporary directory;
never use client filenames. Hash the received bytes. Keep the private source
alive through metadata inspection, parsing and serialization, then delete it on
success, rejection, error or cancellation. The underlying backend can create
its own private snapshots; those remain subject to its existing cleanup.

Default limits are 50 MiB per PDF, ten selected pages per request and 2000 PDF
physical points per page side (72 points/inch, accounting for UserUnit). Operator
flags allow up to 100 MiB, 1000 selected pages and 14400 points. Check both media
and crop boxes, bounding raw side lengths as well as UserUnit-scaled physical
lengths so renderers that ignore UserUnit cannot bypass the limit. Reject
encrypted PDFs and invalid page geometry before invoking the backend. A temporary parent is an optional startup-selected existing absolute
unlinked directory; otherwise use the operating system's temporary location.
Byte and page limits are admission bounds, not native execution timeouts or
process isolation. Decode/parse and disk writes run off the event loop.

Authentication, loopback defaults, explicit remote binding, CPU-only image
adapter and no automatic model acquisition remain as in RFC-0027. PDF and image
inference cannot overlap; a competing call receives 429. Health remains
responsive. Native Paddle imports precede Uvicorn signal-handler installation.
The existing document backend constructs its PP-StructureV3 pipeline on demand;
its scope and pipeline lifetime do not change in this transport addition.

## Errors and supported documents

Errors use a typed `error.code`/`error.message` envelope. Return 400 for an
invalid/encrypted PDF or invalid page bounds, 401 for authentication, 413 for
byte/page/geometry bounds, 415 for Content-Type, 422 only for an explicit
`DocumentRejectedError`, 429 for busy, 500 for unexpected parser/staging/evidence
failures and 503 for `DocumentBackendUnavailableError` or shutdown. Unknown
backend `ValueError` exceptions are processing failures. Paddle classifies model
verification and runtime availability errors explicitly at the provider boundary.
No message matching is used to distinguish input and infrastructure failures.
Every response carries a generated request ID; internal logs include only that
ID, the error code and exception type, excluding exception text and payloads.

The existing `book-balanced` backend accepts reliable nonblank text-layer PDFs
with no raster images on selected pages, within its supported scripts. Scanned,
image-bearing or unsupported-script pages remain rejected; this RFC does not
weaken those checks or add table/formula/chart recognition. The parser can run
OCR and layout analysis on a supported text-layer page while preserving evidence.

## Validation

Test binary and chunked uploads, bounded page selection, encryption, geometry,
canonical codec round trips, original hashes, profile binding, mismatched parser
results, shared admission, authentication, safe failures and staging cleanup.
Exercise a real installed PP-StructureV3 profile using a synthetic text-layer PDF
through the installed wheel and verify normal shutdown. Run repository quality,
test, build and documentation gates, reporting unrelated existing failures.


## Service refinement

The operator requested a production-oriented responsibility and lifecycle
refactoring. `capicove.documents` now owns `DocumentParserBinding`,
`DocumentLimits`, `DocumentParsingService` and typed input, rejection, processing,
result and backend-availability errors. Optional PDF imports remain lazy and no
canonical record or codec format changes. The public backend protocol documents
explicit errors; older backends returning untyped errors fail as server errors.

`api.documents` contains typed HTTP response models, query translation and private
staging only. A router and FastAPI response validation replace handwritten success
responses. The old binding import remains an alias. `api.ocr.create_app` remains
the application factory and shared HTTP exception boundary. `OcrHttpSettings`
owns transport configuration; the single-function `api.model_http` module is
removed. `runtime.ocr_service.OcrRuntime` owns resident provider lifetime,
admission and cancellation-safe native execution without HTTP dependencies.
The CLI prewarms that owner before Uvicorn signal installation. Repeated cleanup
requests reach the underlying provider exactly once.

Worker cancellation must drain native work before releasing admission or staging.
All staging operations, inspection and parsing run off the event loop. A direct
asyncio cancellation is delayed until the worker finishes; partial upload
disconnects still clean up immediately when no native worker is active. These
rules bound resource ownership, not native inference duration.
