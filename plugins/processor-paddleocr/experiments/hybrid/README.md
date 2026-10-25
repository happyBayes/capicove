# Bounded local hybrid PDF experiment

Related: issue #38, PR #24, and the draft hybrid-routing RFC.

This is an opt-in, macOS-only qualification runner. It emits the existing public
`DocumentParseResult` plus external diagnostic evidence for one authorized page.
It does not register a production plugin profile or change `book-balanced` v2.
The runner depends on the accompanying PR #24 qualification candidate's parser
helpers and offline runtime manifest. It is not a standalone wheel.

## Reproduce locally

Use the already provisioned OCR environment (Python 3.12, PaddleOCR 3.7.0,
PaddleX 3.7.2, PaddlePaddle 3.3.1, pypdfium2 5.13.0, Pillow 12.3.0),
pypdf, scipy, NumPy, macOS Vision, a local Swift SDK, and Tesseract 5.5.3 with
English data. Core installation does not install these optional tools.

From this directory, configure existing models into a **new external directory**:

```sh
python configure.py --en-model /local/models/en --el-model /local/models/el \
  --server-model /local/models/server --output /local/new-hybrid-runtime
export CAPICOVE_HYBRID_CONFIGURATION=/local/new-hybrid-runtime/configuration.json
python -I run_case.py --source /local/document.pdf --sha256 SOURCE_SHA256 \
  --page 0 --hints '["en"]' --output /local/new-page-result
```

`configure.py` verifies every model file against `assets.lock.toml` and compiles
the two supplied Swift sources. It never downloads. Model repositories are
PaddlePaddle/en_PP-OCRv5_mobile_rec, PaddlePaddle/el_PP-OCRv5_mobile_rec, and
PaddlePaddle/PP-OCRv5_server_rec; exact revisions are in the lock. The first two
model cards declare Apache-2.0. Obtain assets separately with explicit consent.
Generated absolute paths, caches, page crops, source documents, and model weights
must remain outside Git. Native Vision requires the OS execution permissions
available to a normal local macOS process.

The parent evaluator must enforce a 300-second wall timeout per page and inspect
`record.json`: worker exit zero means a diagnostic record was written, **not**
that parsing succeeded. Require `status=completed`, unchanged source identity,
and the intended result checks. Each output directory is single-use. Retry uses
a fresh directory; no partial result is committed into a book workspace.

Run focused checks from the repository root with the OCR Python:

```sh
python -m unittest discover -s plugins/processor-paddleocr/experiments/hybrid -p 'test_*.py'
make quality
make test
make build
```

## Acceptance and limits

Acceptance requires all original six cases and twelve anchors, the five original
controls, and a frozen six-page independent synthetic set. Source hashes, page
indices, expectations, and original error messages are unchanged. Expected text
is supplied only to the evaluator, never this worker.

The route aligns native characters, Vision regions, Tesseract words, and pinned
Paddle recognizers. It preserves source resolution only for a narrowly proven
full-page opaque raster; otherwise it renders. One fixed secondary render can
corroborate native text. Mixed fractions require source-pixel geometry and digit
recognition, never a fixture substitution. Unresolved coverage, native conflicts,
unsupported-script evidence, or reader disagreements reject the page.

Model agreement is not proof of truth. This sample does not qualify general PDF
accuracy, arbitrary languages, handwriting, tables/charts, whole-book processing,
concurrent service use, or production registration. The Greek recognizer supplies
rejection evidence, not supported Greek transcription. Numeric layout covers
single-digit proper mixed fractions only. Unknown shapes abstain. Block
confidence is zero (unknown), not a calibrated correctness probability.

The Python socket audit hook and provider offline settings are defense in depth,
not an OS network sandbox. Binary hashes do not pin the complete OS/framework,
shared-library, or Tesseract language-data installation. Run qualification on a
controlled offline host; record OS/runtime identity from the receipt. There is
no claim of hostile-file containment or arbitrary provider output resource safety.
Do not expose this experimental CLI as a network service.

## Reliability review

Source snapshot and hashes bind authorized input. Model/binary checks run before
and after parsing, including provider failure. Caller configuration is copied.
Provider subprocesses have timeouts; the parent bounds the overall page. Output
is external and single-use, so interruption leaves diagnostic residue rather than
a valid workspace cache. Explicit close paths release PDF and OCR objects.
Compatibility is preserved by the distinct experimental identity. No persistent
contract, authentication surface, deployment, or migration is introduced here.
