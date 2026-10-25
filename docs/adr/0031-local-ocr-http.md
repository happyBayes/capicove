# ADR-0031: Compose local OCR HTTP through public image provider ports

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- RFC: [RFC-0027](../rfcs/0027-local-ocr-http.md)

Reuse `ImageOcr` and `ConfiguredImageOcrProvider` at an optional FastAPI boundary.
The API accepts a factory for a closable provider. CLI composition selects the
Paddle provider using `ImageOcrConfiguration`, an operator-supplied installation
directory and the existing offline model verification. Keep web and Paddle SDK
imports out of ordinary CLI imports and canonical contracts. Use existing `api`
extras and the separate Paddle plugin; introduce no mandatory dependency.

Use binary uploads so clients need no shared filesystem and requests cannot
expand filesystem or network authority. Preserve public evidence results and
provider fingerprints, with line-level Paddle geometry unchanged. Bound encoded
and decoded input sizes, serialize model admission, warm before Uvicorn takes ownership of process signal handlers,
claim the provider during lifespan startup and close during lifespan shutdown.
The CLI also closes it if listener startup fails. Remote binding requires a startup-owned
bearer credential. This service is independent of the workspace Tool host and
does not change its permission, tracing or isolation contracts.


The service refinement centralizes HTTP configuration and typed errors at the
application boundary, removes `api.model_http`, and moves resident lifetime and
cancellation-safe admission to `runtime.ocr_service`. CLI and lifespan share that
owner; they do not each release the underlying provider. Image verification and
decode use capability-owned preparation through `ImageOcr.recognize_upload`.
