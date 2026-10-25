# RFC-0026: OpenAI-compatible BGE-M3 embedding HTTP service

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- Authorization: The operator explicitly requested an HTTP layer for installed
  BGE-M3 models compatible with OpenAI `/v1/embeddings`. This authorizes local
  implementation, not independent review, publication, or release.
- Decision: [ADR-0030](../adr/0030-bge-m3-embedding-http.md).

## Contract and composition

`capicove models serve bge-m3` explicitly starts a standalone service using an
existing absolute model directory and independently trusted asset manifest
SHA-256. Install `capicove[api,text-embedding]` first. Reuse the verified public
BGE adapter, including its preprocessing, normalization, input bounds and offline
loading. No downloads, cloud fallback, canonical storage or workspace Tools are
introduced. The default listener is loopback, port 8766, CPU, one worker.

`POST /v1/embeddings` accepts `model` (`BAAI/bge-m3` or `bge-m3`), `input`
(nonblank string or 1–128 strings), optional `encoding_format` (`float`, default,
or `base64`), optional `dimensions` (1024 only), and optional `user` (ignored).
Output is an OpenAI list envelope with ordered indexed embedding objects,
canonical model ID and `usage.prompt_tokens/total_tokens`. Base64 encodes
little-endian float32 bytes. Token usage counts actual BGE input tokens including
special tokens, before padding. OpenAI token-ID inputs are rejected: those IDs
do not identify this model's tokenizer. No truncation, dimension shortening,
sparse vectors or ColBERT output is provided. See the
[official reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

The public BGE adapter gains `encode_with_token_counts`, preserving `encode` and
its contract. Counts and vectors come from the same locked, verified runtime;
an optional total-token budget rejects the batch before model inference.

`GET /v1/models` and `GET /v1/models/{model}` expose the configured model only.
`GET /health` reports readiness and non-secret model identity. Startup warms the
runtime before listening; shutdown closes it. Requests are bounded to 2 MiB,
128 texts, 300,000 total tokens and the configured per-input limit (default 512,
maximum 8192). One inference request is admitted at a time; another gets 429.
Health remains responsive while inference runs in a thread. Errors use the
OpenAI error envelope, never echo input, credentials, paths or runtime exceptions.

`--api-key-env CAPICOVE_...` optionally requires a startup-selected bearer key.
A non-loopback bind requires this option and a valid nonempty environment value.
An explicit listener start is the network opt-in. No request content or key is
logged; access logs are disabled. This standalone single-model listener does not
inherit workspace-host identity, permissions, token files or process isolation.
It provides no TLS or multi-tenant authorization.

## Validation

Test HTTP ordering, floats and float32 base64, actual token accounting, malformed
and oversized inputs, unsupported models/dimensions/token arrays, authentication,
concurrent admission, safe failures, shutdown, lazy optional imports and packaged
CLI dispatch. Exercise the installed OpenAI Python SDK without an external API
and compare a real local HTTP result to the existing BGE adapter. Run repository
quality, regression and build gates. No model weights or user payloads enter Git.
