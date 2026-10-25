# ADR-0033: Extend the generic VLM port and use a maintained HTTP transport

- Status: Accepted for implementation under RFC-0029
- Date: 2026-10-21

Keep one generic provider for compatible servers. Extend its message model for
tool conversations and add an opt-in vLLM token-counting operation. Tool execution
stays with the caller. Preserve immutable schemas and request/media provenance.

Use aiohttp with c-ares DNS in an optional extra instead of maintaining private
socket/TLS machinery. A synchronous call owns its event loop and closes resources
before releasing provider capacity. Async callers use a worker thread. This adds
small HTTP dependencies while removing application-managed DNS worker lifetimes.

Keep model acquisition explicit and pinned, and serve only verified local files.
Qwen-specific runtime settings stay in a reusable recipe. The accepted scope and
remaining protocol limitations are recorded in RFC-0029.
