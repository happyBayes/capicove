# ADR-0030: Serve installed BGE-M3 through an optional compatible HTTP boundary

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- RFC: [RFC-0026](../rfcs/0026-bge-m3-embedding-http.md)

Compose a standalone FastAPI boundary over the existing verified BGE-M3 adapter.
Keep FastAPI, Uvicorn and model SDK imports out of ordinary CLI imports. Install
the existing `api` and `text-embedding` extras; introduce no mandatory dependency.
Expose serving through the packaged `capicove models serve bge-m3` command.

Count tokens and produce embeddings in one adapter operation to avoid a second
tokenizer implementation and misleading usage estimates. Preserve existing dense
embedding identities, normalization and offline-only runtime behavior. The HTTP
protocol supports text inputs and fixed dimensions explicitly rather than
silently interpreting another model's token IDs or truncating vectors.

Default to loopback. Remote binding is an explicit operator choice requiring a
bearer credential supplied through a Capicove environment variable. Bound request
size and concurrent admission before inference, and release the model on normal
shutdown. This is a single-owner model endpoint, not the workspace Tool host or a
multi-tenant inference platform.
