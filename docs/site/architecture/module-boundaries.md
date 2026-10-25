# Module boundaries

Dependency direction points inward toward public contracts:

- `capicove.contracts` owns transport-neutral contracts.
- `capicove.source` owns immutable source evidence and its privacy authority.
- `capicove.knowledge` owns temporal facts, entity identity and the knowledge ledger.
- `capicove.agent` owns memory admission, retention, recall and context policy.
- `capicove.runtime` composes host-selected providers and authorization; it does
  not accept model-selected paths or permissions.
- `capicove.processing.<capability>` owns public operations and value objects.
- `capicove.graph_build` owns the document-to-graph application service and depends
  on canonical graph/retrieval contracts plus injected authorization, input,
  generation, embedding, partitioning, storage, and retrieval ports.
- `capicove.retrieval` owns public batch text embedding and text-index generation
  contracts; graph building consumes those public contracts and never imports a
  retrieval capability's private adapter.
- capability-private `_adapters` implement ports and may use optional dependencies.
- `capicove.tools` adapts capabilities to canonical Tool handlers.
- `capicove.batch` keeps discovery and execution backend-independent; its media
  adapters invoke only public canonical Tool handlers.
- `capicove.wiki` owns the standalone file projection, validation, lazy YAML
  parsing, and disposable SQLite FTS index without importing Temporal Memory
  private state.
- CLI and MCP modules adapt Tool handlers where a Tool contract exists. The Wiki
  CLI directly adapts its public application service.

Capabilities must not depend on another capability's private implementation.
Optional model, GPU, and transport dependencies stay out of base import paths until
the relevant feature is used.

Graph snapshots stop at immutable graph refs and optional retrieval bindings.
Temporal Memory and Wiki capabilities may later consume those public revisioned
outputs, but graph building must not import their implementations or write their
stores. NetworkX, PyArrow, Torch, Transformers, and Sentence Transformers remain
behind lazy capability-local adapters.
