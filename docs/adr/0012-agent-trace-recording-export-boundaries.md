# 0012: Establish Agent Trace Recording and Export Boundaries

- Status: Accepted
- Date: 2026-08-18
- Owners: Capicove maintainers

## Context

Capicove needs to preserve detailed observable Agent operations across image,
video, audio, and mixed-modal work. Downstream users need enough immutable evidence
to clean and transform trajectories for different multimodal model families without
making the runtime own any trainer-specific representation.

The capability crosses persistent-format, storage, privacy, CLI, MCP, and batch
boundaries. Without explicit ownership, transport code could implement different
record semantics, batch could depend on a database, or a persistent SQLite layout
could leak into canonical contracts.

## Decision

`capicove.traces` owns event contracts, recording, sanitization, and storage ports.
`capicove.export` owns portable JSONL packages, schemas, media copies, and
validation. `capicove.batch` exposes an observer port and does not import either
capability. CLI and MCP invoke the same tracing wrapper. SQLite remains a private
adapter. Recording and media copying are explicit local opt-ins.

Canonical trace and export contracts use only the Python standard library. The
private SQLite store assigns per-Episode sequence numbers and predecessor hashes
transactionally. Raw records remain append-only; feedback links to existing records
instead of changing them.

The export format is event-per-line, sharded JSONL with a versioned JSON Schema and
canonical manifest. Filters select whole Episodes. Media is referenced by default;
explicit copy mode verifies local content and uses a content-addressed layout
without rewriting stored records.

Tool arguments enter recording through one `ToolSpec`-aware sanitizer and shared
Tool wrapper. Batch exposes only a completion observer protocol; the trace adapter
lives in `capicove.traces`. External Agent hosts may record visible messages only
through an explicit host-supplied `TraceSession`.

## Consequences

- CLI and MCP preserve identical Tool call/result trace semantics.
- Recording can be absent without changing existing execution behavior.
- Configured recording fails closed instead of silently losing observable events.
- Batch execution stays independent from traces, databases, and export policy.
- Downstream cleaners can reconstruct ordered multimodal Episodes without
  model-specific tokens or objective fields.
- SQLite and temporary aggregation databases can evolve privately while public
  ports and export schemas receive compatibility review.
- Exported data remains sensitive operator-owned content even after mandatory
  credential, base64, and absolute-path sanitization.

## Alternatives considered

- **One JSON object per Episode.** Rejected because it requires buffering an
  unbounded trajectory and prevents bounded one-pass export and validation.
- **Direct SFT or Agentic RL rows.** Rejected because `chosen`, `rejected`, reward,
  and model-specific media-token conventions discard source detail and couple
  Capicove to a trainer.
- **Tracing separately in every transport.** Rejected because schemas, sanitization,
  failure behavior, and Tool identity would drift.
- **Make batch import traces.** Rejected because it reverses capability ownership
  and couples reusable execution to one observer implementation.
- **Expose SQLite models publicly.** Rejected because database concerns do not
  belong in canonical contracts and would prevent other storage adapters.

## References

- [RFC-0006: Agent multimodal trace JSONL](../rfcs/0006-agent-multimodal-trace-jsonl.md)
- [ADR-0003: Tool Runtime, Trace, and controlled RSI boundaries](0003-tool-runtime-trace-rsi-boundaries.md)
- Agent multimodal trace export design (archived development reference)
- Issue #56 (archived development review)
