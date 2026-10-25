# 0019: Keep configurable book chunks independent from source evidence

- Status: Accepted for the user-authorized chunk-configuration scope
- Date: 2026-09-03
- Owners: Capicove maintainers

## Context

Textbook and technical-book chunk quality depends on structure, content type,
input budget and tokenizer. Tuning must not rerun OCR, replace immutable evidence
or silently reuse summaries generated from different input.

## Decision

Store version-1 chunk runs separately under each prepared book. Bind source,
representation, finalized structure, algorithm, resolved chunk policy and
tokenizer identity into run IDs; bind ordered source spans into occurrence IDs.
Preserve disjoint core ownership and source-mapped optional context. Revalidate
coverage and source hashes on read. Never treat model-generated text as evidence.

Keep immutable context-read policies separate from chunk identity. Context reads
enforce an explicitly selected policy or the most restrictive saved policy.
Expose read-only previews, bounded pagination and source reads through CLI and
canonical Tools. An optional real MCP stdio adapter is bound to a host-authorized
workspace; it cannot load arbitrary external configuration or tokenizer paths.

Use an explicitly labeled UTF-8 byte estimate by default. An opt-in local
tokenizer records asset/runtime identity and disables truncation/padding. No
network or generation model is selected. The caller budgets full prompts.

Evidence-only preparation has a separate book identity and an immutable
`preparation-policy.json` sidecar (`schema_version: 1`, `evidence_only: true`).
Legacy books without this file keep existing task-generation behavior.

The accepted front-matter clarification in RFC-0014 adds a versioned chunk-only
policy: exclude confidently identified cover/copyright pages, keep TOC chunks as
separate navigation, and retain guidance/prefaces/uncertain content. Missing
persisted policy fields retain legacy behavior and identity. Source-derived
validation applies equally to core, optional overlap and Parent context. The
policy never rewrites parser evidence, old chunk runs or legacy summaries.

## Consequences

Reconfiguration is safe to preview and compare; old runs and Wiki remain intact.
New reference-only reduction DAGs cannot be passed to legacy summary submission
or publication. Integrating execution and durable new-style results requires a
separate explicit contract change. Strict byte/span/token ceilings may split a
protected group; the result must warn rather than silently lose evidence.

## Alternatives considered

- Mutating existing chunks and summaries: loses lineage and invalidates citations.
- Selecting a generation provider internally: conflicts with caller-owned Agents.
- Blind fixed windows: ignore book structure and semantic-unit cues.

## References

- [RFC-0014](../rfcs/0014-configurable-book-chunk-runs.md)
- Configuration design (archived development reference)
