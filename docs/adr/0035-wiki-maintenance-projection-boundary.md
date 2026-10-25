# 0035: Maintain Wiki Knowledge Through Reviewed Projection Revisions

- Status: Accepted for requester-authorized implementation on 2026-10-22.
- Scope: Implementation only; independent review and release remain separate.
- Date: 2026-10-22
- Owners: Capicove maintainers
- RFC: [RFC-0031](../rfcs/0031-wiki-knowledge-maintenance.md)

## Context

ADR-0016 gives Wiki a standalone, user-editable, noncanonical projection boundary.
Its create-only ingest deliberately avoids replacing user work. Knowledge
maintenance requires intentional revision of existing concept pages, which needs
an explicit exception with stronger concurrency and provenance guarantees.

## Decision

Keep ADR-0016's authority boundary. Introduce schema-v3 managed page revisions,
with immutable proposal/review/publication records and an explicit publish step.
Only adopted managed pages may be revised; unmanaged pages and Book-generated
pages are read context until explicitly handled by their owning workflow.
Adoption preserves the original bytes and requires evidence-backed replacement
content. It never certifies arbitrary legacy prose as supported knowledge.

Replace create-only behavior only for an approved managed page whose exact base
hash and projection revision still match. All other ingest and publication
destinations retain their existing semantics. A source capture never overwrites
different bytes. Page identity remains stable across managed revisions.

Use the existing cooperating-process lock and a durable recovery journal to
publish multiple files. Readers fail closed during incomplete publication.
Do not promise atomic visibility to arbitrary Markdown editors or filesystem
readers. SQLite and impact indexes remain disposable. No distributed writers,
automatic page deletion, implicit review, or background model execution is added.

Canonical contracts remain model/database/framework independent. The caller
produces bounded compilation submissions; the host supplies authority and
evidence. Runtime may compose existing configured generation infrastructure
through a Wiki-owned port. Wiki never imports runtime or memory-private adapters.

## Consequences

- Cross-source concepts, disagreements and historical revisions become inspectable.
- Human edits invalidate proposals rather than being silently overwritten.
- Schema migration and all-writer compatibility are required before activation.
- Publication history is durable workflow state, not canonical truth.
- Review quality remains separate from structural integrity and exact quotation.
- Local recovery can temporarily make Wiki query unavailable; the first version
  favors a comprehensible fail-closed protocol over uninterrupted reads.
- ADR-0016 retains its authority boundary; its create-only restriction is amended
  only for the managed v3 path once that implementation is qualified.

## Rejected alternatives

Direct Agent writes lack enforcement and recoverable publication. Automatic
canonical graph mutation violates existing ownership. A new durable SQL Wiki
truth store duplicates portable records. A hidden v2 sidecar is rejected because
older writers could ignore it and expose mixed or unreviewed state.

## Supporting documents

- [ADR-0016](0016-local-wiki-projection-boundaries.md)
- [RFC-0015](../rfcs/0015-book-chunk-results-and-publication.md)
- Detailed design (archived development reference)
