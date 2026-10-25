# RFC-0031: Evidence-Grounded Wiki Knowledge Maintenance

- Status: Accepted for requester-authorized implementation.
- Authorization: The requester approved this design and W0-W5 execution on 2026-10-22.
  Independent code-owner review, release and deployment remain separate gates.
- Date: 2026-10-22
- Owners: Capicove maintainers; resolve reviewers from `.github/CODEOWNERS`.
- Decision: [ADR-0035](../adr/0035-wiki-maintenance-projection-boundary.md).
- Specification: Detailed design (archived development reference).
- Delivery: Implementation plan (archived development reference).

## Problem

Standalone Wiki ingest captures Markdown and creates a page without cross-source
synthesis. Query returns lexical hits rather than an authored answer. Book
workflows already accept caller-owned summaries and publish evidence-linked
claims, but do not maintain concepts across books and notes. Native `kb ask`
already supports generation and exact-quote validation; this proposal must not
describe all of Capicove as lacking question answering or duplicate that runtime.

## Proposed contract

Add a model-neutral Wiki maintenance workflow: capture, discover, issue bounded
tasks, accept proposed compilations, validate, preview, explicitly apply, and
recover. Agent decisions remain proposals. Only the Wiki publisher changes the
visible projection. New material can create concepts, revise existing pages,
record disagreements or supersession, or be recorded as adding no material.
Content hashes establish integrity, not truth or semantic novelty.

The normative detailed specification defines closed versioned records, exact
evidence locators, task limits, review policy, error codes, optimistic concurrency,
publication recovery, and proposed CLI/Tool operations. These names are reserved
design targets, not currently available APIs. Dependency-light interchange values
belong in `capicove.contracts.wiki_maintenance`; capability orchestration and ports
belong in `capicove.wiki.maintenance`. Generation composition belongs in runtime.

## Compatibility and authority

Introduce explicit Wiki schema v3 for maintenance-enabled workspaces. Leave v1/v2
behavior and existing CLI output unchanged. Migration is previewable and does
not infer claims or rewrite source bytes. Old binaries must reject v3. The v3
gate is not merely a relaxed integer check: every writer, Book publisher, reader,
lint and index path must participate in the publication protocol before v3 ships.
Book chunk packet v1 remains explicitly v2 and is rejected on v3; a separate
versioned v3 publication packet is proposed rather than changing its meaning.

Wiki revisions and dispute records describe a local projection. They are not
canonical Temporal Memory facts, identity merges, or evidence acceptance. There
is no automatic synchronization to `knowledge`, native KB, or Temporal Memory.
The existing graph ledgers and Book citation ledger retain their meanings.

Network access remains explicitly configured. Tasks and citations grant neither
filesystem authority nor permission to send private content to a provider.
Publication approval is a host-owned operation bound to the exact proposal and
base revision; a model response cannot manufacture it.

## Alternatives

1. Skill-only file editing: simplest interface, but cannot enforce concurrent
   edit preservation, review identity, or recovery. Use a Skill as a client later.
2. Model-neutral tasks plus guarded publication: recommended; builds on Book's
   separation of generation from evidence validation and preserves offline use.
3. Wiki as an autonomous canonical graph: rejected; conflicts with ADR-0016 and
   duplicates existing knowledge ownership and governance.

## Acceptance

Accept this RFC and ADR-0035 before public-contract implementation. Separate gates
cover deterministic correctness, bilingual retrieval, actual Agent compilation
quality, privacy, and human acceptance. Passing citation checks is not semantic
entailment. The plan specifies independent tests, release blockers and receipts.
No benchmark, implementation, deployment or approval is claimed by this document.
