# 0024: Serve offline workspace Search and Evidence from verified FTS5 generations

- Status: Accepted
- Date: 2026-10-04
- Owners: Capicove maintainers

## Context

The installed `capicove search` and `capicove get-evidence` commands could only
run when a Python host injected handlers. RFC-0018 defines the first offline
source-to-Search-to-Evidence journey under the version-2 workspace host, and its
maintainer review fixed SQLite FTS5 as the base backend, the six-field
`capabilities.retrieval` schema, and deferred Markdown-caption-to-image linkage.

## Decision

- The `retrieval` capability is a version-2 host capability only. It owns the
  canonical `capicove.search` and `capicove.get_evidence` Tool IDs; no new Tool
  ID is registered. Both the CLI (`--runtime-config`) and MCP compose the same
  bindings through the workspace host.
- The text lane reuses the RFC-0008 `TextSearchService`, the SQLite generation
  catalog, and the generation lifecycle ports. A capability-owned SQLite FTS5
  adapter implements those ports; it does not use the Wiki capability's private
  search index.
- `capicove source import` and `capicove index build` are operator-only local
  writes that require the write or admin profile. Search, Evidence, doctor,
  catalog, and MCP startup never create storage, build an index, or open the
  network.
- Each build produces an immutable generation whose identity is derived from
  the verified source inventory, chunking version, index format, workspace, and
  corpus. Activation writes one atomic alias file after validation; a failed
  build leaves the previous active generation unchanged.
- Evidence IDs are opaque and bound to one generation. The reader resolves only
  the active generation, rechecks workspace identity, the stored text hash, and
  the source hash, and fails closed for stale, foreign, or tampered references.
  It never reopens a source file by path.
- The canonical Evidence sections are closed and include no document-text
  section, so the exact chunk text is returned inside `metadata`. Chunks are
  bounded so that section stays within the Tool's 4 KiB section limit.
- The canonical Search spec declares network access because other providers
  need it. The local binding declares `network_access=False`, because the FTS5
  lane is offline; the shared input and output contract is unchanged.
- The retrieval capability is POSIX-only in this release. Windows fails closed
  until no-follow source reads and ACL checks have an equally strong Windows
  implementation, matching the Memory capability precedent.

## Consequences

- An installed wheel can import authored Markdown and images, build, search,
  and open exact Evidence with no service, model, or network.
- Image imports are Evidence only; they are not lexical hits.
- Unicode word tokenization favors space-delimited languages. CJK recall is a
  known limitation until a reviewed tokenizer or dense lane is added.
- Windows support and native-OS wheel qualification remain release gates.

## Amendment 1 (2026-10-06)

RFC-0018 amendment 1 adds `capicove source forget` and an append-only version-2
import ledger with an incompatible sentinel view, so a version-1 reader fails
closed instead of ignoring retirement. Builds may activate an empty generation,
all ledger mutations and builds share one interprocess lock, and builds stream
sources in bounded batches instead of holding the corpus in memory.

## References

- [RFC-0018: Offline workspace retrieval and source import](../rfcs/0018-offline-workspace-retrieval.md)
- [RFC-0008: Unified text retrieval contracts](../rfcs/0008-unified-text-retrieval-contracts.md)
- [ADR-0023: Compose capabilities from one versioned workspace authority](0023-versioned-workspace-host-authority.md)
