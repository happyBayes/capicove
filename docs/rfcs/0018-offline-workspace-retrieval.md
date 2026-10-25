# RFC-0018: Offline Workspace Retrieval and Source Import

- Status: Accepted (2026-10-04), with the review decisions recorded below.
- Date: 2026-09-30
- Owners: Capicove maintainers
- Review scope: The first installed-wheel, offline source-to-Search-to-Evidence
  journey under the version-2 workspace host. This RFC does not approve model
  downloads, OCR, dense embeddings, remote search, or a new MCP write Tool.
- Related contracts: [RFC-0002](0002-multimodal-retrieval-contracts.md),
  [RFC-0008](0008-unified-text-retrieval-contracts.md), and
  [RFC-0017](0017-workspace-cli-mcp-host-v2.md).

## Decision requested

Approve SQLite FTS5 as the mandatory, standard-library, lexical backend for
the first local text profile. It is a capability-owned adapter, not a canonical
contract or a dependency on the Wiki capability's private search adapter.
Unsupported SQLite builds fail readiness with `missing_dependency`; they do
not silently select another backend. The backend may later be supplemented by
reviewed, explicitly provisioned dense or remote lanes without changing the
identity of existing source evidence or generations.

## Version-1 capability configuration and authority

The version-2 host's `capabilities.retrieval` object has exactly these keys:

| Key | Rule |
| --- | --- |
| `schema_version` | Exact integer `1`. |
| `source_roots` | Nonempty map of at most 16 unique lowercase aliases to absolute, pre-existing, local directories. |
| `index_root` | Absolute, private, local directory inside `workspace_root`, disjoint from every source root. |
| `corpus_ref` | Nonblank, startup-owned canonical corpus identity. |
| `retrieval_profile_ref` | Exact `local-default`; no caller-selected backend or weights. |
| `max_source_bytes` | Integer from 1 through 16 MiB, applied per imported file. |

No extra fields, relative roots, symlink/reparse components, remote drives,
world-writable output, or overlapping input/output roots are accepted. Host
identity, permissions, network policy, and enabled Tool IDs remain owned by
RFC-0017. The read profile can select `capicove.search` and
`capicove.get_evidence`; these Tools require their existing exact permissions.
CLI import/build commands are operator-only local writes under the configured
roots and require the write profile. They never execute during host startup,
search, evidence expansion, or doctor. This version adds no new MCP Tool ID.

## Import and generation lifecycle

The proposed commands are `capicove source import --runtime-config PATH
--root ALIAS --relative-path RELATIVE` and `capicove index build
--runtime-config PATH`. `ALIAS` must be configured; `RELATIVE` is a bounded,
nonempty path beneath that root, never an absolute path or a traversal. The
importer accepts UTF-8 Markdown and local image files with recognized magic
bytes and a reviewed media-type allowlist. It opens source and parent
components with platform-native no-follow security, verifies regular-file
identity before and after a bounded read, and records source SHA-256, size,
media type, locator, and import version. The original file is never modified.

Markdown is deterministically split into nonempty text chunks with immutable
character ranges and document origin under RFC-0008. Base-profile image
import records exact bytes and metadata as Evidence but does not claim that
unprovisioned OCR or image embeddings make it text-searchable. An authored
Markdown caption may link to that image as source-linked evidence only when
both imports are explicitly present; the linkage format and its validation
must be fixed in acceptance tests before implementation. No inferred filename
or adjacent-file association is permitted.

`index build` creates a new immutable generation from the current verified
import inventory. It verifies every imported source hash immediately before
building, writes a generation manifest and FTS5 index in private staging,
validates counts, hashes, and schema, then atomically activates one generation.
A failed build leaves the previous active generation intact. Identical inputs
and policy may return the same generation identity. A source edit requires a
new explicit import and build; neither search nor evidence reads an uncommitted
generation. Old generation manifests and source identity remain available for
lineage even when derived index payloads are later evicted by explicit policy.

## Search and exact Evidence

The existing `capicove.search` and `capicove.get_evidence` contracts and
schemas are reused. CLI `search` and `get-evidence` gain an explicit
`--runtime-config PATH` that composes the same selected capability handler as
MCP. Both pin the active validated generation at invocation start. Search
supports document text chunks through the lexical lane and canonical filters;
it returns bounded snippets and opaque, generation-bound Evidence IDs. It does
not pretend image-only imports are lexical hits. Evidence expansion rechecks
workspace, corpus, source hash, representation and generation identity, and
the caller's read permission. Foreign, changed, stale, or tampered references
fail closed; no fallback opens a current file by path alone. CLI and MCP
results differ only in transport envelope fields already defined by their
contracts.

SQLite FTS5 schema, cursor encoding, tokenization version, chunking version,
manifest format, and activation receipt are private versioned storage details.
They must have deterministic migration/fail-closed behavior and tests; they do
not leak raw SQL, database paths, provider options, or backend scores through
public requests. Startup and ordinary reads are offline and do not create
storage, download a model, or rebuild an index.

## Acceptance and open review questions

PR02 requires an installed wheel outside the source tree: import authored
Markdown and an image, build generation A, Search and open exact Evidence,
edit a source, explicitly re-import and build generation B, restart, and replay
Search/Evidence over MCP. Tests must prove source bytes unchanged; stale or
foreign Evidence rejected; path escape, symlink/reparse swaps, missing FTS5,
invalid cursors, and interrupted activation fail closed. Record the exact
wheel, OS, SQLite, source-fixture, and generation identities. `make quality`,
complete `make test` with coverage, `make build`, and `make docs-check` remain
repository gates; native Linux, macOS, and Windows wheel cases remain release
gates.

Before acceptance, reviewers must explicitly decide whether SQLite FTS5 is
the required base backend, whether the proposed six-field config and two CLI
commands are the intended public surface, and how an authored Markdown caption
binds to an imported image without an implicit association. Until those
decisions are accepted, this document grants no public registration authority.

## Review decisions (2026-10-04)

The maintainer accepted this RFC with these answers to the open questions:

1. SQLite FTS5 is the required base lexical backend for `local-default`.
2. The six-field `capabilities.retrieval` configuration and the
   `capicove source import` / `capicove index build` commands are the public
   surface of this version.
3. Markdown-caption-to-image linkage is deferred. Image imports are recorded as
   source Evidence with exact bytes metadata only; no caption or image linkage is
   produced, inferred, or accepted until a later amendment fixes its format.

[ADR-0024](../adr/0024-offline-workspace-retrieval.md) records the resulting
implementation boundaries.

## Amendment 1: source retirement and bounded builds (2026-10-06)

- Status: Accepted by the maintainer on 2026-10-06 (package WP-1 of the
  next-stage proposal (internal record; public support status is documented in the capability guide)).
- Scope: one operator command, an import-ledger format change, empty
  generations, and bounded build memory. The six-field version-1 retrieval
  configuration is unchanged. A tokenizer field and CJK recall remain deferred.

### Command

`capicove source forget --runtime-config PATH --root ALIAS --relative-path RELATIVE`
requires the `write` or `admin` profile, a configured alias, and a normalized
relative path; the source file need not exist. It prints one JSON object
`{"source_ref": ..., "forgotten": true|false}`. `false` means the source had no
live inventory entry and nothing was appended. It never touches source bytes,
the active alias, or any generation.

### Ledger version 2

The import ledger (`<index_root>/imports.sqlite3`) becomes an append-only event
log:

| Object | Content |
| --- | --- |
| `ledger_meta` | `ledger_version` = `2` |
| `ledger_events` | `sequence`, `event` (`import` or `retire`), `source_ref`, `alias`, `relative_path`, nullable `sha256`/`size_bytes`/`media_type`/`import_version`, `recorded_at` |
| `imports` | A view with a single sentinel column, so a version-1 reader fails on read and on insert |

A source is live when its latest event is `import`. Retirement appends a
`retire` event; re-import after retirement appends a new `import` event even for
identical bytes. Prior events, hashes, manifests, and generations are never
rewritten.

Version detection: a ledger with an `imports` table and no `ledger_meta` is
version 1; it is migrated on the next write-profile operation inside one SQLite
transaction (copy events in sequence order, drop the table, create the sentinel
view, write the version). An interrupted migration leaves the original version-1
ledger. An unknown or newer version fails closed with no writes. Read paths
(Search, Evidence, doctor, catalog) never open the ledger.

### Builds

- Zero live sources in a ledger with events produces a valid, activated empty
  generation: Search returns no hits and former Evidence IDs are stale. A
  workspace that never imported anything keeps the existing error.
- A live source whose file is missing or changed still fails the build closed.
- Import, forget, and build share one index-scoped interprocess lock. A build
  holds it from inventory pinning through activation; other operations wait and
  take effect only at the next explicit build.
- Generation identity is derived from the ledger inventory, so it is known before
  any source is read. Sources are verified and chunked one at a time, and units
  are written in bounded batches; peak memory is bounded by `max_source_bytes`
  rather than the total corpus. Each source is read twice (count, then write) and
  both reads must match the recorded hash.
