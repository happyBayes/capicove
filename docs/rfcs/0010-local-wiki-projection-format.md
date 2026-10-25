# RFC-0010: Local Wiki Projection Format

- Status: Accepted
- Date: 2026-09-01
- Owners: Capicove maintainers
- Approval: The user explicitly approved the previously identified Wiki version 1
  format decision on 2026-09-28. Code review, merge, and release remain separate
  gates.

This record documents the accepted Wiki version 1 format. Acceptance does not
assert that implementation or documentation tests pass and does not authorize
merge or release.

## Summary

Capicove provides a lightweight, offline `wiki/` workspace for user-editable
Markdown pages, local source snapshots, optional graph-interchange ledgers, a
deterministic human index, and a disposable SQLite FTS5 search projection. The
`standalone` manifest value means that these operations need no running database
or network service. Wiki records are not canonical Capicove truth.

Temporal Memory remains the authority for persistent, automatically evolving
knowledge when it is connected. A future adapter may project its public,
revisioned snapshots into this format or submit explicitly reviewed Wiki records
through its public commands. Wiki never imports or writes a memory-private store.

## Persistent format

Schema version 1 contains `wiki.toml`, human documentation and logs, `raw/`,
`pages/`, `graph/ontology.yaml`, `graph/facts.jsonl`, and
`graph/identity-events.jsonl`. `_derived/` and `_migration/` are operational
directories rather than canonical knowledge. `_migration/writer.lock` serializes
cooperating mutating processes and is not replaced as part of an operation.

Raw Markdown captures use `source:sha256:<digest>` identities and
`raw/<digest>.md` filenames. Lint recomputes every digest before a build. Page,
fact, and identity records retain stable IDs and explicit source references so a
later bridge can preserve lineage without treating Wiki files as memory events.

Version changes require an explicit migration preview. Capicove never silently
rewrites pages, raw captures, facts, or identity records.

Ingest publishes new raw captures and pages with create-only filesystem operations.
If a page is durable but its log append fails, a same-source retry repairs the one
missing source-identified log entry rather than creating another page.
This guarantee requires the local filesystem containing `wiki/_migration`,
`wiki/raw`, and `wiki/pages` to support hard links within that filesystem. Ingest
fails closed with an explicit error when hard links are unsupported; it does not
fall back to copying, replacing, or overwriting a destination. The unsupported-link
path removes its private `.create-*` staging file without publishing a raw capture
or page. Source bytes remain unchanged, and every successfully published raw
capture remains byte-for-byte identical to the supplied file.

The following schema describes the current version 1 reader, validator, and
writer. Required fields and reference rules are the conditions for a clean lint
result; explicitly listed validator limitations are not stronger guarantees.
Pages, facts, and identity events inherit the manifest version and have no required
per-record version field.

### Manifest: wiki.toml

The manifest is a UTF-8 TOML document with these required top-level keys:

```toml
schema_version = 1
authority = "standalone"
```

| Field | Version 1 rule |
|---|---|
| `schema_version` | Exactly integer `1`; booleans, floats, strings, and other integers are rejected. |
| `authority` | Exactly the string `standalone`, the sole `WikiAuthority` value. It describes service independence, not canonical ownership. |

The required directories are `raw/`, `pages/`, `graph/`, `_derived/`, and
`_migration/`. Required regular files are `wiki.toml`, `SCHEMA.md`, `index.md`,
`log.md`, the three graph files below, and `_migration/writer.lock`. Lint checks
the log is readable UTF-8. The current validator checks the existence of
`SCHEMA.md` and `index.md`, not their headings or content. Initialization writes
a one-byte NUL lock sentinel; no knowledge schema is attached to its bytes.

`_migration/build.pending` is an optional operational marker whose presence means
a build has not completed. Build writes `Build incomplete; rerun wiki build.`
followed by a newline before replacing derived output, and removes the marker
only after SQLite, `index.md`, and the build log entry succeed. A failed build can
leave some of those outputs updated; the marker makes doctor report an unhealthy
index and query reject it until a successful rebuild. Its presence, not parsing
its payload, is the completeness check. This is recoverable publication with a
completion marker, not a physical multi-file transaction or rollback of the old
projection.

Ingest, build, query, and the index-health check use the same cooperating-process
lock. Query holds it across validation, index checking, and reading results.
Doctor takes its final lint, regular non-symlink index-presence, and index-health
values from one lock-protected snapshot whenever the Wiki passes its initial
read-only lint preflight. An invalid layout is reported as an ordinary unhealthy
diagnostic without following a derived-index symlink. The lock is opened
read/write. After the lock holder validates the manifest, acquisition may clean
stale `_migration/.create-*` publication files. Query and doctor preserve pages,
raw captures, and graph ledgers, but can maintain this operational state;
byte-for-byte read-only access or support for read-only mounts is not promised.
An invalid manifest prevents entry to the locked operation before its lint pass,
so build and query surface `WikiFormatError` (`wiki_format_invalid` in the CLI)
rather than a `WikiValidationError` lint report.

### Pages: pages/**/*.md

Pages are recursively discovered regular files whose suffix is exactly `.md`.
Each basename must match `^[a-z0-9]+(?:-[a-z0-9]+)*\.md$`. Nested page directories
are allowed. Directory names, filename length, and Windows reserved basenames are
not additional lint checks; the ingest writer separately generates portable stems
of at most 200 ASCII characters. Human-index link targets percent-encode each
UTF-8 page path while preserving `/` directory separators; stored `page_path`
values remain raw Wiki-relative POSIX paths.

Page text is UTF-8 Markdown with a YAML mapping delimited by an opening `---`
line at the start of the text and a closing `---` line. Normal text-file newline
translation precedes parsing. A stored page body may be empty, but its frontmatter
must contain the required fields below. Source Markdown accepted by ingest may
instead have empty frontmatter. In either context, the closing delimiter may be
the final line without a terminal newline. PyYAML's safe loader parses the
frontmatter; every stored page uses these fields:

| Field | Version 1 rule |
|---|---|
| `id` | Required string matching `^page:[A-Z0-9]{10,64}$`, unique across all discovered pages. |
| `title` | Required nonblank string that can be encoded as UTF-8. No title-length bound is enforced. |
| `kind` | Required string: `entity`, `concept`, `comparison`, `query`, or `summary`. |
| `created`, `updated` | Both required dates, defined below. Their relative ordering is not checked. |
| `sources` | Required list of strings; each string must be a resolvable `source:sha256:<digest>` ID. An empty list and repeated IDs are accepted. |
| `entityTypes` | Optional only when `kind` is `entity`; defaults to `[]`. If present, it must be a list of registered ontology entity-type names. Presence on any other page kind is an error, even with an empty list. |

A date is a calendar-valid canonical `YYYY-MM-DD` string, or an exact Python
`date` value produced by a YAML date scalar. Timestamps, including YAML datetime
scalars, are not dates in this format. JSONL dates must therefore be strings.

For example, a page without source claims is valid:

```yaml
---
id: page:EXAMPLE0001
title: Example concept
kind: concept
created: 2026-09-01
updated: 2026-09-01
sources: []
---
```

Ingest derives the page ID from the first 26 uppercase hexadecimal characters of
the complete source SHA-256 digest, initially writes `kind: concept`, sets both
dates to the ingest date, and writes the single complete source ID in `sources`.
It selects the title from nonblank source frontmatter `title`, otherwise the first
level-one Markdown heading, otherwise the source basename without its suffix.
Other source-frontmatter fields are not copied into the generated page; the exact
original file remains in the raw capture. The body is preserved after removal of
source frontmatter and leading blank lines. Ingest strips one leading UTF-8 byte
order mark and normalizes CRLF or CR line endings to LF only in the generated
page text. The content-addressed raw capture retains the exact input bytes.

Wikilinks of the form `[[target]]`, optionally with an anchor or display label,
resolve case-insensitively to a discovered page basename without `.md` or a page
title. They do not resolve by page ID or relative path. Titles and basenames are
not required to be unique, so lint proves only that some target exists, not that
navigation is unambiguous. Wikilinks never create facts.

### Ontology: graph/ontology.yaml

The ontology is a UTF-8 YAML mapping. Initialization writes this empty ontology:

```yaml
schemaVersion: 1
entityTypes: {}
predicates: {}
labels:
  namespaces: []
```

| Field or declaration | Version 1 rule |
|---|---|
| `schemaVersion` | Required and expected to be `1`. The current validator uses equality rather than an integer type check, so YAML `true` and `1.0` also pass this comparison. Writers emit integer `1`. |
| `entityTypes` | Required mapping of nonempty string names to mappings. Whitespace-only names are not separately rejected. |
| `entityTypes.<name>.parents` | Optional list, default `[]`; every item must name a registered entity type. Cycles, self-parenting, and repeated parents are not checked. |
| `predicates` | Required mapping of nonempty string names to mappings. |
| `predicates.<name>.kind` | Required string `object` or `data`. |
| `predicates.<name>.domain` | Optional list, default `[]`, of registered entity-type names. |
| `predicates.<name>.range` | For an object predicate, optional list, default `[]`, of registered entity-type names. A data predicate's `range` is not validated. |
| `predicates.<name>.valueType` | Required string for a data predicate. The validator does not define a datatype enumeration or require the string to be nonblank. It is not checked for object predicates. |
| `labels.namespaces` | Required list within a required `labels` mapping. Every item must be a nonblank string; the list may be empty. |

There is no label assignment or namespace-membership validation on page/fact
extension fields. Domain and range declarations resolve entity-type names but do
not constrain the actual types of a fact's subject or object in version 1.

### Facts: graph/facts.jsonl

The file is UTF-8 JSON Lines: every nonblank line must parse as one JSON object;
blank lines are ignored and the empty file is valid. The current parser does not
require objects to have a closed set of properties.

| Field | Version 1 rule |
|---|---|
| `factId` | Required string matching `^fact:[A-Z0-9]{6,64}$`, unique within this ledger. |
| `subjectId` | Required string resolving to an existing page ID. |
| `predicate` | Required registered ontology predicate name. |
| `objectId` | The object form: a string resolving to an existing page ID. Required for an object predicate. |
| `value`, `valueType` | The data form: the `value` key must exist and `valueType` must be a string exactly equal to the predicate's declared `valueType`. Required for a data predicate. |
| `status` | Required string `current`, `superseded`, or `retracted`. |
| `knowledge` | Required string `asserted` or `inferred`. |
| `sources`, `rationale` | At least one nonempty source list or nonblank string rationale is required. A non-null `sources`, if supplied, must be a list of resolvable content-addressed source IDs. |
| `rule`, `inputFacts` | When `knowledge` is `inferred`, require a nonblank string rule and a nonempty list of strings. Each input must match `^fact:[A-Z0-9]{6,64}$` and resolve to a `factId` in this ledger; forward references are allowed. Repeated IDs, self-references, cycles, and inference correctness are not further checked. |
| `assertedAt`, `validFrom`, `validTo` | Optional dates. If both interval endpoints are present and valid, `validFrom` must not exceed `validTo`. `assertedAt` has no ordering relationship to them. |

Exactly one recognized object form is required: an `objectId` string, or presence
of `value` together with a `valueType` string. This is a check on recognized
markers, not a strict prohibition on unused properties: for example, a non-string
`objectId` does not select the object form, and `value` without a string
`valueType` does not select the data form. The value's JSON type is not validated
against the declared datatype; `null`, containers, and scalar values are all
accepted when the data-form markers and predicate declaration agree.

For asserted facts, `rule` and `inputFacts` are not checked. A valid source list
can satisfy provenance without a rationale; conversely, a nonblank rationale can
satisfy it when sources are absent, `null`, or empty. A malformed non-null sources
field is still an error even when a rationale exists. Duplicate source references
are accepted. Status values are labels only: lint neither verifies supersession
chains nor applies retractions to another record.

### Identity events: graph/identity-events.jsonl

The UTF-8 JSONL object and blank-line rules are identical to the fact ledger.

| Field | Version 1 rule |
|---|---|
| `eventId` | Required string matching `^event:[A-Z0-9]{6,64}$` or `^identity:[A-Z0-9]{6,64}$`, unique within this ledger. |
| `type` | Required string `MERGED_INTO` or `SPLIT`. |
| `rationale` | Required nonblank string. |
| `effectiveAt` | Required date string as defined for pages. |
| `fromId` | Required string resolving to an existing page. |
| `toId` | For `MERGED_INTO`, required resolvable page ID distinct from `fromId`. |
| `toIds` | For `SPLIT`, required list of at least two strings, each resolving to an existing page. |

The split validator does not require distinct targets or exclude `fromId` from
them. Unused `toId`/`toIds` properties are not rejected. Event chronology, merge
cycles, and consistency with other events are not checked. These records are
interchange data: lint does not execute identity changes or rewrite page IDs.

### Raw captures: raw/<digest>.md

Every regular file under `raw/` must be a direct child named with exactly 64
lowercase hexadecimal SHA-256 digits followed by `.md`. Nested captures and
duplicate digest identities are errors. The source ID is
`source:sha256:<digest>` and lint recomputes SHA-256 over the exact file bytes;
different bytes at that name are an error. Clean page/fact source references must
resolve to these captures.

Ingest accepts only readable UTF-8 source bytes and preserves them exactly, with
no newline, frontmatter, or Unicode normalization of the raw capture. Raw lint
checks path, identity, and bytes, but does not separately decode manually created
captures as UTF-8 or parse their Markdown. Hash validity proves byte identity, not
authenticity, source permission, or acceptance into canonical Capicove evidence.

### Compatibility and extension fields

Version 1 rejects unsupported manifest versions; there is no automatic migration
implementation. Changing the persistent format still requires an explicitly
reviewed migration preview. `SCHEMA.md` is descriptive and is not another version
authority.

Earlier experimental Wikis without the now-required `_migration/writer.lock`
are rejected by layout validation or lock acquisition. Operations do not silently
add the lock or migrate those Wikis, even if their manifest also says version 1.
Adopting this candidate for such data requires a reviewed migration; the shared
version number alone does not establish backward compatibility.

Unknown TOML keys, YAML mapping fields, and JSON object fields are not rejected
by the current readers. Existing Wiki pages and graph files are not rewritten by
lint, build, or query, so their unknown fields remain on disk. This is permissive
read behavior, not a registered extension system or a promise that extensions are
validated, included in the search index, executed, or understood by future
adapters. Ingest creates its own page fields and does not round-trip arbitrary
source metadata. Extra non-`.md` page files are not treated as pages.

YAML uses `safe_load`; JSONL uses Python's default JSON parser. Neither adds
duplicate-key rejection (the last mapping value wins); YAML anchors/aliases are
not separately forbidden. Default JSON non-finite constants are not rejected.
No schema-specific file-size, nesting-depth, string-length, or record-count limit
is enforced by these parsers. These are current validation limits for maintainer
review, not claims that arbitrary hostile input is safe. Tightening accepted
forms or assigning meaning to extension fields needs explicit compatibility
consideration.

## Validation and safety

The Wiki root, required files, managed directories, and all descendants must not
be symbolic links and must resolve beneath the Wiki root. Parsing uses
local UTF-8, TOML, YAML, JSONL, and Markdown. YAML is an optional dependency loaded
only when needed. No network, telemetry, model, graph database, or external API is
enabled by Wiki operations.

Input source leaves and their lexical ancestors must not be symbolic links.
Source paths containing a `..` component are rejected before normalization. A
source file must remain the same regular-file snapshot while it is read. Writer
locks coordinate Capicove processes; non-cooperating writers and network
filesystems are outside the version 1 concurrency guarantee.

Ontology, Fact, Identity Event, page, and source records are versioned and checked
before derived output is rebuilt. SQLite indexes carry a format version and a
fingerprint of their page inputs; health checks reject missing, corrupt,
incompatible, or stale indexes.

## Query contract

Local search accepts at most 2,048 Unicode code points, 8,192 UTF-8 bytes, 128
searchable tokens, and 100 results. The searchable-token count comes from the
same SQLite `unicode61` tokenizer used by the index; punctuation-only input that
emits no token is rejected. This prevents application-side word approximations
from understating the actual FTS workload. Word grouping is retained only when
re-tokenizing those groups produces the identical ordered token sequence;
otherwise the exact `unicode61` terms form the match clauses so tokenizer-only
code points are not dropped. Results contain bounded plain-text
snippets, page IDs, relative page paths, and source IDs. Complete content remains
in the local workspace.

## Acceptance gates

Implementation uses failing tests for persistent layout, path containment,
content hashes, schemas, idempotency, deterministic output, corrupt/stale indexes,
query bounds, CLI behavior, and lazy imports. Completion requires `make quality`,
`make test`, and `make build` subject to documented pre-existing repository
baselines.

## References

- [ADR-0016](../adr/0016-local-wiki-projection-boundaries.md)
- Proposed design (archived development reference)
- Repository architecture (archived development reference)
