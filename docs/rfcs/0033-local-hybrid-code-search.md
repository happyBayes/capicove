# RFC-0033: Embedded hybrid text retrieval and isolated code search

- Status: Accepted for staged implementation by requester direction on 2026-10-23
  (S1-S4 implemented); semantic, capacity and human acceptance remain separate
- Date: 2026-10-22
- Updated: 2026-10-23
- Owners: Capicove maintainers
- Decision: [ADR-0037](../adr/0037-local-hybrid-code-search.md)
- Design: Local hybrid and code search (archived development reference)
- Acceptance: Development and acceptance matrix (internal record; public support status is documented in the capability guide)

## Authorization and status

The user requested implementation, testing and review. This RFC proposes the
additional persisted vector format, versioned host settings and source-intake
authority needed to implement that outcome. The detailed decisions are not yet
accepted. Maintainer acceptance must precede production implementation. No model
transfer, Git commit, push, publication or production qualification is authorized
by this record. The 2026-10-23 request authorizes this staged document revision,
not acceptance of its proposed contracts or production implementation. Proposed
commands below are not currently runnable features.

RFC-0018 approves the existing FTS5-only local profile, not dense storage or code
intake. RFC-0008 provides reusable text candidates, embedding ports, generation
lifecycle and weighted RRF. RFC-0032 is a separate proposed multimodal/audio
program; this RFC neither depends on its acceptance nor changes its documents.

## Requested outcome

Add three opt-in, offline, single-node preview profiles:

1. `local-hybrid-v1`: local Markdown text chunks, embedded lexical and dense
   retrieval, canonical Search and exact Evidence.
2. `local-code-lexical-v1`: repository source and documentation with symbol/path
   lookup and exact evidence, without a model requirement.
3. `local-code-hybrid-v1`: explicitly selected repository source and documentation
   files, code-aware chunking and identifier lookup, hybrid retrieval and cited
   line coordinates.

Use independent configured index roots and corpora for notes and code. A code
profile never reads the notes inventory, and a notes profile never reads code.
Retain version-1 configuration and the `local-default` FTS5 implementation without
an automatic migration or model requirement.

## Staged delivery

All stages are proposed and unimplemented. Accept and qualify each stage before
enabling it; later stages do not block the earlier model-free code profile.

| Stage | Scope | Exit evidence |
| --- | --- | --- |
| S1: Symbols and paths | Safe code intake, deterministic line chunks, complete and split identifiers, qualified names, relative-path fields and independent definition lookup | Exact-definition Top-1, path/component fixtures, source/evidence integrity and offline CLI/MCP |
| S2: Optional AST | Build-time Tree-sitter adapter for Python and JavaScript/TypeScript, symbol scopes/signatures and source spans; bounded line fallback | Independent span oracles, Unicode/CRLF mapping, missing grammar and malformed-source recovery; S1 remains usable without AST |
| S3: Hybrid | Existing encoder ports, persisted vectors and weighted RRF for notes/code; preserve identifier priorities | Real local-model replay, retrieval ablations, filtered recall, typed failures and measured resource costs |
| S4: Bounded relations | Optional containment/import/direct-call facts and one-hop context expansion through public ports | Independently labelled edges, unresolved/ambiguous cases, cycle/fanout bounds, same-generation evidence and measurable benefit |

Use S1 as the minimum useful release. S2 is optional, S3 adds a model only when
explicitly selected, and S4 requires an AST-enabled policy. Ship each stage only
when its applicable acceptance rows pass. No LLM-generated Cypher, graph service,
LSP, runtime tracing or full type resolution is required by this proposal.

## Alternatives

1. **Selected:** SQLite FTS5 plus application-side bounded exact vector scoring,
   using existing encoder and generation ports. Small deployment footprint and
   independently testable retrieval; scoring remains linear in eligible units.
2. SQLite FTS5 plus FAISS text retrieval: additional native runtime and a new text
   adapter; defer until measurements justify ANN. The current visual FAISS adapter
   is not a text backend.
3. Require Elasticsearch/OpenSearch: reuse existing hybrid backends, but fails the
   requested embedded, service-independent deployment goal.

DigUp is a behavioral reference, not a dependency. Reimplement the required
behavior against Capicove contracts; do not copy Swift code or introduce a new
model family merely to match DigUp. Keep weighted RRF rather than treating its
corpus-dependent score calibration as portable confidence.

## Configuration and transport compatibility

Keep the workspace host schema at version 2. Add retrieval capability schema
version 2, with exactly the existing six fields plus `model_config_path`,
`model_config_sha256`, `model_binding`, `max_units`, `max_vector_bytes` and
`code_policy_ref`. These are proposed fields, not implemented configuration.

- `retrieval_profile_ref` is exactly one of the three new profile IDs.
- `code_policy_ref` is null for notes. For code, the startup-owned closed registry
  selects `code-symbols-v1` (S1), `code-ast-v1` (S2), `code-hybrid-v1` (S3) or
  `code-relations-v1` (S4). S1/S2 require the lexical code profile; S3/S4 require
  the hybrid code profile. S3 includes optional AST; S4 requires AST support.
  Unknown or incompatible combinations fail configuration validation. Each policy
  fixes parser/fallback, ranking and expansion settings; request text cannot
  enable stages or select parsers. Future variants require a reviewed registry
  revision rather than arbitrary flags.
- `model_config_path` is an absolute owner-private no-follow local configuration;
  `model_config_sha256` pins its exact bytes. The selected binding must resolve to
  a local `text_embedding` provider through the public model factory.
  All three model fields must be null for the lexical code profile; readiness,
  import, build and search then acquire no encoder. Hybrid profiles require all
  three fields to be non-null and valid.
- `model_binding` is a startup-owned binding name, not a caller-selected model.
- `max_units` is an exact integer from 1 to 10,000, across both lexical-only and
  vector-bearing units. `max_vector_bytes` is exactly integer zero for the lexical
  code profile, otherwise an exact integer from 1 to 268,435,456; reject a
  publication whose declared vectors exceed that budget.
- Existing root, permission, file-size, no-follow and network restrictions apply.
  Version-1 configurations reject new fields; version-2 configurations reject
  missing/unknown fields, booleans in integer fields and unsupported providers.

All profiles reuse `capicove.search`, `capicove.get_evidence` and their existing
MCP schemas. Add the local CLI alias `capicove code search QUERY --runtime-config
PATH`, which requires a code profile and calls the same Search handler. Ordinary
`capicove search` can query either explicitly configured profile. No `code:` prefix
silently switches roots, models, inventories or authority. No new canonical
modality, target, origin, entry-point group or MCP Tool ID is needed.

## Embedded retrieval format

Add a distinct `capicove-local-hybrid-v1` derived index format and provider ID.
Do not alter or reopen an existing FTS5-only generation for writes. Each new
immutable generation contains a private SQLite database, the existing canonical
text manifest and validation report, and a versioned provider descriptor.
The database has source and unit metadata, FTS5
body/path/identifier fields, and one vector row per vector-bearing unit. S1/S2
declare no embedding spaces or vectors and use the same new format's explicitly
versioned lexical variant. Stage/parser policy and optional symbol/edge payload
inventories are sealed in the descriptor; absent optional tables are declared,
never inferred from missing data. S3/S4 require a new generation, not an in-place
upgrade of a lexical generation. The legacy FTS5 format remains separate.

Vectors use little-endian float32, an exact declared dimension and full
`EmbeddingSpaceRef`. Hash each vector payload; bind the ordered vector inventory,
row counts, schema, text representations, model configuration digest and complete
space in the provider descriptor. Its canonical SHA-256 is bound into the existing
opaque `physical_index_ref`, recorded by the catalog and validation report.
Verify that descriptor and its payload hashes at reader open; never trust a
replaceable database's self-declared hashes. Preserve canonical manifest field
meanings and schemas. Validate finite values, normalization, unique
unit IDs, references and hashes before activation. Reject unknown versions,
truncated blobs, missing vectors and mismatched spaces. Lexical-only units are
explicitly marked; missing expected vectors are never interpreted as intentional.

Generation identity includes source hashes, workspace/corpus/profile identities,
chunking, tokenizer and code policies, parser/grammar versions and fallback
outcomes, model/preprocessing identity when applicable, and vector format.
No timestamps or absolute source paths enter portable content identity. Model or
preprocessing changes require a new generation. No implicit historical migration
or cross-runtime vector equivalence is assumed.

Build in private staging under the existing operator lock; verify sources before
encoding and publication. Failed builds leave the active generation unchanged.
Prepare and seal vector/database payloads before catalog registration so the
descriptor digest is known when lifecycle `create` records `physical_index_ref`.
This is a preparation step, not activation. Canonical manifest hashing does not
include the physical reference, avoiding a descriptor/database hash cycle. Reuse
an already published identical generation rather than re-encoding it. A retry
whose bytes conflict with a registered physical identity fails explicitly; it
cannot overwrite that identity or quietly change the descriptor.
Atomically activate after validation. Search and Evidence bind the same resolved
generation and never re-resolve the active alias halfway through a call. Reuse
the existing generation catalog and reconciliation behavior. Search never runs a
build, discovers sources or mutates an index.

## Model and ranking policy (S3)

Use the existing public `ModelFactory.text_encoder`, `TextBatchEncoder`,
`TextEmbeddingService` and `TextQueryEmbeddingProvider`. Reuse
`TextSearchService` only after adding tested, backward-compatible fatal-error
classification for the new backend: its current optional-lane exception handling
does not distinguish corruption/authority failures from provider unavailability.
Legacy providers retain their documented degradation behavior. The new backend
uses bounded typed failures that preserve the distinction before candidate fusion.
Only a narrow encoder-execution-unavailable category is degradable; invalid or
overlong queries, permission, storage, corruption and space errors are fatal even
when the configured dense lane is optional.
The first real-provider qualification uses already verified local BGE-M3 assets
when available; no assets are acquired by startup, doctor, build or search.
Deterministic encoders test orchestration, not semantic model quality.

Use role-specific encoding with explicit path/title context for code documents.
Include the context formatting policy in the processing identity. Keep query
encoding compatible with the selected provider; never inject EmbeddingGemma task
prompts into BGE. Respect provider token limits without silent truncation: split
oversized documents deterministically before final unit publication and reject an
overlong query. A single unrepresentable source character or unsupported encoding
fails the selected import with a bounded diagnostic.

Filter corpus, artifact, media type and text origin before dense top-k and lexical
top-k. Unsupported filters fail closed or yield the documented empty result;
they may never be ignored. Score normalized vectors by exact dot product in
bounded row batches. Do not load the full corpus matrix on every query. Fuse
lexical and dense ranks through existing weighted RRF with stable unit-ID ties.
Startup profiles fix weights, candidate limits and lane policy; request text
cannot change them. A declared optional encoder failure may yield visibly
degraded lexical results. Corrupt storage, permission failures and incompatible
spaces fail the query; they are not successful lexical fallback.

Emit `retrieval_preview` for all new profiles. Scores indicate ordering, not
probability. Formal no-answer calibration and an ANN capacity claim are excluded.

## Code intake and evidence

Support explicitly selected UTF-8 source files and repository Markdown documents.
The initial source extension/name allowlist, denylist and chunker version are
closed implementation constants covered by tests. Exclude notebooks, binary
files, generated/minified content, vendored code, hidden directories, symlinks,
credentials, lockfiles and dataset payloads in this initial release. Notebook
cell evidence needs a separately reviewed representation mapping and is deferred.

Individual `source import` operations use the same policy as repository discovery;
an explicit path cannot bypass denied paths or secret-content checks. Add an
operator-only `capicove code import --runtime-config PATH --root ALIAS` for bounded
repository discovery. The alias itself must be the configured repository root.
No recursive traversal outside that root or into submodules is permitted.

Use a selected local Git executable for `ls-files` enumeration of tracked and
untracked non-ignored files. Disable fsmonitor and hooks, use a scrubbed environment
and argument lists without a shell, and enforce subprocess deadline/output bounds.
Apply ignored-path policy to tracked files too. Git only enumerates candidates;
use a bounded `check-ignore --no-index` pass to enforce repository ignore policy
on tracked candidates rather than assuming `ls-files --exclude-standard` does so.
Safe descriptor-relative reads authorize and verify each selected source. Never
run repository scripts, filters, package managers or arbitrary Git commands.
Reject an unavailable Git dependency rather than falling back to unrestricted
recursive discovery. Record the executable identity, policy and selected file-list
digest in the import receipt. Bound candidates to 10,000 and cumulative selected
bytes to 64 MiB per invocation; permit subsequent explicit invocations. Validate
the complete selection before atomically appending its import events.

In S1, chunk complete lines at natural boundaries, targeting 1,800 Unicode characters
subject to the smaller model-token and existing bounded Evidence JSON budgets.
Use two-line overlap only when a block must be split. Preserve exact source text;
never normalize newlines, Unicode or indentation in cited content. Keep one-based
line start/end and canonical half-open character ranges. Break an overlong single
line into exact subranges rather than dropping its suffix. Bind every chunk to
source SHA-256, configured root alias and relative locator in private evidence.

Keep code as `text_chunk` with `TextOrigin.DOCUMENT`; line metadata belongs to
the versioned provider-owned code representation. Extend this profile's bounded
Evidence metadata with language, source hash, start/end lines, exact character
range and definition-focus line when supported. Search returns canonical snippet
and evidence ID; it never adds undeclared top-level fields or exposes absolute
paths. JSON-escape-aware chunk sizing must keep every Evidence section within its
existing 4 KiB limit. Definition focus comes from persisted, hashed source text;
search must not reopen the live source to find it.

S1 stores raw case-sensitive identifiers and qualified names separately from
normalized exact keys and snake_case/camelCase components. Index normalized
relative paths and path segments separately from body text, without indexing
absolute paths. Version Unicode/case and path-separator normalization; never
change the evidence text. Exact raw matches outrank normalized-only matches
within each identifier priority class. Split-term/path matches improve lexical
recall but never qualify as exact identifier matches. Fixed field weights and a
deterministic query classifier belong to the code policy. Path queries search
the path field without requiring the path to appear in the body; they cannot
select another corpus or open a live file.
For identifier queries, rank exact-definition matches, then exact uses, then other
hybrid matches; use existing weighted RRF within each class and unit ID for ties.
Version this code-only policy in the profile fingerprint. Natural-language code
queries retain ordinary hybrid RRF. A dedicated code orchestrator consumes public
candidate ports before top-k projection; do not reorder a truncated Search result
or claim the unchanged `TextSearchService` performs this policy.
Add a public code-specific identifier-candidate/metadata port scoped to the exact
pinned generation. Query its persisted normalized-identifier and definition index
with canonical filters before its independent 1,000-candidate bound. Merge those
candidates with lexical and dense candidates before class ordering, so a definition
outside ordinary lane cutoffs can still win. Metadata binds unit ID, matched
identifier, definition class/focus and policy version to retained source evidence.
The code orchestrator must not import the SQLite adapter's private methods.
S1 definition detection uses versioned lexical heuristics and marks uncertainty;
it makes no AST, type-resolution or call-graph claim. Retain distinct chunk hits for Agent use rather than
collapsing a repository file to its single best chunk.

## Optional structural enrichment (S2)

Add a lazily loaded, build-only Tree-sitter adapter behind a public code-parser
port. Initially support Python and JavaScript/TypeScript, including JSX/TSX with
matching grammars. Extract symbols, kinds, qualified scopes, signatures and
definition/body spans. Prefer symbol boundaries but split oversized symbols
under existing token/Evidence budgets, preserving parent-symbol identity. Retain
comments, imports, top-level statements and unparsed suffixes through exact-source
fallback chunks; AST selection must not discard non-symbol text.

Translate UTF-8 byte offsets to canonical Unicode character ranges and one-based
lines against retained source bytes. Validate boundaries, decorators, nested
scopes, CRLF and non-ASCII fixtures independently. Record parser/grammar identity,
parse status and fallback reasons in hashed provider metadata. Missing grammars,
syntax errors and parser resource exhaustion use deterministic bounded S1
fallback, visibly recorded at build and in evidence. Never label fallback facts
AST-derived. Authority, source-integrity and storage errors remain fatal. Parser
workers have deadlines, input/output bounds and cleanup; no project code executes.
Doctor checks dependency availability without loading a parser or acquiring a model.

## Optional relation expansion (S4)

Persist a small SQLite edge table rather than requiring a graph server or query
language. Start with containment, repository-internal imports and direct calls
whose targets are unambiguous under a versioned language resolver. Store source
and target symbol IDs, source site/range/hash, relation kind, extraction method
and resolution status. Ambiguous/dynamic/external targets remain unresolved;
never promote name similarity to a verified call edge or claim complete dispatch.
Cap each generation at 100,000 edge records and 32 MiB serialized relation
metadata, including unresolved records; overflow fails publication without
silently dropping facts or changing the active generation.

A generation-scoped public relation-candidate port may expand at most the first
five ranked seed chunks by one hop, with at most 20 unique added chunks total.
The startup policy fixes relation kinds/direction and limits. Apply canonical
filters to seeds, edges and target chunks before limits; deduplicate by unit ID,
bound visited nodes and stop on deadlines. Keep seed ranking ahead of expansion
candidates; place remaining non-seed base candidates after expansion, using
deterministic relation-kind, seed-rank and unit-ID ties for added chunks. Expansion
consumes the same final top-k and a 16 KiB UTF-8 snippet budget; drop whole tail
hits to fit, never truncate cited text. Existing per-Evidence limits still apply.
When top-k is filled by the retained seeds, add no expansion hits.
These limits are preview defaults, not measured capacity claims.

Search/Evidence read edge and unit payloads from the same pinned generation.
Missing required or corrupt edge data is fatal; an explicitly recorded unresolved
target yields no expansion. Attach relation provenance in bounded provider-owned
Evidence metadata, with no new canonical hit fields. Changed/deleted files require
recomputing affected derived facts before atomic publication; S4 may initially
rebuild all relations for correctness. No mutable live graph or automatic watcher
is introduced. Measure context benefit and false-positive amplification before
enabling S4; if benefit is unproven, retain S3 as the default.

## Security, compatibility and acceptance

Model assets and indexes remain outside the source tree. Normal logs/traces contain
neither full code/query text nor private paths, vectors or credentials. The source
denylist and secret heuristics reduce accidental intake; they are not proof that
arbitrary repository content is safe to redistribute. This task authorizes local
processing only, not upload or transfer of indexed code.

Maintain POSIX no-follow platform support for these profiles. Unsupported Windows
source-intake primitives fail readiness explicitly. Existing unrelated capabilities
retain their own platform support. Doctor/catalog inspection may validate config
and local manifests but never instantiate a model or start a subprocess.

Required acceptance covers failing-test-first development; deterministic ranking;
filtered dense recall; immutable evidence; compatibility; hostile inputs; installed
wheel CLI/MCP journeys; a separate real-model replay; independent review; and
`make quality`, `make test`, `make build`. Required skipped or blocked cases do not
pass. Human maintainer approval and semantic/capacity qualification remain distinct
from automated functional acceptance. See the linked acceptance matrix.

## References

- [RFC-0018](0018-offline-workspace-retrieval.md)
- [RFC-0008](0008-unified-text-retrieval-contracts.md)
- [RFC-0023](0023-unified-model-factory.md)
- [DigUp reference commit](https://github.com/ARahim3/DigUp/tree/d6f1f480427d151640196e4d6588906e1f3ee782)
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag/tree/main): structural
  extraction, deterministic relation queries and independent evaluation oracles.
  The inspected `main` pages were not pinned to one verified commit; pin the
  reference revision before implementation comparisons. Its fixture scores and
  component benchmarks do not establish Capicove quality or end-to-end speed.

## Implementation record

On 2026-10-23 the requester directed delivery of local code/file search in stages
S1-S4. This records acceptance for implementation; stage qualification follows the
acceptance matrix and is reported per stage.

S1 decisions as implemented:

- `capabilities.retrieval` schema 2 with exactly the twelve listed fields; the
  only selectable combination is `local-code-lexical-v1` + `code-symbols-v1` with
  null model fields and `max_vector_bytes = 0`. Schema-1 behavior is unchanged.
- `capicove code import --runtime-config PATH --root ALIAS --git GIT` adds an
  explicit absolute Git executable argument, because the closed configuration
  has no executable field and implicit `PATH` lookup is not allowed.
- `capicove index build` and `capicove source import|forget` dispatch to the
  code runtime for schema-2 configurations; `index build` verifies every live
  source against its imported hash, seals the generation and activates it.
- Generation format `capicove-local-hybrid-v1`, lexical variant `lexical-code-v1`:
  `cg-` content-derived IDs, a hash-pinned SQLite unit store with body, path and
  identifier-term FTS5 lanes, an identifier table, and `cev-` evidence IDs.
- Evidence metadata adds `relative_path`, `language`, `media_type`
  (`text/x-<language>`), line range, and every heuristic definition in the chunk
  (`definitions`, at most 16) plus the first as `definition_focus_*`. These are
  provider-owned metadata fields within the existing Evidence Tool sections.

S2 decisions as implemented: `code-ast-v1` is selectable only with
`local-code-lexical-v1`; Tree-sitter is the optional `code-ast` extra, imported only
at build time. Python, JavaScript/JSX and TypeScript/TSX symbols carry qualified scope,
kind, signature and decorators; chunk boundaries follow whole symbols, packing
adjacent symbols of one scope and splitting only oversized ones. Markdown and other
documentation formats are `not_applicable`; other languages, syntax errors and
budget overruns use recorded S1 line fallback. Evidence adds `structure`,
`parse_status`, `parse_reason` and per-definition `method`/`signature`.
`index build` reports `parse_outcomes`.

S3 decisions as implemented: `local-code-hybrid-v1` accepts `code-hybrid-v1` or
`code-relations-v1` and requires `model_binding: "code"` with a pinned model
configuration digest and a positive vector budget. The encoder is the verified
local EmbeddingGemma 2 GGUF on llama.cpp exposed as the text encoder
`embeddinggemma2-llamacpp-code` (requester direction; BGE-M3 assets were not
available locally). Document input is `path-header-v1`; vectors are verified per
row and scored by exact dot product in 1,024-row batches; dense and lexical lanes
fuse by weighted RRF after the S1 identifier classes. Only encoder-execution
unavailability degrades to lexical results with a warning; corruption, budget,
space and pin errors are fatal.

S4 decisions as implemented: `code-relations-v1` extracts containment, internal
import and unambiguous direct-call edges with a versioned Python resolver at build
time (unresolved targets are recorded, never guessed), caps 100,000 edges per
generation, and expands at most five seeds by one hop (`calls`, `contains`) with
at most 20 added units inside the final top-k and a 16 KiB snippet budget. Evidence
adds bounded `relations` provenance. S3 remains the default because no context
benefit threshold has been predeclared and met.

Evidence: acceptance matrix (internal record; public support status is documented in the capability guide).
