# 0014: Isolate Elasticsearch behind unified text retrieval contracts

- Status: Accepted
- Date: 2026-08-24
- Owners: Capicove maintainers

## Context

Capicove needs lexical and dense retrieval across document bodies, OCR output, and
ASR transcripts. SearcherKit demonstrates useful asynchronous Elasticsearch,
highlighting, concurrency, retry, and bulk-indexing patterns, but its flat document
model, provider-owned hybrid scores, direct dependencies, and destructive index
overwrite do not satisfy Capicove's immutable evidence, lineage, optional
dependency, and generation requirements.

The existing retrieval contract covers text queries but only image, keyframe, and
video-segment targets. Adding text targets and a networked persistent retriever
changes public discriminators, plugin compatibility, and the index trust boundary.

## Decision

Capicove uses one canonical `text_chunk` target for ordinary document text, OCR,
and ASR. A closed `TextChunkLocator` validates character coordinates plus required
OCR page or ASR time coordinates. Search hits return only a bounded plain-text
snippet; complete content and fine-grained coordinates remain evidence.

Elasticsearch is an independently distributed
`capicove-retriever-elasticsearch` plugin registered through
`capicove.retrievers`. Core contracts and retrieval orchestration do not import the
Elasticsearch SDK or accept provider DSL, endpoints, credentials, field names, or
index names.

One physical Elasticsearch index represents one immutable Capicove generation.
Capicove owns the authoritative manifest and lifecycle events. A validated
generation is activated atomically through an alias, and every search pins the
resolved physical generation before reading candidates. Active indexes are never
overwritten in place.

Elasticsearch supplies separate BM25 and dense KNN candidate lanes. Capicove owns
deterministic weighted reciprocal-rank fusion and final result projection. Raw
scores from incompatible lanes are retained for full diagnostics but are not
directly compared or added.

Plugin enumeration and availability checks remain side-effect-free. Network access
is an explicit runtime opt-in. Logs and normal traces exclude credentials, full
queries, complete chunks, vectors, and raw Elasticsearch responses.

## Consequences

- One search contract retrieves text regardless of whether it came from documents,
  OCR, or ASR.
- Other lexical/vector backends can implement the same candidate contract without
  emulating Elasticsearch scores or DSL.
- Elasticsearch deployment and SDK compatibility do not affect core installation
  or import paths.
- Index activation, query consistency, rollback, and lineage remain governed by
  Capicove rather than cluster convention.
- Building an Elasticsearch plugin requires explicit runtime configuration,
  compatibility metadata, contract tests, and security documentation.
- The first implementation must add more orchestration than SearcherKit's native
  hybrid query because fusion and generation pinning stay provider-neutral.

## Implementation qualification: explicit vector parameters (2026-10-19)

Native server qualification exposed an implementation mismatch with the existing
strict mapping identity: Elasticsearch expands omitted vector `index_options`
when reading a mapping back. The plugin now emits explicit unquantized HNSW
options (`type=hnsw`, `m=16`, `ef_construction=100`) and includes them in the
existing mapping digest. This implements the accepted immutable-generation
boundary without normalizing away server differences or accepting unknown fields.
Changing any of these options fails validation. No public manifest fields or
plugin interfaces are added. Pre-fix vector-bearing generations need rebuilding
under their new mapping identity; text-only mapping identities are unchanged.

The measured deployment is Elasticsearch 9.5.5 with the 8.19.3 compatibility
client, verified TLS and API-key authentication. HNSW is explicitly selected so
server quantization defaults do not change retrieval precision or mapping hashes.

## Alternatives considered

### Depend directly on SearcherKit

Rejected because it requires Python 3.12 and installs Agent, LLM, MCP, dataset, and
provider dependencies that are unrelated to Capicove's retrieval core. Its public
types do not preserve Capicove evidence and generation identities.

### Use Elasticsearch native hybrid scoring as the public score

Rejected because provider score behavior is not a portable or versioned Capicove
contract and does not safely compare raw BM25 and vector scores.

### Use separate document, OCR, and ASR target types

Rejected because production modality would leak into query planning, fusion, and
evidence access, multiplying otherwise identical behavior.

### Put locators in arbitrary metadata

Rejected because untyped metadata cannot enforce page/time invariants or provide a
stable serialized plugin contract.

### Delete and rebuild an active index

Rejected because it exposes partial state, prevents generation pinning and atomic
rollback, and breaks immutable index lineage.

## References

- [RFC-0008](../rfcs/0008-unified-text-retrieval-contracts.md)
- [ADR-0008](0008-multimodal-retrieval-boundaries.md)
- Design (archived development reference)
