# RFC-0008: Unified Text Retrieval Contracts

- Status: Accepted
- Date: 2026-08-24
- Owners: Capicove maintainers

## Local adapter amendment (2026-10-13)

The operator selected BGE-M3 for local embedding. Its startup profile explicitly
selects `model_family="bge-m3"`: XLM-R, CLS pooling, L2 normalization, and a
separate 1024-dimensional dense space. Existing BERT profiles retain their
preprocessing identity. Sparse and ColBERT outputs are outside this amendment.
The public text encoder contract and search request schemas remain unchanged.

Acquisition uses the official `BAAI/bge-m3` model repository at immutable revision
`5617a9f61b028005a4858fdac845db406aefb181` and a packaged file/hash allowlist.
The operator-only provisioner verifies upstream bytes before CPU conversion with
`torch.load(weights_only=True, mmap=True)` and safetensors serialization. Runtime
continues to reject pickle weights, dynamic code, network fallback and silent
truncation. A version-1 local provisioning receipt records upstream identities,
license, dense-only scope, conversion versions, and the derived file inventory;
the operator supplies its trusted inventory digest at startup. This receipt is
not a new shared-host model installation descriptor.

## Summary

Capicove retrieval contract version 2 adds one `text_chunk` target that represents
ordinary document text, OCR text from images and PDFs, and ASR transcripts from
audio and video. Text chunks retain immutable evidence and processing lineage and
use a closed locator contract for character, page, and temporal coordinates.

Elasticsearch is an optional retriever plugin. It may provide lexical and dense
candidate lanes, but provider configuration, query DSL, index names, and raw score
fusion do not enter canonical search contracts.

## Scope

Version 2 adds:

- `TargetType.TEXT_CHUNK`;
- `MatchKind.LEXICAL`;
- `TextOrigin`, `CharacterRange`, and `TextChunkLocator`;
- text locators on searchable units and text hits;
- bounded text snippets on compact and full text hits;
- `SearchFilters.text_origins` with filter schema version 2; and
- a retrieval contract version exported for plugin compatibility checks.

This RFC defines Elasticsearch generation, candidate, fusion, error, and security
semantics required by later implementation stages. It does not make Elasticsearch
a core dependency or add provider parameters to `SearchRequest`.

## Canonical text coordinates

`CharacterRange` uses zero-based, non-empty half-open coordinates into one
immutable text representation:

```text
[start_char, end_char)
```

`TextOrigin` has the stable values `document`, `ocr`, and `asr`.
`TextChunkLocator` has one origin, one character range, and closed origin-specific
coordinates:

- `document` forbids page and temporal coordinates;
- `ocr` requires a positive one-based page number and forbids temporal coordinates;
- `asr` requires a non-empty millisecond time range and forbids page coordinates.

OCR chunks do not cross pages. ASR chunks do not contain discontinuous time spans.
Fine-grained OCR boxes and ASR word timing remain in referenced evidence.

## Searchable units and hits

A `SearchableUnit` with type `text_chunk` requires a `TextChunkLocator`. Other unit
types forbid it. Existing immutable source hash, evidence, representation,
processing, fingerprint, and embedding-space identities retain their RFC-0002
meanings.

A text `CompactHit` requires the same locator plus a non-blank plain-text snippet
of at most 2,048 Unicode code points. Non-text hits forbid locator and snippet.
Complete source text is not returned by search; Agents use the existing
permission-checked evidence-expansion operation.

`SearchFilters.text_origins` is an immutable set of `TextOrigin` values. Filter
schema version 2 keeps the existing corpus, artifact, media-type, and temporal
predicates. Temporal filtering applies to ASR locator intervals.

## Contract compatibility

`RETRIEVAL_CONTRACT_VERSION` is 2. Existing v1 discriminator meanings remain
unchanged. A plugin or transport must declare version-2 compatibility before it
accepts or emits text targets, lexical match kinds, text locators, snippets, or
origin filters.

The Elasticsearch retriever plugin declares retrieval contract version 2 through
the existing `capicove.retrievers` entry-point group. Core Capicove does not import
the Elasticsearch SDK.

## Elasticsearch generation semantics

One immutable physical Elasticsearch index corresponds to one Capicove
`IndexGeneration`. Capicove retains the authoritative manifest and lifecycle
events. A generation is built and validated before one atomic alias operation can
activate it. Search pins the resolved physical generation before reading either
lexical or dense candidates.

In-place overwrite of an active physical index is prohibited. A corrupt or invalid
generation cannot fall back to a building or unvalidated generation. Superseded
derived payloads may be evicted without removing manifests, hashes, lineage, or
lifecycle events.

## Candidate and fusion semantics

Elasticsearch BM25 and dense KNN are independent candidate lanes. Each lane
returns unit identity, raw score, score semantics, rank, provider identity, and
pinned generation identity.

Capicove owns deterministic weighted reciprocal-rank fusion version `rrf-v1`.
Fusion does not add or compare raw BM25 and vector scores. It returns a normalized
score within `[0, 1]` and uses stable unit identity as the final tie breaker.
Retrieval profiles own weights, candidate bounds, and required/optional lane
policy; requests cannot override them.

An unavailable optional dense lane may yield successful degraded lexical results.
A required lane failure fails the operation.

## Provider and security boundary

Plugin discovery and availability checks are side-effect-free and do not connect
to Elasticsearch. Endpoint, TLS, credentials, physical index naming, aliases,
timeouts, and provider tuning are startup configuration, not query input.

Adapters translate only canonical filter fields. Raw DSL, scripts, arbitrary field
names, index names, credentials, and endpoints are rejected from public search
input. Credentials, full queries, complete chunks, vectors, and raw provider
responses are excluded from logs and normal traces.

## Stable failures

Version 2 uses the RFC-0002 failure mapping and explicitly applies:

- `invalid_query` to invalid text, locators, snippets, and origin filters;
- `index_generation_unavailable` to missing or inactive pinned generations;
- `embedding_space_incompatible` to dense-space mismatches;
- `provider_unavailable` to an unavailable Elasticsearch provider;
- `deadline_exceeded` to invocation deadline exhaustion; and
- `index_corrupt` to manifest, mapping, or content-digest mismatches.

## Acceptance gates

Contract implementation begins with failing tests for enum values, coordinate and
origin invariants, text-unit and text-hit requirements, snippet bounds, filters,
public exports, and existing-target regressions. Canonical contracts must remain
standard-library-only and must not import Elasticsearch.

Each later stage adds its own contract, unit, integration, and packaging tests.
Every stage must pass `make quality`, `make test`, and `make build`.

## References

- [ADR-0014](../adr/0014-elasticsearch-text-retrieval-boundaries.md)
- [RFC-0002](0002-multimodal-retrieval-contracts.md)
- Approved design (archived development reference)
