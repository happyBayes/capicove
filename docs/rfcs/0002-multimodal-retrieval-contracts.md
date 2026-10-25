# RFC-0002: Versioned Multimodal Retrieval Contracts

- Status: Accepted
- Date: 2026-08-13
- Owners: Capicove maintainers
- Reverse-image Stage 2 amendment status: Pending review

## Summary

Capicove exposes one transport-neutral multimodal retrieval operation,
`capicove.search`, and one progressive-disclosure operation,
`capicove.get_evidence`. Search accepts exactly one text, image, or video-clip query
and retrieves one or more of image, keyframe, or video-segment targets. CLI and MCP
translate the same canonical request and result without embedding retrieval policy
or invoking providers directly.

The MVP runs one complete Capicove node per workspace. The node owns immutable
evidence, processing caches, provider composition, retrieval indexes, the
index-generation catalog, Tool Runtime, and traces. It may run locally with CPU
providers or on a GPU server with server-local CPU/GPU providers. A remote Agent
selects the GPU node by connecting to its authenticated MCP endpoint; a search
request never selects a host.

This RFC fixes the version-1 contract and compatibility rules required by the
accepted design. The capability boundaries and deployment decision are recorded in
[ADR-0008](../adr/0008-multimodal-retrieval-boundaries.md).

## Pending reverse-image Stage 2 amendment

This amendment is not accepted. It prepares the contract language needed for
code-owner review and does not authorize production implementation, activation,
merge, or release. Until that review is recorded, the previously accepted contract
outside this section remains authoritative and the internal research vertical is
not a public capability.

### Staged delivery boundary

The proposed Stage 2 delivery is limited to the offline `personal-lite` profile for
at most 10,000 independent images. It includes exact content equality, radius-two
perceptual matching, Agent-declared semantic intent, immutable local generations,
and equivalent Python, Tool, and MCP query behavior. It does not include keyframes,
text or video queries, live index mutation, remote workers, or a formal enterprise
capacity claim.

The optional `enterprise-disk` profile, FAISS/MIH adapters, and any 30-million-image
quality, latency, or capacity claim belong to a separately reviewed Stage 3. Stage
2 must not add placeholder enterprise dependencies or weaken its personal release
gate in anticipation of Stage 3.

### Agent-declared search intent

The amendment proposes a closed `SearchIntent` with the stable values `instance`,
`concept`, and `balanced`, plus this optional request field:

```text
search_intent: SearchIntent = balanced
```

`instance` requests the same person, object, shot, scene, composition, diagram, or
other visual arrangement. `concept` requests a shared topic, genre, style, mood, or
educational meaning. `balanced` executes both semantic spaces only when the caller
cannot support either narrower interpretation. The calling Agent selects the
intent from user context. Retrieval executes that declared intent and never runs a
server-side intent classifier or rewrites an explicit value after a failure.

### Perceptual hash identity

`HashSpaceRef` is a complete immutable identity containing:

- stable space, provider, provider-version, preprocessing, algorithm, and
  hash-version identifiers;
- fixed bit width and supported target modalities; and
- a schema version.

Reverse-image Stage 2 uses a 64-bit `phash-64-v1` space with a profile-owned maximum
Hamming distance of two. Equal bit width alone is not compatibility. Any provider,
provider version, preprocessing, algorithm, width, hash version, target-modality,
or schema change creates an incompatible space unless a later accepted contract
defines and tests compatibility. Request input cannot select a hash provider or
widen the Hamming radius.

### Immutable generations and lifecycle events

Each proposed `IndexGeneration` describes exactly one content-addressed lane and
binds the workspace, corpus, activation set, ordered unit-manifest digest, complete
embedding or hash space, target bundle-manifest digest when applicable, retriever,
index format, closed parameter-payload digest, immutable shard references, build
fingerprint, validation-report reference, and creation time. The
manifest contains no mutable lifecycle state.

If accepted, this subsection replaces the lifecycle-state field and mutable-state
transition list in the previously accepted `IndexGeneration` section below. Until
then, the difference remains visible for code-owner review rather than being
silently presented as an accepted compatibility change.

Lifecycle is represented by a versioned, append-only `GenerationLifecycleEvent`,
not by rewriting a manifest. Closed event kinds cover build, validation,
publication, qualification, activation, supersession, requalification, lease,
release, and eviction. Activation atomically replaces one versioned alias snapshot
for a complete compatible generation set. A query pins that full snapshot and its
qualification identity before opening readers; later activation cannot mix lanes
in the in-flight request. Candidate-evaluation leases may pin a published set but
cannot activate it or impersonate an active qualification.

### Stable outcomes

The amendment proposes closed `SearchFailureCode` and `SearchWarningCode` enums.
Stable primary failure codes are `source_not_authorized`, `permission_denied`,
`client_ingress_denied`, `invalid_query`, `stale_cursor`, `source_changed`,
`input_too_large`, `invalid_media`, `provider_unavailable`, `profile_unavailable`,
`capacity_exceeded`, `index_generation_unavailable`,
`embedding_space_incompatible`, `hash_space_incompatible`, `index_corrupt`,
`deadline_exceeded`, and `internal`.

Warnings are bounded supplemental outcomes for a successful but degraded request;
they never determine the primary Tool error. Stable warning codes are
`provider_unavailable`, `profile_unavailable`, `index_generation_unavailable`,
`embedding_space_incompatible`, `hash_space_incompatible`, `index_corrupt`,
`deadline_exceeded`, `retrieval_preview`, and `internal`. A failed result has
exactly one primary `SearchFailureCode`; a succeeded result has none. No hits is a
successful result.

### Trusted invocation authorization and same-object reads

Every Python, Tool, or MCP image search receives a host-created
`InvocationAuthorizationContext` after authentication and startup-policy
resolution. It binds at least the workspace, actor, operation, permission set,
authorization scope, privacy class, and permitted data-policy and source roots.
It is not serialized inside `SearchRequest`; request or transport parsing cannot
construct or replace it. Request fields may narrow this authority but
cannot grant or widen it. A request-provided `data_policy_ref`, artifact reference,
mount name, or path is a requested restriction or locator, never proof of
authorization.

For path and mount sources, the authorization boundary performs a
descriptor-relative open beneath an approved root with no-follow semantics on every
path component (or an equivalent platform primitive). It validates type, size,
identity, and policy, then hashes and decodes bytes from the same opened object. It
must not authorize a canonical path and reopen that path later. Symlinks, devices,
directories, replacement races, unsupported platform guarantees, and identity
changes fail closed as `source_not_authorized` or `source_changed` before provider
dispatch.

## Scope

Version 1 covers:

- unified text, image, and video-clip queries;
- image, keyframe, and temporal video-segment targets;
- near-duplicate and semantic image-to-image search;
- CPU text-to-keyframe semantic retrieval independent of OCR and ASR;
- direct or assembled video-segment retrieval;
- compact and full result projections;
- progressive evidence expansion;
- processing, embedding-space, searchable-unit, and index-generation identity;
- provider, score, cursor, degradation, and failure semantics;
- server-relative CPU/GPU execution selection;
- remote MCP client-ingress policy; and
- incremental processing and rebuildable-index lifecycle compatibility.

Version 1 excludes index synchronization, cross-node generation mapping, remote
Retriever dispatch, and federated candidate fusion. Enterprise federation is a
Post-MVP capability and requires a separate accepted RFC and ADR.

## Canonical operations

| Surface | Search | Evidence expansion | Explicit ingest |
| --- | --- | --- | --- |
| Tool ID | `capicove.search` | `capicove.get_evidence` | `capicove.ingest` |
| CLI | Reserved after Stage 2 | Reserved after Stage 2 | Reserved after Stage 2 |
| MCP | `capicove_search` | `capicove_get_evidence` | `capicove_ingest` |

`capicove.ingest` is defined with the GPU-server ingress boundary before remote
media ingest is implemented. Index build, validation, activation, recovery, and
eviction are server-local administrative operations and are not Agent MCP tools in
the MVP.

The table retains the existing text search identity. The pending Stage 2 image
amendment uses Python, `capicove.search.image`, and `capicove_search_image`, with
separate argument, authorization, and result handling. Earlier draft image clients
must migrate to these image-specific names rather than replacing the text Tool or
MCP handler. The draft implementation additionally exposes query-only
`capicove image search` and `capicove image serve` (local MCP stdio) for provisioned,
activated image indexes. It does not implement evidence expansion or explicit
ingest. These image names remain subject to maintainer
acceptance of this amendment.

## Search query contract

`SearchQuery` is a closed tagged union. Exactly one variant is present:

```text
SearchQuery = TextQuery | ImageQuery | VideoClipQuery
```

The stable discriminator values are `text`, `image`, and `video_clip`.

### TextQuery

```json
{
  "type": "text",
  "text": "a cyclist crossing a rainy street",
  "language": "en"
}
```

`text` is non-empty after validation. `language` is optional. A query contains no
provider-specific model, tokenizer, prompt, endpoint, or index parameter.

### ImageQuery

```json
{
  "type": "image",
  "source": {
    "type": "artifact",
    "artifact_ref": "artifact_123"
  }
}
```

The source is exactly one of an artifact reference, an authorized node-local path,
an authorized mount reference, or bounded inline bytes. Transport policy decides
which source variants are accepted. Every accepted source is authorized, bounded,
content-typed, hashed, and checked for mutation before provider dispatch.

### VideoClipQuery

```json
{
  "type": "video_clip",
  "source": {
    "type": "artifact",
    "artifact_ref": "artifact_456"
  },
  "time_range": {
    "start_ms": 12000,
    "end_ms": 18000
  }
}
```

`time_range` is optional and uses the same temporal contract as results. Inline
image and video-clip query payloads have a 4 MiB decoded-byte ceiling in the MVP.
Larger media uses an authorized node mount or explicit ingest.

Query assets are ephemeral unless a separate authorized `capicove.ingest`
operation creates immutable evidence.

## SearchRequest

The version-1 request contains:

| Field | Type and rule |
| --- | --- |
| `query` | Required `SearchQuery` tagged union |
| `targets` | Non-empty set of `image`, `keyframe`, `video_segment` |
| `filters` | Versioned, closed `SearchFilters`; defaults empty |
| `top_k` | Integer 1–100; defaults to 20 |
| `cursor` | Optional opaque continuation cursor |
| `detail_level` | `compact` or `full`; defaults to `compact` |
| `retrieval_profile_ref` | Runtime-owned profile identity |
| `execution_profile` | `auto`, `cpu`, or `gpu`; defaults to `auto` |
| `data_policy_ref` | Runtime-owned ingress/privacy/retention identity |
| `deadline` | Absolute deadline inherited from `InvocationContext` |

`SearchFilters` contains only explicitly versioned corpus, artifact, media, and
temporal predicates. Adapters translate those fields; callers cannot supply raw
database predicates.

Profile references are configuration identities. Query input cannot name a model,
provider, device, host, URL, credential, filesystem root, subprocess flag, index
path, or index generation.

`execution_profile` is relative to the node handling the request:

- `auto` prefers an authorized compatible GPU provider when available and may fall
  back to a compatible CPU provider inside the same node and pinned generation;
- `cpu` requires a CPU provider; and
- `gpu` requires a GPU provider and fails when none is ready.

It never means local host versus remote host.

## Temporal coordinates

All temporal coordinates are integer milliseconds and use a non-empty half-open
interval:

```text
[start_ms, end_ms)
```

An image has no temporal locator. A keyframe has `timestamp_ms` and may reference a
containing scene range. A video segment has a non-empty `time_range`.

## ProcessingFingerprint

`ProcessingFingerprint` is the SHA-256 digest of a normalized canonical JSON
manifest containing every input that can change a derived output:

- source content hash and ordered upstream representation hashes;
- processor ID and processor contract version;
- implementation and processing version;
- normalized configuration hash;
- provider and model revision when applicable;
- preprocessing version;
- output schema version; and
- deterministic or nondeterministic execution declaration.

Canonical JSON uses sorted object keys, compact separators, UTF-8, and rejects NaN
and infinity. Lists remain ordered. Changing only a downstream embedding provider
does not invalidate compatible decode, scene-metric, or keyframe layers. Layered
cache keys include `source_hash`, layer-specific `processing_version`, normalized
configuration, and upstream representation hashes.

## EmbeddingSpaceRef

`EmbeddingSpaceRef` contains:

- stable `space_id`;
- provider ID and pinned model revision;
- preprocessing version;
- vector dimension;
- normalization method;
- distance metric;
- supported query modalities;
- supported target modalities; and
- vector schema version.

A query representation may search an index only when its complete space reference
is compatible. Equal dimensions are insufficient. A model, tokenizer,
preprocessing, normalization, metric, dimension, or vector-schema change creates a
new space unless an accepted provider contract explicitly declares and tests
compatibility.

Raw vectors are derived sensitive data. They are never returned in search results
or normal traces.

## SearchableUnit

`SearchableUnit` is immutable metadata for one retrievable unit and contains:

- `unit_id` and unit type (`image`, `keyframe`, or `video_segment`);
- immutable `source_hash`;
- artifact and evidence references;
- representation references;
- optional temporal locator;
- processing-run reference;
- `processing_version`;
- `processing_fingerprint`; and
- zero or more `EmbeddingSpaceRef` values.

Vectors remain in addressed derived representations or immutable index shards.
Mutable eviction state does not belong to `SearchableUnit`; cache and generation
catalogs own lifecycle state.

## IndexGeneration

An immutable `IndexGeneration` manifest contains:

- generation ID, workspace ID, and corpus identity;
- unit-manifest digest and embedding-space references;
- retriever and index-format identity;
- ordered immutable shard references;
- calibration profile;
- build processing fingerprint;
- creation timestamp;
- validation report reference; and
- lifecycle state.

Valid lifecycle transitions are:

```text
building -> ready -> active -> superseded -> evicted
```

Activation is a separately recorded atomic alias event. An immutable manifest is
not rewritten to impersonate another state or generation. A request pins all
generation aliases at query start, and an active-alias switch cannot mix old and
new shards in that request. Eviction may remove validated rebuildable payloads but
retains the manifest, hashes, lineage, and lifecycle events.

## Query representations and provider ports

The public retrieval port accepts this closed representation union:

```text
QueryRepresentation = QueryVector | PerceptualHashQuery | LexicalQuery
```

`QueryEncoder` produces query representations. Processing-owned
`MediaRepresentationEncoder` implementations produce target image, keyframe, or
segment representations. Even when one plugin distribution supplies both, neither
capability imports the other's private adapter.

`CandidateRetriever` searches one pinned generation and returns bounded candidate
identities with raw score metadata. Fusion compares ranks or explicitly calibrated
normalized scores; it never adds or directly compares raw scores from incompatible
lanes.

Plugin metadata declares contract range, roles, modalities, embedding spaces,
model/preprocessing revisions, score semantics, CPU/GPU execution modes, resource
requirements, native/subprocess behavior, privacy classes, and deterministic,
cache, and retry behavior. Discovery and availability checks cannot load models,
inspect media, open the network, start a process, allocate an accelerator, or
download assets.

## Search result contract

`SearchResult` contains a status, query ID, degraded flag, warning codes, hits, and
optional continuation cursor. No hits is a successful result.

### Compact projection

Compact is the default Agent projection. A compact video-segment hit contains only:

```json
{
  "id": "unit_123",
  "target_type": "video_segment",
  "time_range": {
    "start_ms": 125000,
    "end_ms": 139000
  },
  "brief_score": 0.87,
  "evidence_id": "evd_789"
}
```

A keyframe uses `timestamp_ms` instead of `time_range`; an image has no temporal
field. `brief_score` is finite and within `[0, 1]`, and is a versioned normalized
ranking value rather than a probability. The compact top-level envelope includes
only query ID, status, degraded flag, short warning codes, hits, and next cursor.
Twenty compact segment hits must serialize within the configured 16 KiB regression
ceiling.

### Full projection

Full hits may additionally contain artifact/searchable-unit references,
representation/scene/keyframe references, match kinds, raw and normalized lane
scores, embedding spaces, processing fingerprint, index-generation IDs, provider
and execution identities, detailed warnings, and lineage references.

Each lane score records lane identity, raw provider score, normalized score, score
semantics, calibration version, and rank. Full results still never inline raw
vectors, complete media, complete OCR/ASR documents, complete lineage graphs,
credentials, or private provider diagnostics.

The complete internal result and retrieval trace are recorded independently from
the selected projection, so compact output never reduces lineage or audit data.

## Progressive evidence disclosure

The canonical Agent flow is:

```text
capicove_search(detail_level=compact)
    -> Agent selects one or more evidence IDs
    -> capicove_get_evidence(evidence_id, requested sections)
```

Evidence expansion supports bounded metadata, preview resource references,
transcript/OCR excerpts, score explanation, processing lineage, and provider/index
diagnostics. Access is permission-checked and section-bounded. Normal results never
inline base64 media.

## Cursor compatibility

A cursor is opaque, signed canonical JSON bound to:

- normalized query hash without raw query content;
- target/filter identity;
- retrieval and data-policy profile identities;
- pinned generation IDs;
- a digest of the complete bounded ranking and lane outcome state;
- last stable sort key;
- page size; and
- expiry.

A modified, expired, query-mismatched, profile-mismatched, generation-mismatched,
or lane/ranking-state-mismatched cursor fails with `stale_cursor`; it never
silently restarts pagination. Cursor signing keys are runtime credentials and never
appear in query input or traces.

## Degradation and failures

If a profile permits degradation and at least one required candidate path succeeds,
an unavailable optional lane yields `status=succeeded`, `degraded=true`, and a
stable warning such as `retrieval_lane_unavailable`. All required lanes failing
yields a failed Tool result. Partial timeout results are allowed only when the
profile permits them and candidate generation completed.

Stable capability failures map to Tool Runtime codes as follows:

| Retrieval failure | Tool error |
| --- | --- |
| `invalid_query` | `invalid_input` |
| `source_changed` | `invalid_input` |
| `stale_cursor` | `invalid_input` |
| `permission_denied` | `permission_denied` |
| `client_ingress_denied` | `policy_rejected` |
| `embedding_space_incompatible` | `unavailable` |
| `provider_unavailable` | `unavailable` |
| `index_generation_unavailable` | `unavailable` |
| `index_corrupt` | `internal` |
| `gpu_provider_retryable` | `retryable` |
| `deadline_exceeded` | `timeout` |

An invalid or corrupt active generation cannot fall through to an unvalidated
building generation. Explicit `gpu` never falls back to CPU. `auto` fallback stays
inside one node, one workspace, and the same pinned generation.

## Client ingress and security boundary

Remote MCP media/vector disclosure is governed independently from execution
selection. The closed version-1 ingress modes are:

| Mode | Maximum remote query payload |
| --- | --- |
| `deny_remote` | Text only; no media or vectors |
| `embedding_only` | Text or a compatible derived vector |
| `redacted_media` | Text, compatible vector, or lineage-bound redacted media |
| `raw_media` | Text, compatible vector, redacted media, or bounded raw media |

The configured policy evaluates authenticated actor, listener, workspace,
operation, privacy class, modality, size, retention, and whether the payload is an
ephemeral query or explicit ingest. Tool arguments cannot widen the active mode.
Vectors remain privacy-classified and require exact `EmbeddingSpaceRef`
compatibility. Redacted media requires its processing fingerprint and lineage.

Non-loopback MCP listeners are explicit startup opt-ins and require encrypted
transport and authentication. Listener addresses, TLS files, credentials,
workspace/model/index paths, and authorized mounts come from startup configuration,
never Tool arguments. Credentials, raw media, vectors, and complete sensitive text
are excluded from logs and traces.

## Persistence and compatibility

These are compatibility-controlled persistent or public surfaces:

- query, request, result, error, warning, and evidence schemas;
- `ProcessingFingerprint` canonicalization;
- `EmbeddingSpaceRef` compatibility;
- `SearchableUnit` and representation manifests;
- cache/checkpoint and `IndexGeneration` manifests;
- cursor payload semantics;
- plugin metadata and entry-point groups;
- client-ingress, authentication, privacy, and retention boundaries; and
- compact/full projection meaning.

Adding an optional field with a documented default may be backward compatible.
Changing a discriminator, required field, field meaning, temporal unit, hash
canonicalization, score meaning, lifecycle transition, plugin role, or security
boundary requires a new schema/contract version and compatibility review. Adding
or renaming an entry-point group, broadening automatic disclosure, or enabling a
new credential/listener class requires another accepted RFC and ADR.

Provider implementation changes that preserve these contracts, pass provider
contract tests, and create correctly versioned processing/index generations do not
change the public contract.

## Acceptance and delivery gates

Implementation proceeds in independently reviewable stages: contracts and local
keyframes; local temporal retrieval and Agent interfaces; incremental lifecycle;
then single-node GPU-server providers and authenticated MCP. Every behavior begins
with a failing test. Each stage must pass `make quality`, `make test`, and
`make build` before completion.

Model weights, generated indexes, credentials, source media, and user content are
never committed. Pull-request tests use authored deterministic fixtures and no
implicit model download or external network call.

## References

- [ADR-0008: Establish multimodal retrieval boundaries](../adr/0008-multimodal-retrieval-boundaries.md)
- Approved multimodal retrieval design (archived development reference)
- Implementation program (archived development reference)
- [Module boundaries](../architecture/module-boundaries.md)
