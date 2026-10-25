# RFC-0006: Agent Multimodal Trace JSONL

- Status: Accepted
- Date: 2026-08-18
- Owners: Capicove maintainers

## Summary

Capicove will record and export observable Agent operation trajectories as a
versioned, model-neutral JSONL package. The format preserves visible messages,
Tool calls and results, image/video/audio references, produced artifacts, batch
facts, errors, warnings, timing, feedback, and lineage. It is source material for
downstream cleaning; it is not a model-specific SFT, preference, or Agentic RL
format.

The version 1 format is event-per-line. A complete Episode is reconstructed by
grouping on `episode_id` and ordering by `sequence`. Records form a per-Episode
SHA-256 predecessor chain. Export packages contain bounded JSONL shards, the exact
versioned JSON Schema, and one canonical manifest. Media bytes remain referenced by
default and MAY be copied explicitly into a content-addressed bundle.

## Scope

Implementations conforming to version 1:

- MUST record only visible messages and observable operation facts;
- MUST NOT record hidden chain of thought, private scratchpads, logits,
  credentials, raw base64 media, or unredacted absolute paths;
- MUST preserve immutable source evidence identities, hashes, and lineage;
- MUST keep raw records append-only and represent feedback as linked records;
- MUST keep recording, exporting, and media copying local and explicitly enabled;
- MUST NOT fetch remote media while recording, exporting, or validating; and
- MUST NOT introduce model-specific `<image>`, `<video>`, or `<audio>` tokens.

Downstream systems MAY transform validated packages into model-specific examples.
Such transformation is outside this RFC.

## Version 1 record envelope

Every JSONL line is one closed record object with these fields:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `schema_version` | integer constant `1` | yes | Envelope schema version |
| `record_type` | string enum | yes | Payload discriminator |
| `record_version` | integer constant `1` | yes | Record-type version |
| `record_id` | non-empty string | yes | Immutable record identity |
| `episode_id` | non-empty string | yes | Episode identity |
| `trace_id` | non-empty string | yes | Distributed trace identity |
| `sequence` | non-negative integer | yes | Contiguous Episode order |
| `occurred_at` | UTC timestamp | yes | Observable event time |
| `payload` | typed object | yes | Record-specific body |
| `modalities` | unique array | yes | `image`, `video`, or `audio` |
| `step_id` | string or null | yes | Optional host step identity |
| `request_id` | string or null | yes | Tool request identity |
| `parent_record_id` | string or null | yes | Optional causal parent |
| `media` | array | yes | Media references used by the record |
| `lineage` | object or null | yes | Additional captured lineage |
| `warnings` | array | yes | Stable visible warnings |
| `error` | object or null | yes | Canonical visible error |
| `privacy_class` | string enum | yes | Canonical privacy class |
| `previous_record_sha256` | SHA-256 or null | yes | Previous canonical record digest |

Every version 1 object, including nested objects, MUST reject unknown properties.
All identifiers and human-readable required fields MUST be non-blank. Python
booleans MUST NOT be accepted where an integer is required.

## Record types and payload matching

Version 1 defines exactly these record types:

| `record_type` | Required payload | Purpose |
| --- | --- | --- |
| `episode.started` | `EpisodeStarted` | Immutable Episode context |
| `message.observed` | `MessageObserved` | Visible ordered content blocks |
| `tool.call` | `ToolCallObserved` | Sanitized Tool arguments |
| `tool.result` | `ToolResultObserved` | Canonical Tool outcome |
| `artifact.produced` | `ArtifactProduced` | Derived artifact and media link |
| `batch.item` | `BatchItemObserved` | Terminal batch execution fact |
| `episode.feedback` | `EpisodeFeedback` | Linked post-hoc annotation |
| `episode.completed` | `EpisodeCompleted` | Terminal Episode state |

The payload class MUST match `record_type`; decoders MUST reject a mismatch rather
than infer another type. `tool.call` and `tool.result` MUST include a non-blank
`request_id`. A successful Tool result MUST NOT carry an error. A failed or rejected
Tool result MUST carry an error. Tool result identity MUST match the corresponding
open call's request ID, Tool ID, and Tool version.

An Episode MUST begin with exactly one `episode.started` at sequence zero. Sequence
numbers MUST then be contiguous. An Episode MAY end with exactly one
`episode.completed`. Ordinary events MUST NOT follow completion; linked
`episode.feedback` records MAY follow completion. A completed Episode MUST NOT have
a dangling Tool call.

## Payload fields

### Episode start

`episode.started` records capture `goal`, `workspace_id`, `actor_id`,
`environment_id`, a positive `environment_version`, `tool_catalog_sha256`, a closed
mapping of active asset versions, and optional Agent lineage. Agent lineage states
whether identity was `captured`, `not_observed`, or `redacted`. Captured lineage MUST
contain at least one Agent, model, revision, generation-parameter, or prompt-template
identity.

### Visible messages

`message.observed` contains a role (`system`, `user`, or `assistant`) and a non-empty
ordered array of content blocks. A text block stores visible sanitized Unicode text.
A media block stores only a `media_id`. The order of text and media blocks is
semantic and MUST be preserved.

### Tool calls and results

`tool.call` contains Tool ID, positive contract version, sanitized JSON arguments,
the input-schema SHA-256, and a positive attempt number. `tool.result` contains the
same Tool identity, canonical status, output-schema SHA-256, an optional JSON result,
evidence references, and finite numeric metrics.

Arguments and results MUST contain finite JSON values only. Decoders MUST reject
bytes, non-string object keys, cycles, non-finite numbers, and excessive nesting.
Raw Tool arguments MAY be passed to the Tool implementation but MUST NOT bypass the
sanitizer before persistence.

### Artifacts, batch facts, feedback, and completion

`artifact.produced` links an immutable artifact reference to one media identity and
optional derivation parents. `batch.item` stores the plan/item identity, terminal
state, attempts, cache-hit fact, modality, media type and size, source SHA-256, cache
key, and bounded detail without a source path.

`episode.feedback` contains a kind, closed JSON value, provenance, and optional
target record. Feedback never mutates the target. `episode.completed` contains one
of `succeeded`, `failed`, `rejected`, or `incomplete`, plus an optional visible
outcome.

## Media references and locators

One media reference contains a non-blank media ID, modality, media type, SHA-256,
non-negative size, optional evidence reference, optional locator, optional time
selection, optional strictly increasing frame indices, and unique derivation IDs.
Both time bounds MUST be present together, finite, non-negative, and ordered.

Locator kinds are:

| Kind | Meaning |
| --- | --- |
| `artifact` | Host-owned immutable artifact reference |
| `workspace_relative` | Path relative to an explicit workspace root |
| `bundle` | Path relative to an export package |

Locator values MUST be relative, normalized POSIX paths. They MUST NOT be absolute,
empty, contain `..`, or resolve through a symlink. An exporter MUST NOT download a
locator. Media references in records remain unchanged when bytes are copied, so
record hashes are stable.

## IDs and append idempotency

The store assigns sequence numbers transactionally. A caller supplies one stable
idempotency key per observable event. In version 1, draft equality covers every
field except `occurred_at`: a retried append keeps the timestamp of the first
persisted observation and returns that existing record. Reusing a key when any
other draft field differs MUST fail. Excluding only the retry-local timestamp
preserves the original observable time while detecting semantic key reuse.

The version 1 SQLite adapter derives a record ID as:

```text
"record:" + sha256(episode_id + NUL + decimal_sequence + NUL + idempotency_key)
```

This derivation is deterministic but does not expose the idempotency key. Other
adapters MUST preserve the same public identity semantics for a given append.

## UTC timestamps

Timestamps MUST be timezone-aware and serialized in UTC as exactly:

```text
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

Six fractional digits and the trailing `Z` are required. A decoder MUST reject a
naive timestamp, offset timestamp, missing fractional digits, or leap/invalid date.

## Canonical JSON and record hashes

Canonical JSON uses UTF-8 and the equivalent Python parameters:

```python
json.dumps(
    value,
    sort_keys=True,
    ensure_ascii=False,
    allow_nan=False,
    separators=(",", ":"),
).encode("utf-8")
```

Canonical record bytes contain no trailing line feed. A JSONL shard appends exactly
one `LF` byte after each canonical record. `record_sha256` is the lowercase SHA-256
of canonical record bytes.

At sequence zero, `previous_record_sha256` MUST be null. At sequence `N > 0`, it
MUST equal the canonical record digest of sequence `N - 1`, including that prior
record's own predecessor field. Feedback after completion continues the same chain.

## Shards

Shard names are contiguous, zero-padded POSIX paths:

```text
records-00000.jsonl
records-00001.jsonl
```

The exporter rotates before a next record would exceed the configured record or
byte limit. One individually valid record MAY occupy a shard alone. A record above
the configured maximum record size MUST fail the export. Episode boundaries need not
align to shard boundaries; shard order and line order MUST preserve
`(episode_id, sequence)` order.

Every shard manifest entry contains path, record count, exact byte size, and
SHA-256. A zero-record export has no shards.

## Manifest

`manifest.json` is canonical JSON followed by one `LF`. Version 1 contains:

- `format_version` and `schema_version`, both `1`;
- `export_id` and UTC `created_at`;
- normalized selection and sharding options;
- ordered shard entries;
- total record count and record-type, modality, Tool-status, and Episode-status
  counts;
- included, incomplete, and omitted-restricted Episode counts;
- unique media-ID count, unique-content byte count, and missing-media count;
- `media_layout`, fixed as `sha256-v1`;
- the relative schema path and schema SHA-256.

The total record count MUST equal both the sum of shard counts and the sum of
record-type counts. Count maps use non-negative integers. Paths MUST be relative
POSIX paths and MUST NOT escape the package.

The package includes the exact schema at:

```text
schemas/capicove-agent-trace-v1.schema.json
```

## Media copy mode

Reference mode is the default and MUST NOT access media bytes. Copy mode requires an
explicit local resolver and MUST resolve and verify every referenced medium before
publication. Copy mode stores bytes at:

```text
media/sha256/<first-two-hex>/<sha256>.<fixed-extension>
```

Extensions come only from a fixed media-type allowlist; unknown types use `.bin`.
The copier MUST stream, hash, verify size and digest, reject symlinks and path
escape, fsync temporary content, and publish atomically. Equal content hashes MUST
deduplicate. Conflicting size or media identity for one hash MUST fail. Source bytes
MUST NOT be rewritten, moved, or deleted.

## Episode selection and incomplete Episodes

Filters select whole Episodes. Modality, Tool ID, and Tool-status predicates match
by existence; when one record matches, every record in that Episode is returned.
Start and completion time predicates apply to Episode timestamps. Episode status
applies to terminal state.

Incomplete Episodes are excluded by default. When explicitly included, they remain
without a fabricated completion record and the manifest counts them. A validator
reports their missing completion as a warning only when the manifest options permit
incomplete Episodes. Restricted Episodes are excluded as a whole unless explicitly
included. A read snapshot MUST keep its selection summary and record iterator
consistent under concurrent appends.

## Sanitization and privacy

The Tool-spec sanitizer applies declared dotted redaction paths, including
single-array-element `*` wildcards. Secret-looking object keys are redacted
case-insensitively at every depth. Base64 media is validated, bounded, replaced by a
size/hash descriptor or resolved media reference, and never persisted as bytes.

Visible text preserves ordinary Unicode while replacing high-confidence bearer
tokens, OpenAI-style secret keys, AWS access key IDs, absolute POSIX paths, and
drive-qualified Windows paths with fixed markers. Error records MUST expose only
stable public codes and bounded public messages; unexpected exception text and
stacks MUST NOT be persisted.

Recording is disabled when no recorder is configured. When configured, sanitizer
or storage failure MUST prevent unrecorded Tool execution if it happens before the
Tool call and MUST surface after execution if result persistence fails. Recording
MUST NOT silently degrade.

## Compatibility and rejection behavior

Readers MUST reject unsupported schema/record versions, unknown record types,
unknown fields, invalid payload discrimination, invalid enum values, invalid
timestamps/hashes/paths, non-canonical semantic values, and lifecycle violations.
Writers MUST NOT emit an extension field into version 1. A future compatible
extension requires a new record or schema version and compatibility review.

Validators report stable machine-readable issue codes and bounded public messages.
They MUST detect manifest, schema, shard, canonical JSON, identity, sequence,
predecessor-chain, Tool pairing, lifecycle, derivation, and copied-media tampering.
They MUST retain bounded memory by spilling cross-record state to local temporary
storage and MUST remove that storage after validation.

## Complete mixed-modal fixture

The following abbreviated but complete five-record Episode demonstrates ordered
image/video/audio references. SHA-256 values use fixed fixture digests; the
predecessor values are illustrative and are recomputed from canonical bytes by a
conforming implementation.

```json
{"error":null,"episode_id":"episode-demo","lineage":null,"media":[],"modalities":["image","video","audio"],"occurred_at":"2026-08-18T12:00:00.000000Z","parent_record_id":null,"payload":{"active_asset_versions":{},"actor_id":"user-demo","agent_lineage":{"agent_id":"agent-demo","agent_version":"1","availability":"captured","generation_parameters_sha256":null,"model_id":"qwen-compatible-host","model_revision":null,"prompt_template_sha256":null},"environment_id":"local","environment_version":1,"goal":"Describe and transcribe the supplied media.","tool_catalog_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","workspace_id":"workspace-demo"},"previous_record_sha256":null,"privacy_class":"internal","record_id":"record:start","record_type":"episode.started","record_version":1,"request_id":null,"schema_version":1,"sequence":0,"step_id":null,"trace_id":"trace-demo","warnings":[]}
{"error":null,"episode_id":"episode-demo","lineage":null,"media":[{"derived_from":[],"end_seconds":null,"evidence_ref":"evidence:image","frame_indices":[],"locator":{"kind":"workspace_relative","value":"inputs/page.png"},"media_id":"media-image","media_type":"image/png","modality":"image","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size_bytes":12,"start_seconds":null},{"derived_from":[],"end_seconds":4.0,"evidence_ref":"evidence:video","frame_indices":[0,24,48],"locator":{"kind":"workspace_relative","value":"inputs/clip.mp4"},"media_id":"media-video","media_type":"video/mp4","modality":"video","sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","size_bytes":34,"start_seconds":0.0},{"derived_from":[],"end_seconds":4.0,"evidence_ref":"evidence:audio","frame_indices":[],"locator":{"kind":"workspace_relative","value":"inputs/speech.wav"},"media_id":"media-audio","media_type":"audio/wav","modality":"audio","sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","size_bytes":56,"start_seconds":0.0}],"modalities":["image","video","audio"],"occurred_at":"2026-08-18T12:00:01.000000Z","parent_record_id":null,"payload":{"content":[{"media_id":"media-image","type":"media"},{"text":"Inspect this page, clip, and narration.","type":"text"},{"media_id":"media-video","type":"media"},{"media_id":"media-audio","type":"media"}],"role":"user"},"previous_record_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","privacy_class":"internal","record_id":"record:message","record_type":"message.observed","record_version":1,"request_id":null,"schema_version":1,"sequence":1,"step_id":"step-1","trace_id":"trace-demo","warnings":[]}
{"error":null,"episode_id":"episode-demo","lineage":null,"media":[{"derived_from":[],"end_seconds":null,"evidence_ref":"evidence:image","frame_indices":[],"locator":{"kind":"workspace_relative","value":"inputs/page.png"},"media_id":"media-image","media_type":"image/png","modality":"image","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size_bytes":12,"start_seconds":null}],"modalities":["image"],"occurred_at":"2026-08-18T12:00:02.000000Z","parent_record_id":null,"payload":{"arguments":{"image":{"media_id":"media-image","type":"media_reference"}},"arguments_schema_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","attempt":1,"tool_id":"capicove.image.ocr","tool_version":1},"previous_record_sha256":"1111111111111111111111111111111111111111111111111111111111111111","privacy_class":"internal","record_id":"record:call","record_type":"tool.call","record_version":1,"request_id":"request-1","schema_version":1,"sequence":2,"step_id":"step-2","trace_id":"trace-demo","warnings":[]}
{"error":null,"episode_id":"episode-demo","lineage":null,"media":[],"modalities":["image"],"occurred_at":"2026-08-18T12:00:03.000000Z","parent_record_id":"record:call","payload":{"evidence_refs":["evidence:image"],"metrics":{"region_count":1.0},"result":{"regions":[{"text":"hello"}],"text":"hello"},"result_schema_sha256":"2222222222222222222222222222222222222222222222222222222222222222","status":"succeeded","tool_id":"capicove.image.ocr","tool_version":1},"previous_record_sha256":"3333333333333333333333333333333333333333333333333333333333333333","privacy_class":"internal","record_id":"record:result","record_type":"tool.result","record_version":1,"request_id":"request-1","schema_version":1,"sequence":3,"step_id":"step-2","trace_id":"trace-demo","warnings":[]}
{"error":null,"episode_id":"episode-demo","lineage":null,"media":[],"modalities":[],"occurred_at":"2026-08-18T12:00:04.000000Z","parent_record_id":null,"payload":{"outcome":"Visible multimodal evidence was processed.","status":"succeeded"},"previous_record_sha256":"4444444444444444444444444444444444444444444444444444444444444444","privacy_class":"internal","record_id":"record:complete","record_type":"episode.completed","record_version":1,"request_id":null,"schema_version":1,"sequence":4,"step_id":null,"trace_id":"trace-demo","warnings":[]}
```

The fixture is explanatory. Conformance fixtures in tests use actual computed
record IDs and predecessor hashes.

## Security and operational boundaries

Temporary export directories use owner-only permissions. Publication is an atomic
rename to an absent target. Failure MUST leave no partially published package.
SQLite remains a private adapter. Canonical modules MUST NOT depend on database,
web, MCP, model, GPU, or execution-backend packages.

Exports may contain sensitive user-visible content even after credential and path
redaction. Operators own access control, retention, backup, deletion, and approval
to include restricted Episodes. Export packages and user media MUST NOT be committed
to this repository.

## Alternatives considered

- **One JSON row per Episode.** Rejected because long trajectories require
  unbounded buffering and cannot be validated or exported incrementally.
- **Model-specific training rows.** Rejected because media-token conventions and
  objective fields couple evidence capture to one trainer and discard observable
  detail needed by later cleaning.
- **Store copied-media paths in records.** Rejected because export-local rewriting
  would invalidate source record hashes and predecessor chains.
- **Embed base64 media.** Rejected because it duplicates large sensitive bytes and
  makes bounded streaming and redaction harder.
- **Record hidden reasoning.** Rejected because it is not an observable execution
  fact and expands privacy and security risk.

## References

- [ADR-0012: Agent trace recording and export boundaries](../adr/0012-agent-trace-recording-export-boundaries.md)
- [ADR-0003: Tool Runtime, Trace, and controlled RSI boundaries](../adr/0003-tool-runtime-trace-rsi-boundaries.md)
- Agent multimodal trace export design (archived development reference)
- Issue #56 (archived development review)
