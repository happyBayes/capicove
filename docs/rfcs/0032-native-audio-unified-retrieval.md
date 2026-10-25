# RFC-0032: Native audio and optional unified multimodal retrieval

- Status: Accepted for implementation by requester direction on 2026-10-23
  (`audio-local-v1` first); not semantically or production qualified
- Date: 2026-10-22
- Owners: Capicove maintainers
- Request: Add native audio retrieval and an optional unified retrieval loop,
  informed by DigUp, on top of the existing multimodal search capabilities.
- Decision: [ADR-0036](../adr/0036-native-audio-unified-retrieval.md)
- Design: Implementation design (archived development reference)

## Acceptance boundary

The requested outcome is authorized. This document proposes the new canonical
types, persistent format and composition boundaries required to implement it;
those details have not been accepted. Do not mark this RFC accepted, change public
contracts, or advertise runnable new commands until maintainer acceptance is
recorded. Design acceptance does not authorize model transfers, commits, pushes,
publication, or production qualification.

RFC-0002 and RFC-0008 provide visual and text retrieval foundations. RFC-0022
provides audio import/transcription, not native audio embedding. RFC-0030's audio
model capability is transcription. None currently defines an audio search target
or this unified generation format.

## Outcome and options

Choose an optional `unified-local-v1` profile. It indexes text chunks, independent
images, video keyframes and audio windows through one explicitly provisioned
shared-space model. It can fuse compatible, authorized existing lexical or
specialist candidate lanes. A separate `audio-local-v1` profile uses the same
audio preparation and generation machinery for audio-only collections.

Existing text, image and video profiles retain their current behavior and model
spaces. The first release is an offline, single-node preview with explicit import,
build, activation, search and evidence operations. It is not a desktop crawler,
automatic folder watcher, remote worker, or distributed index.

Alternatives considered:

1. Merge existing result lists only: smaller change, but cannot supply native
   sound retrieval or a common text-to-image/audio recall space.
2. Replace all existing encoders: simpler conceptual model, but discards specialist
   matching and requires unnecessary migration of existing indexes.
3. Add a shared-space profile and preserve specialist lanes: selected, with an
   explicit additional index and model cost.

## Canonical contract extension

Propose retrieval contract version 4, additive to the current version 3:

- `QueryModality.AUDIO_CLIP = "audio_clip"` and an `AudioClipQuery` containing an
  existing media source, optional `TemporalRange`, and optional zero-based audio
  stream ordinal. Inline audio retains the 4 MiB media ingress ceiling.
- `TargetType.AUDIO_SEGMENT = "audio_segment"` requires a nonempty half-open
  millisecond `time_range`; it forbids a text locator or generated transcript.
- `MatchKind.AUDIO = "audio"` identifies native waveform evidence. ASR matches
  continue to use `MatchKind.ASR` and `TextOrigin.ASR`.
- Audio searchable units bind a versioned audio representation receipt containing
  source digest, selected stream, exact sample interval, sample rate, channels,
  decoder identity, PCM digest, preprocessing and embedding-space identity.
  Existing result fields carry the receipt and evidence references.
- Keep the `SearchRequest` fields and the `SearchResult` envelope. The selected
  host profile explicitly declares allowed query/target pairs and contract version.
  Unknown audio variants fail before dispatch in older profiles and adapters.

The initial unified profile supports text queries across text chunks, images,
keyframes and audio segments. Image and bounded audio-clip queries search the
shared dense lane across these targets, without lexical fallback. Audio-only
profiles support text and bounded audio-clip queries to audio segments. Video-clip
queries remain unsupported. Video files contribute keyframes and separately
selected audio windows; neither represents joint temporal video understanding.

An audio query without a range is accepted only if the selected stream fits one
configured window. Longer queries require an explicit range; no silent truncation
or averaging of a long query. Multiple audio streams require explicit selection,
consistent with RFC-0004. Native audio processing must not use speech VAD to
discard environmental sounds or music.

## Filters, scores and results

Apply workspace, authorization, corpus, artifact, media-type and temporal filters
before ranking. A time filter excludes units with no temporal coordinates.
`text_origins` restricts results to matching text chunks; it must not silently
admit native audio or visual units. For video-derived audio, media-type filtering
uses the original video's media type, while target type remains `audio_segment`.

Within one compatible shared space, rank normalized dense vectors by cosine.
Fuse dense, lexical and enabled specialist candidate ranks with existing weighted
RRF. Never compare raw scores from BGE, SigLIP2, ONNX or the new shared model.
Specialist adapters must supply unit-level candidates, complete pinned generation
identity and score semantics; public preview hits alone are insufficient.

Results identify units rather than collapsing each file to one best hit. Different
moments of the same recording remain distinct. Deduplicate the same canonical unit
across lanes; do not merge overlapping windows into a new claimed evidence span.
Stable ties use unit ID. Scores are ordering values, not probabilities. The first
profile emits `retrieval_preview`; semantic accept-or-abstain calibration requires
separate representative evaluation. Do not copy DigUp's corpus-dependent z-score
gate as a production confidence threshold.

## Immutable unified generation

Add a separately versioned `unified-generation-v1` format. Its manifest binds:

- workspace, corpus, contract and format versions;
- ordered source inventory and searchable-unit manifest digests;
- complete embedding space and model receipt digest;
- windowing, extraction and text chunking fingerprints;
- immutable vector shard and lexical index payload digests and counts;
- pinned member generations for each enabled specialist lane;
- validation report and preview qualification identity.

Build in private staging, validate, publish, then explicitly activate atomically.
An unsuccessful build preserves the active generation. Query and evidence reads
lease the same complete generation set. Retain member leases through outstanding
workers, including timeout cleanup. No mixing of independently current aliases.

Imports update an append-only inventory. Build consumes all non-retired imports,
so adding a second video does not replace the first. Unchanged source and complete
processing identities allow representation reuse. Forget affects the next build;
re-importing changed content creates a new source version. Exact repeats are
idempotent. Generation ID derives from canonical content, not creation time.

The preview supports at most 10,000 searchable units across all modalities, not
10,000 files. Check limits before publication. Use an embedded lexical index and
bounded exact vector retrieval as the reference backend. ANN remains an explicit
later backend selection; no capacity or latency guarantee follows from this cap.

## Evidence and authority

Search requires host-created evidence-search authority. Builds additionally
require source read, derived artifact write, and subprocess permission when a
decoder runs. Query fields may narrow, never widen, configured authority.

Source inspection, hashing and decoding use verified source snapshots through
public intake operations. Model assets and binaries require verified local
receipts. Reject external URLs, unapproved roots, symlinks, changed sources,
unsupported platforms, malformed model responses and incompatible spaces.
No provider download, ASR fallback, or network request is implicit.

`get_evidence` reauthorizes each receipt and source and returns bounded metadata,
preview references and lineage. It never trusts a path returned by a search
provider. Audio metadata includes the original source and selected stream, sample
interval and millisecond range. Native audio hits do not fabricate transcript text.
Preview resource access remains permission checked and bounded.

Evidence IDs bind workspace, generation and immutable unit content. The initial
reader resolves the active leased generation; superseded generation evidence
fails closed unless a separately authorized historical reader exists. Search
responses and normal traces expose neither vector payloads nor private paths.

## Model and provider boundary

Add a typed, optional `multimodal_embedding` capability to the existing model
registry/factory. Its port exposes text-query, text-document, image and audio-PCM
encoding with explicit full space identity, deadlines and batch limits. The
provider validates shape, finite values, count and normalization on every output.
Core contracts import no numerical, model or native-runtime SDK.

EmbeddingGemma 2 is the first proposed provider, because its official model card
and DigUp integration expose a shared text/image/audio space. The initial Python
adapter should use an independently verified local implementation with explicit
query/document prefixes and local-only loading. No model ID alone authorizes
loading. Record upstream revision, file hashes, license, runtime versions,
precision, preprocessing and output projection in the deployment receipt.

Select the exact backend and pin its dependencies during provider qualification;
do not claim Transformers and llama.cpp vectors are interchangeable. Existing
locally verified assets should be reused where available. Any required model or
dataset transfer needs separate exact-scope authorization under AGENTS.md.

## Compatibility and acceptance

Old source inventories, profiles, generation readers and commands continue to
work. New readers reject unknown format versions; older plugins are not asked to
interpret audio targets. Existing profiles never silently select the new model.
Unified builds do not mutate or migrate an active legacy index.

The design's acceptance matrix is required before reporting implementation
complete. Deterministic tests establish orchestration and authorization; a real
local-model replay establishes inference functionality; representative labeled
media establish semantic quality. Missing assets are `unqualified` or skipped,
never successful semantic acceptance.

## Implementation record

On 2026-10-23 the requester directed implementation of local native-audio search
with the EmbeddingGemma 2 8-bit GGUF and a pinned llama.cpp CPU runtime, plus the
setup/benchmark script, audio indexing, search and evidence lookup. This records
acceptance for implementation; it is not maintainer review of every schema detail,
semantic qualification or production support. Implemented contract decisions:

- Retrieval contract version 4 adds `QueryModality.AUDIO_CLIP`, `AudioClipQuery`
  (source, optional range, optional stream ordinal 0-63), `TargetType.AUDIO_SEGMENT`
  (requires a time range; forbids text locators and snippets) and `MatchKind.AUDIO`.
  Existing services reject the new variants before dispatch.
- Host configuration `capabilities.multimodal` is a closed version-1 object:
  `profile`, `source_roots`, `index_root`, `staging_root`, `corpus_ref`,
  `model_config`, `model_binding`, `ffmpeg`, `ffprobe`, `max_source_bytes`.
  Only `audio-local-v1` is selectable in this slice.
- Model capability `multimodal_embedding`; first provider
  `embeddinggemma2-llamacpp` (CPU only, Unix-socket `llama-server --offline`,
  hash-pinned model, projector and server binary).
- Tools `capicove.search.multimodal` (MCP `capicove_search_multimodal`) and
  `capicove.multimodal.get_evidence` (MCP `capicove_multimodal_get_evidence`),
  both read-only; CLI `capicove retrieval import|forget|build|activate|search|get-evidence`.
- Persistent format `unified-generation-v1`: content-derived `ug-` generation IDs,
  per-generation SQLite unit store pinned by SHA-256, atomic `active.json` pointer,
  `mev-` evidence IDs binding workspace, generation, unit and receipt.

Second slice, `unified-local-v1`: one generation holds text chunks (Markdown,
`capicove-markdown-paragraphs-v1`, document prompt `title: none`), whole images
(PNG/JPEG) and audio windows in the shared space. Text queries fuse the shared dense
lane with an FTS5 lexical lane over text chunks using weighted RRF (k = 60, weights
1.0); image and audio-clip queries use only the dense lane. Unit IDs are unique
across lanes, scores never cross lanes, and filters apply before every lane.
Video keyframes and soundtracks are not yet part of this profile.

Evidence and remaining gates: native audio acceptance (internal record; public support status is documented in the capability guide).
