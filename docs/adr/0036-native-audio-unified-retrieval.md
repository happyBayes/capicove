# ADR-0036: Isolate native audio encoding and optional unified retrieval

- Status: Accepted for implementation with RFC-0032 on 2026-10-23 (requester direction)
- Date: 2026-10-22
- RFC: [RFC-0032](../rfcs/0032-native-audio-unified-retrieval.md)

## Context

Capicove has separate text, reverse-image and video retrieval paths. Audio intake,
clipping and transcription exist, but transcription does not preserve non-speech
sound semantics. A single optional search loop needs native audio representations
and a common candidate space without weakening evidence lineage or replacing
existing specialist indexes.

## Proposed decision

Keep source intake and immutable payload authority in their current owners.
`processing.audio` owns bounded stream selection, PCM window preparation and
sample-accurate receipts. Retrieval consumes public preparation results; it must
not import `processing.audio._adapters` or independently invoke FFmpeg.

`model` owns the optional verified multimodal encoder, its registry binding and
instance lifecycle. `retrieval` owns compatible-space validation, unit candidates,
fusion and result projection. `runtime` composes these public ports, explicit
source inventory operations, generation publication, activation and evidence
resolution. CLI and MCP translate the same operations without owning policy.

Use a new optional shared-space generation for unified recall. Preserve existing
specialist lanes through public candidate adapters whose exact generations can be
pinned in the unified manifest. If an adapter cannot bind its candidates to the
unified catalog, exclude that lane at build validation rather than inventing a
mapping from a filename or a returned preview score.

Use native audio and ASR as independent representations. Native windows need no
Whisper or VAD runtime. Video frames and audio windows remain separate units with
their own temporal evidence; shared embedding dimensions do not establish joint
event-order understanding.

## Consequences

- Existing profiles remain independently usable and migration-free.
- The additional model and vector index incur explicit storage and startup costs.
- Unified generation publication must validate all member identities together.
- Model changes rebuild incompatible representations; runtime upgrades may reuse
  them only under a declared, tested vector-compatibility policy.
- File-level result presentation can be added above unit-level evidence later.
- Desktop watching, remote execution, historical evidence access and formal
  no-answer calibration are outside the initial profile.

## Rejected alternatives

Replacing every encoder loses useful instance, perceptual and lexical behavior.
Merging final search responses lacks comparable score semantics and pinned unit
identity. Treating ASR text as native audio cannot search non-speech content.
Importing DigUp's Swift runtime into canonical contracts introduces an unnecessary
backend dependency; its extraction and evaluation patterns are references instead.
