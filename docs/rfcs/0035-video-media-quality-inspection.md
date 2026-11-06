# RFC-0035: Deterministic local video media quality inspection

- Status: Accepted by the maintainer on 2026-10-29; S1 accepted; S2 integration authorized on 2026-10-30
- Date: 2026-10-29
- Owners: Capicove maintainers
- Decision: [ADR-0039](../adr/0039-video-media-quality-inspection.md)
- Guide: [Usage and qualification](../site/guides/video/quality.md)

## Problem and requested outcome

Add a read-only video media quality inspection Tool. Today, `video filter` can
check dimensions and duration and evaluate sampled visual criteria. It does not
produce a full-file report for decoding, black intervals, static intervals,
silence, loudness, or audio/video timing differences. Calling FFmpeg externally
and assigning a reviewer score is not an implemented Capicove quality feature.

The proposed names are `capicove.video.quality` contract version 1,
`capicove video quality`, and `capicove_video_quality`. These names and contracts define the accepted S1 scope. Acceptance is
required by [AGENTS.md](../../AGENTS.md) and [GOVERNANCE.md](../../GOVERNANCE.md).
The maintainer explicitly accepted this RFC and ADR-0039 in the originating
conversation on 2026-10-29 and authorized S1 implementation. On 2026-10-30,
the maintainer explicitly requested completion of workspace-host integration,
authorizing S2. Existing filter behavior is unchanged.

## Scope and interpretation

The first release uses local FFprobe and FFmpeg, without model packages, network
access, telemetry, automatic repair, or source modification. It measures media
properties and emits policy findings. It does not grade teaching effectiveness,
factual accuracy, aesthetics, legal compliance, intelligibility, or semantic
audio/video synchronization. No 1–10 aggregate score is proposed: a measured
quantity, a policy threshold, a detector event, and human acceptance are different
things.

Black backgrounds, still slides, pauses, mono audio, and variable frame rates can
be intentional. Detector events are observations requiring review unless an
explicit policy makes them violations. No universal online-video loudness target
or minimum educational frame rate is assumed.

## Checkpoints and implementation

| Check ID | Measurements | Method | Interpretation |
| --- | --- | --- | --- |
| `streams` | Container, actual stream indexes, codecs, default/attached-picture dispositions, subtitle stream inventory, audio presence | Bounded FFprobe JSON | Unsupported media, ambiguous selection, and missing required audio are distinct outcomes; absence of a subtitle stream does not prove absence of burned-in captions |
| `video_format` | Encoded/display dimensions, rotation, sample/display aspect ratio, pixel format, nominal and average frame-rate rationals, declared bitrate and frame count | Normalize FFprobe output without inventing missing values | Report rate disagreement; do not infer dropped frames or playback smoothness from metadata alone |
| `audio_format` | Sample rate, channels, layout, codec, declared bitrate | Normalize the selected audio stream | Compare only with explicitly selected delivery requirements |
| `decode` | Selected-stream decode completion, decoded video frame/audio sample counts, diagnostic categories, first/last observed timestamps | Full selected-stream FFmpeg decode to null, bounded progress and diagnostics | Success requires normal EOF, no decode-error evidence, and complete coverage; container probing and short samples are insufficient |
| `black` | Near-black interval start/end/duration, union duration, fraction of observed video coverage | `blackdetect` on every decoded video frame | Review candidates, not proof of missing instructional content |
| `freeze` | Low-change interval start/end/duration, union duration, fraction of observed video coverage | `freezedetect` on every decoded video frame | Report as static intervals; a detector cannot distinguish a frozen player from an intentional slide |
| `silence` | Quiet intervals, union duration, fraction of observed audio coverage | `silencedetect` on the selected audio stream | Aggregate all channels by default; optional per-channel mode must preserve channel identity |
| `loudness` | Integrated LUFS, loudness range in LU, maximum true peak in dBTP | `ebur128=peak=true` without gain, resampling normalization, or loudness correction | Measure the original signal; missing true-peak support fails this requested check rather than silently substituting sample peak |
| `av_timing` | Declared and observed stream starts/ends; signed audio-minus-video start/end deltas | FFprobe plus decoded presentation timestamps, corrected for sample/frame duration where known | Timing difference only; equal starts/ends cannot establish lip synchronization or narration alignment |

The decoder covers the selected tracks, not every alternate language or camera
track in the container. That selection is explicit in every report.
The selected check list also defines the claim: a metadata-only invocation does
not establish full decoding, loudness, or temporal detector coverage. Report
unrequested checks as `not_run`; do not promote a scoped policy pass into a
claim that every quality checkpoint passed.

### Detector defaults

Defaults are reproducible detection settings, not publishing standards:

- Black: minimum 2 seconds; picture-black ratio 0.98; pixel-black threshold 0.10.
- Static: minimum 2 seconds; noise tolerance 0.001.
- Silence: minimum 2 seconds; threshold -50 dBFS; all-channel mode.
- Loudness: original channel layout, `dualmono=false`, true-peak measurement.
- Full timeline; no sampling or temporal trimming in version 1.

These defaults must be in normalized configuration and its digest. In particular,
black thresholds depend on decoded luma/range handling, and a small blackboard
annotation may not stop a frame from being near-black. Documentation and tests
must preserve this interpretation.

## Input and authority

One invocation inspects one regular local file. Batch callers can invoke the Tool
per file; version 1 accepts no directories, URLs, inline bytes, playlists,
caller-defined FFmpeg filters, or arbitrary command arguments.

| Field | Rule |
| --- | --- |
| `path` | Absolute, existing, authorized regular file |
| `video_stream_index` | Absolute container stream index; optional only when exactly one non-attached-picture video stream is eligible |
| `audio_stream_index` | Absolute audio index or absent; absence auto-selects exactly one audio stream, permits no audio, and rejects multiple audio streams as ambiguous |
| `checks` | Nonempty unique list of checkpoint IDs above; defaults to all |
| `detector_options` | Closed, finite numeric settings for the listed detectors only |
| `policy` | Closed version-1 requirements, separate from detector options; omitted means observation-only |

Selecting a stream of the wrong type, unknown fields/checks, nonfinite values,
reversed ranges, or out-of-bound thresholds is rejected before decoding. There
is no implicit "first audio track" choice when several audio tracks exist.

Policy supports only explicit media requirements: audio required, minimum display
dimensions, allowed codecs, frame-rate and audio-format ranges, loudness/true-peak
ranges, maximum absolute timing deltas, and maximum event duration/fraction.
Every threshold is inclusive and carries its unit. An omitted requirement makes
no quality claim. Policy cannot select an executable, source root, output path,
filter expression, network destination, or execution budget.

Execution budgets belong to the host/composition root: default deadline 300
seconds, maximum decoded media duration 7,200 seconds, maximum input size 4 GiB,
maximum 1,000 retained events per detector, 64 KiB diagnostic tail, bounded
FFprobe JSON and metadata lines, and explicit CPU/thread limits. Limits may be
narrowed by the operator; requests cannot widen them. Finite duration limits
must also be enforced while decoding when container metadata is absent or false.

## Result and completion semantics

Use the existing `ToolResult` envelope and preserve request/trace/operation
identity. A policy violation is a successful inspection, not an execution error.

The closed version-1 result contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exactly `1` |
| `source` | Original authorized path, SHA-256, byte size; no scratch locator |
| `selection` | Actual video/audio stream indexes and available track inventory |
| `configuration` | Normalized checks, detector options, policy and configuration SHA-256 |
| `coverage` | Per-stream observed start/end, frames/samples, EOF flag, decoded duration and applicable check completion |
| `checks` | Ordered check results, each with `check_id`, execution `status`, policy `verdict`, typed/unit-bearing metrics, events, findings and unavailable reason |
| `verdict` | `observed`, `pass`, `review`, `fail`, or `incomplete` |
| `warnings` / `failures` | Bounded typed diagnostics, scoped to a check/stream where possible |
| `lineage` | Tool/parser versions, FFmpeg/FFprobe versions, executable/configuration identities and existing invocation IDs |

Check execution status is `completed`, `not_applicable`, `failed`, or
`not_run`. Check verdict is `observed`, `pass`, `review`, `fail`, or `unknown`.
An interval event contains its type, stream index, start/end in source-relative
seconds, duration, and whether its end was closed by verified EOF. Findings have
a stable code and a severity of `info`, `warning`, or `error`.

All source-relative time values use a shared origin: the earliest known selected
stream presentation start. Preserve the original signed stream start values and
record this origin. Do not reset each track independently. If a common origin or
end duration cannot be established, affected timing metrics are null with reasons.
Do not use frame ordinal as seconds or duration metadata as a verified EOF.

Rules for trustworthy reporting:

1. A requested backend/filter missing, timeout, truncated parse stream, source
   change, decode error, or unverified EOF produces an incomplete operation and
   an outer failed ToolResult. Completed diagnostics remain visible; overall
   pass is prohibited.
2. Successful observation-only inspection yields `observed`, or `review` when
   detector candidates exist. Policy-evaluated complete results yield `fail` for
   any violation, otherwise `review` for review findings, otherwise `pass`.
3. No audio makes audio-only checks `not_applicable`; `require_audio=true`
   produces a policy failure in `streams`. No video makes the input unsupported.
   An unmeasurable required metric yields `incomplete`, never a substituted zero.
4. Silence and below-gate loudness have explicit reasons. JSON never contains
   NaN or infinity; absent measurements serialize as null. Do not convert the
   backend's loudness floor/sentinel into a plausible measured voice loudness.
5. End-of-file black/static/quiet intervals are closed only using observed EOF
   for the relevant track, and remain open/unknown after interrupted decoding.
   Account for the last frame/sample duration; do not silently clip to a
   mismatched declared container duration.
6. Event fractions use union durations over the corresponding observed track
   coverage, not a sum that double-counts overlap. Audio and video use separate
   denominators. Report the denominator and do not divide by zero.
7. Retention may cap event details while continuing exact event count and union
   totals. Mark `events_truncated`; a threshold decision may use those exact
   totals, but never imply that all interval details were returned. Parser limits
   that prevent exact totals instead make the affected check incomplete.
8. Missing bitrate/frame-rate metadata alone is not a detected media defect.
   Zero frames, no decodable selected track, and corrupt packets are distinct
   from an unavailable optional metadata field.

### CLI

```text
capicove video quality --input /absolute/video.mp4
  [--checks streams video_format audio_format decode black freeze silence loudness av_timing]
  [--video-stream-index INTEGER] [--audio-stream-index INTEGER]
  [--detector-options /absolute/options.json]
  [--policy /absolute/policy.json]
  [--output human|json]
  [--runtime-config /absolute/host.json]
```

The CLI parses local configuration files; the canonical Tool receives validated
objects. It does not pass JSON text through shell interpolation. Human output
always labels units, policy violations, incomplete checks, and review candidates.
JSON preserves canonical output without transport-specific score conversion.
CLI exit codes: 0 for complete observations/pass/review, 1 for complete policy
failure, 2 for rejected or incomplete inspection. MCP callers inspect structured
status and verdict instead of shell exit codes.

## Architecture and security boundary

`capicove.processing.video.quality` owns immutable contracts, policy rules,
inspection orchestration and provider ports. Its private adapters own FFprobe,
FFmpeg argument construction, streaming process supervision and result parsing.
`capicove.tools.video_quality` owns the canonical Tool boundary; CLI and MCP only
translate and invoke it. Reuse video source/path contracts through public APIs;
do not reach into audio capability private adapters.

Use a new inspection Tool rather than extending `VideoFilterCriterion`: decoding
coverage, unit-bearing stream metrics and temporal events are not sampled scalar
filter criteria. Existing filter v1, ADR-0006 and its `keep` semantics remain
unchanged. Share low-level work inside this new capability without publishing
one Tool for each detector.

The Tool declares `network_access=false`, confidential source handling,
`video:read` and `subprocess:execute`, and read-only observable effects. Invocation
scratch is private, bounded and removed on all exits; reports and traces are
persisted only through existing explicit caller/trace choices. Never invoke a
shell. Set deterministic locale and fixed, validated argv. No model or dataset
acquisition is needed.

Both executables must inspect the same verified private input copy, whose digest
and identity are checked throughout handoff. A before/after hash of an arbitrary
mutable original is not sufficient. Use reviewed capability-owned intake ports
and the existing host handoff when configured, with no wider filesystem roots.

Version 1 supports self-contained MP4/MOV and MKV/WebM containers only, validated
from content, not just extension. Reject URL/playlist/image-sequence manifests
and external media references. Restrict demuxers/protocols and test that nested
file/network references cannot read outside the staged source. `network_access`
metadata alone is not enforcement. Discovery runs no media subprocess or decode.

S1 standalone CLI/Python/MCP composition supports qualified macOS/Linux local
execution, with composition-owned FFmpeg/FFprobe executables. S2 workspace host
integration reuses configured absolute executables, existing descriptor pinning,
verified media staging and profile ceilings. S2 initially remains Linux-only
under RFC-0021/ADR-0026; do not weaken this boundary to claim macOS host support.
Host support must be exposed explicitly by catalog/readiness, not inferred from
the standalone command.

## Backend execution

Probe metadata once. For S1, collect frame counts and presentation coverage with
a structured FFprobe frame pass per selected stream; concurrent FFmpeg human
logs are not authoritative timing records. List each pass, its selected streams,
purpose and completion in coverage. This uses additional full decodes under the
same deadline and duration budget. Where compatible, chain `blackdetect,freezedetect` on video and
`silencedetect,ebur128=peak=true` on audio in one full decode. Map only selected
streams, disable stdin prompts and hardware acceleration, use a null output and
do not add `-shortest`, a truncated output duration, frame dropping, scaling, or
audio gain correction. Any additional pass must declare its purpose and coverage.

Prefer structured filter metadata for events and a version-qualified parser for
the final EBU summary. Use separate bounded progress/metadata/diagnostic channels
where supported. Drain outputs concurrently; avoid unbounded `capture_output`
for full-video logs. Match complete tokens/records, not arbitrary stderr
substrings; reject invalid or contradictory records. Check process exit,
decoder diagnostics, EOF and observed coverage independently. Cancellation and
deadlines terminate and reap the process group and clean private scratch.

Source-relative metrics retain timestamp precision and rational frame rates.
True peak is expressed in dBTP, not mislabeled as a sample-peak dBFS measurement.
No duration equality, clean decoder exit, or low motion score proves semantic
audio/video synchronization or instructional quality.

## Acceptance and rollout

The linked plan defines test-first slices and independent synthetic media oracles.
Acceptance requires canonical/CLI/MCP parity, actual FFmpeg adapter tests,
offline and immutable-source checks, missing-filter and resource-limit cases,
and the required `make quality`, `make test`, and `make build` gates. S2 additionally
requires Linux host tests; macOS failures cannot be counted as those tests passing.
Machine pass and teacher/listener/release acceptance remain separate.

## References

- [FFmpeg filter documentation](https://www.ffmpeg.org/ffmpeg-filters.html):
  `blackdetect`, `freezedetect`, `silencedetect`, `ebur128`.
- Local FFmpeg 8.1.2 `-h filter=...` output checked during proposal preparation.
- [ADR-0006](../adr/0006-unified-local-video-filter.md).
- [RFC-0021](0021-workspace-video-capability.md).
- [Module boundaries](../architecture/module-boundaries.md).


## S2 integration amendment (2026-10-30)

The maintainer explicitly requested completing integration into the Capicove
chain. `capicove.video.quality` is selectable in the version-2 workspace host's
existing video capability schema 1. It requires both configured FFmpeg and
FFprobe, `video:read` and `subprocess:execute`; no output root is required.
Optional `quality_limits` is a closed object with the public `QualityLimits`
fields. Operator values may only narrow default ceilings and the host's input
byte ceiling. Invocation arguments and canonical contract version remain v1.

Runtime adapts the existing verified media handoff through `QualityIntake`.
Composition revalidates the staged snapshot's size, identity, SHA-256 and container
references, holds a Linux descriptor across all passes, and preserves executable
descriptor locators without resolving them to mutable filesystem paths. It does
not make a second standalone temporary copy. Reports retain original source
identity and omit private locators. Existing Linux-only host support remains;
standalone macOS support does not bypass this boundary.

Catalog/readiness, MCP stdio and configured CLI reuse the existing host binding.
Configured tracing belongs to the host configuration; the CLI rejects a second
standalone trace store. This amendment grants no remote operation qualification,
network access, semantic quality scoring, or release acceptance.
