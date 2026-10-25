# ADR-0005: Make Video Split Outputs Accurate by Default

- Status: Accepted
- Date: 2026-08-10
- Owners: Capicove maintainers

## Context

Manual testing against a 141-second H.264/AAC video exposed three gaps that the
initial synthetic tests did not protect. ffmpeg stream copying drifted across
non-keyframe duration and scene boundaries, keyframe splitting produced one clip
per short GOP, and the scene extra installed PySceneDetect without an OpenCV
runtime. The initial PyAV writer also omitted audio.

The Tool is intended for agent execution and training-data preparation. Its default
must therefore favor reproducible media boundaries and complete audio/video output;
speed-oriented lossy semantics must be explicit.

## Decision

`capicove.video.split` contract version 1 is corrected before its first release:

- `write_mode` is an optional common input with `accurate` as the default and
  `copy` as an explicit fast path.
- Accurate ffmpeg writing transcodes the primary video to H.264 and selected audio
  streams to AAC. Copy mode remains keyframe-aligned and may drift.
- Write mode is included in normalized configuration, output identity, and lineage.
- Keyframe configuration gains `min_segment_duration`, defaulting to 10 seconds.
  Adjacent GOPs are grouped at keyframe boundaries and a short tail is merged into
  the preceding segment.
- `capicove[video-scene]` and `capicove[video]` install headless OpenCV.
  PySceneDetect starts in-scene so detected ranges include the beginning.
- PyAV remains the optional inspector for the `av` backend; range writing reuses the
  same audited ffmpeg writer so accurate boundaries and audio preservation do not
  diverge between inspection backends.
- Human CLI output lists every segment's chronological index, requested time range,
  and path or output identity.

## Consequences

- Default duration and scene outputs are slower and larger to compute, but their
  media boundaries are frame-accurate within codec time-base tolerance.
- Copy mode remains available for trusted pipelines that accept GOP drift.
- Keyframe output volume is bounded by a caller-controlled minimum duration instead
  of source encoder GOP frequency alone.
- The optional `av` backend also requires ffmpeg for writing; Tool resources already
  declare both ffmpeg and ffprobe.
- Output identities change when write mode or keyframe grouping changes.
- Integration tests assert track preservation, timestamp alignment, optional
  dependency usability, and real detector execution rather than file existence only.

## Alternatives Considered

- Keep stream copy as the default: rejected because arbitrary duration and scene
  boundaries are not generally keyframes.
- Re-encode only the first GOP around each cut: rejected for the initial release due
  to timestamp and codec-composition complexity.
- Keep one output per keyframe: rejected because normal one-second GOPs create
  impractical training artifacts.
- Maintain a separate PyAV muxer: deferred until it can meet the same audio,
  timestamp, and failure contracts as the shared writer.

## References

- Video split Tool design (archived development reference)
- [Testing strategy](../development/testing.md)
- [Module boundaries](../architecture/module-boundaries.md)
