# 0026: Stage host media through verified private copies

- Status: Accepted
- Date: 2026-10-06
- Owners: Capicove maintainers

## Context

FFmpeg, FFprobe, PyAV, and Whisper.cpp open files by path. If a host passed the
caller's path straight through, a file or parent directory swapped after
authorization could make an external backend read something else.

## Decision

- A capability-owned intake opens each authorized source through pinned
  no-follow directory handles, checks the regular-file type, size bound, and
  container signature, and copies it in bounded chunks to a new read-only file in
  a per-invocation directory under the configured `staging_root`. It records
  SHA-256 and verifies the source identity (`st_dev`, `st_ino`, size, mtime)
  before and after the copy; any change rejects the call with no output.
- Backends receive only the staged copy. Results replace staged locators with
  the original path; SHA-256 and lineage refer to the original bytes.
- The invocation directory is removed when the call ends. Directories left by a
  process that no longer exists are removed at the start of the next call.
- Hash-only operations such as exact deduplication read through the same pinned
  handles without copying.
- Model and executable files are checked by SHA-256 and passed to subprocesses
  through the same staging copy when the RFC for that capability requires it.
- Staging happens only during explicit calls. Startup, doctor, and catalog do
  not create staging directories.

## Consequences

- Each media call costs one extra sequential copy, bounded by `max_input_bytes`.
- A source that changes during intake fails the call; a source that changes
  after intake does not affect the result.
- Windows media execution waits for an equivalent handle-based implementation.

## References

- [RFC-0021: Local video Tools in the workspace host](../rfcs/0021-workspace-video-capability.md)
- [ADR-0023: Compose capabilities from one versioned workspace authority](0023-versioned-workspace-host-authority.md)

## Review amendment (2026-10-06)

- Staging is bound to descriptors: invocation directories and staged copies are
  created through a pinned `staging_root` handle and exposed to backends as
  `/proc/<pid>/fd/<n>/…` paths. Split outputs use the same mechanism through a
  pinned `output_root` handle.
- Executables get the identity-preserving handoff the original decision required:
  startup snapshot of identity and SHA-256, per-call re-verification, and
  execution of the verified open file through its descriptor path. Copying an
  executable was rejected because it breaks `$ORIGIN`-relative shared libraries.
- Audio inputs must match their extension's container signature, and FFmpeg runs
  with a forced demuxer and the local `file` protocol only.
- Descriptor paths rely on Linux procfs; host media execution fails closed on
  other platforms until an equivalent exists. A same-user process that rewrites an
  executable's bytes in place between verification and `exec` remains a residual
  risk; replacement of the path is fully covered.
