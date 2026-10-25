# Import audio evidence

`AudioProcessor` validates an authorized local path, verifies supported content,
hashes the source, and can place it in a content-addressed immutable store. Existing
objects are checked before reuse; source changes during processing are rejected.

Callers provide path authorization, stream inventory, and validation implementations.
The included filesystem authorizer, bounded FFprobe inspector, and FFmpeg validator
are adapters, not canonical contracts.
Import failures are explicit and do not leave a successful evidence record.

The processor counts audio streams before storage. One stream may select ordinal
zero implicitly; multiple streams require an explicit zero-based audio ordinal.
Only the selected stream is decoded for admission, while the evidence hash still
identifies the complete original container.

See the [audio API](../../api/audio.md) for evidence and processor contracts.
