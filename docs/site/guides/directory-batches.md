# Resumable image and video batches

`capicove batch run` discovers supported media below one local directory and invokes
one canonical image or video Tool for each item. The command is an optional
composition layer over the public batch planner, executor, cache, and Tool adapters;
it does not introduce a second processing contract.

Every run requires explicit operation parameters, a provider fingerprint, and a
SQLite state path. The state database stores checkpoints and content-addressed Tool
results. Repeating an identical run resumes terminal checkpoints and reuses results
without invoking the Tool again.

## Image example

Create `image-size.json`:

```json
{
  "min_size": "32KiB",
  "max_size": "20MiB",
  "aggregation": "any"
}
```

Then run the local size filter:

```bash
PROVIDER_SHA="$(printf '%s' 'capicove-image-size:pillow-local-v1' | shasum -a 256 | cut -d ' ' -f 1)"
capicove batch run \
  --input media/images \
  --operation image_size_filter \
  --parameters image-size.json \
  --state state/image-size.sqlite3 \
  --provider-fingerprint "$PROVIDER_SHA" \
  --workers 4 \
  --output json
```

The unified CLI composes the local aspect-ratio, shape, size, and subplot filters.
The Python adapters can also wrap another single-image canonical Tool handler whose
provider has been configured by the host. Pair comparison and directory-wide
deduplication Tools are deliberately excluded from the one-item adapter.

## Video filtering example

Create `video-filter.json` with the same canonical criteria accepted by
`capicove video filter`:

```json
{
  "aggregation": "all",
  "criteria": [
    {
      "type": "metadata",
      "criterion_id": "minimum-duration",
      "min_duration_seconds": 2.0
    }
  ]
}
```

Run it against a directory:

```bash
capicove batch run \
  --input media/videos \
  --operation video_filter \
  --parameters video-filter.json \
  --state state/video-filter.sqlite3 \
  --provider-fingerprint "$PROVIDER_SHA"
```

## Video splitting example

Video splitting always writes path artifacts. Its artifact root is mandatory and
must remain outside the input tree so generated segments cannot enter later
discovery plans.

Create `video-split.json`:

```json
{
  "strategy": "duration",
  "config": {
    "split_duration": 60.0,
    "overlap_duration": 0.0,
    "min_last_split_duration": 5.0
  },
  "backend": "ffmpeg",
  "write_mode": "accurate"
}
```

```bash
capicove batch run \
  --input media/videos \
  --operation video_split \
  --parameters video-split.json \
  --artifact-root "$(pwd)/artifacts/video-segments" \
  --state state/video-split.sqlite3 \
  --provider-fingerprint "$PROVIDER_SHA"
```

Each source receives an isolated output directory derived from the operation,
provider, path identity, and immutable source hash. Changing any cache-affecting
parameter or provider fingerprint creates a different plan and cache key.

## Execution behavior

- Discovery is recursive, deterministic, ignores unsupported files, and never
  follows symbolic links.
- `--workers` is globally bounded to 1–64; `--modality-workers` can impose a lower
  limit for the selected modality.
- `--max-retries` is bounded to 0–10. Only failures explicitly marked retryable by
  the canonical Tool are retried.
- A source changed after discovery is not processed or cached.
- JSON output contains the plan ID, state counts, cache-hit count, terminal item
  outcomes, source identity, and the complete canonical successful Tool envelope.
- The command exits with status 1 if any item fails or changes during execution.

The provider fingerprint is caller-owned because Capicove cannot infer the identity
of every injected model, executable, or runtime. Keep it stable only while all
provider behavior that can affect output remains stable.

## Record operation trajectories

Place the opt-in trace store before the command name to record every Tool attempt:

```bash
capicove --trace-store state/agent-operations.sqlite3 batch run \
  --input media/images \
  --operation image_size_filter \
  --parameters image-size.json \
  --state state/image-size.sqlite3 \
  --provider-fingerprint "$PROVIDER_SHA"
```

Each actual attempt is a complete Episode with an attempt-scoped request and Episode
identity. Attempts for the same item share a trace identity, so downstream cleaning
can reconstruct retries. Cache hits do not fabricate Tool calls; the original Tool
Episode remains in the trace store, while the batch JSON reports the cache hit.

Use the separate [trace recording and export workflow](agent-trace-export.md) to
produce model-neutral JSONL. The records preserve source hashes, Tool identities,
evidence, timing, and lineage without generating ready-made SFT or Agentic RL rows.
