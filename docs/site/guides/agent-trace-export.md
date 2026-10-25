# Agent operation trace export

Capicove can record observable, multimodal Agent operations and export them as a
model-neutral JSONL package. The package preserves visible messages, Tool calls and
results, media identity, artifacts, batch outcomes, lineage, completion, and linked
feedback. It is intended as source material for downstream cleaning and dataset
construction. It is not an SFT or Agentic RL dataset format.

Recording, export, and validation are local-only unless an application explicitly
configures a Tool that uses the network. No recording occurs by default.

## Record from Python

Create one recorder at the application composition root and reuse its sanitizer for
every traced boundary. The built-in SQLite adapter is private because storage is not
part of the public trace contract; applications may instead provide any implementation
of `TraceStore`.

```python
from datetime import UTC, datetime
from pathlib import Path

from capicove.traces import TraceRecorder, TraceSanitizer
from capicove.traces._adapters.sqlite import SqliteTraceStore

store = SqliteTraceStore(Path("state/traces.sqlite3").resolve())
recorder = TraceRecorder(
    store,
    sanitizer=TraceSanitizer(),
    clock=lambda: datetime.now(UTC),
)
```

Use `TraceRecorder.start_session()` when the host can observe a complete multi-step
Agent interaction. Record only messages actually visible to the host, retaining their
original ordered `TextContent` and `MediaContent` blocks. Then record Tool calls and
results, artifacts, batch facts, completion, and optional post-completion feedback.
Never infer or synthesize hidden reasoning.

CLI and MCP recording creates one Episode per canonical Tool invocation. Those
transports cannot observe the remote Agent's whole conversation. A host that needs a
multi-step trajectory must keep a public `TraceSession` open and explicitly record its
visible messages and operations.

## Record from the CLI

Place the opt-in `--trace-store` option before the command name:

```bash
capicove --trace-store state/traces.sqlite3 audio transcribe \
  --input media/meeting.wav \
  --model models/transcription.bin \
  --vad-model models/vad.bin
```

The same option applies to canonical image Tool execution, video filtering, video
splitting, and `batch run`. A batch records one complete Episode per actual Tool
attempt; retries share a trace identity and use attempt-scoped request and Episode
identities. Cache hits do not fabricate calls. `video deduplicate` is not recorded
because it does not cross the canonical `ToolHandler` boundary. Omitting
`--trace-store` preserves the original behavior and does not construct or write trace
storage.

## Record from MCP

Pass the same explicitly constructed recorder to a registration or server factory:

```python
from capicove.mcp.audio_transcribe import register_audio_transcribe_mcp

register_audio_transcribe_mcp(server, audio_handler, recorder=recorder)
```

Audio transcription, image Tools, video filtering, and video splitting share the same
fail-closed recording boundary. If recording a call or result fails, the operation does
not silently produce an untracked success.

## Export a package

Reference mode is the default. It keeps immutable hashes and safe locators in JSONL
without copying source bytes:

```bash
mkdir -p exports
capicove traces export \
  --store state/traces.sqlite3 \
  --output exports/agent-traces
```

Copy mode resolves only workspace-relative locators below one explicit absolute media
root and writes hash-verified, content-addressed media into the package:

```bash
WORKSPACE_ROOT="$(pwd)/workspace"
capicove traces export \
  --store state/traces.sqlite3 \
  --output exports/agent-traces-with-media \
  --media-mode copy \
  --media-root "$WORKSPACE_ROOT"
```

The exporter refuses to overwrite an existing output directory. It publishes the new
directory atomically and leaves no final package after a failed export. Selection flags
include Episode IDs, UTC start/completion bounds, modalities, Tool IDs and statuses,
Episode statuses, incomplete Episodes, and restricted Episodes. Restricted and
incomplete Episodes are excluded unless explicitly requested.

## Inspect and validate

```bash
capicove traces inspect exports/agent-traces --verify-media --output json
capicove traces validate exports/agent-traces --output json
```

Validation streams bounded JSONL records and checks manifest/schema hashes, shard
counts and hashes, canonical encoding, Episode sequence and predecessor chains, Tool
call/result pairing, media layout, unexpected files, and optional copied-media bytes.
Treat any validation error as package tampering or corruption. Warnings can describe
explicitly allowed incomplete Episodes.

## Reconstruct trajectories for cleaning

The portable schema is record-oriented:

1. Group records by `episode_id`.
2. Sort each group by integer `sequence`.
3. Join `tool.call` and `tool.result` records by `request_id`; use the parent record ID
   as an additional integrity link.
4. Join content blocks, Tool records, and artifacts to media by `media_id`.
5. Use immutable SHA-256 values to deduplicate bytes and preserve lineage.

Downstream cleaning can select successful, failed, rejected, retried, cache-hit,
skipped, and source-changed trajectories. Keep timing, warnings, errors, evidence,
artifact derivation, and Agent/model lineage when they are relevant to the target
task. Convert ordered text/media blocks into a model-specific template only in the
downstream dataset builder. Capicove does not introduce `<image>`, `<video>`, or
`<audio>` tokens and does not add `chosen`, `rejected`, `reward`, hidden reasoning, or
chain-of-thought fields.

## Privacy and retention

- Tool-declared media fields become media references or redacted media identities;
  secret-like keys, bearer credentials, known key formats, and absolute paths are
  redacted before persistence.
- Recording is fail-closed, but operators remain responsible for choosing privacy
  classes and reviewing visible content before including restricted Episodes.
- Reference exports may depend on media retained outside the package. Copy exports are
  self-contained only for successfully resolved local media.
- Back up, rotate, and delete trace stores and exports according to the host's data
  retention policy. Capicove does not do this automatically.
- Never commit trace databases, exported packages, copied media, user content, model
  weights, or credentials to the source repository.
