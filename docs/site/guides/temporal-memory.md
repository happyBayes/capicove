# Local temporal memory

Capicove Temporal Memory stores scoped Episodes and time-aware facts in private
local SQLite databases. Recall returns bounded context with citations to immutable
Evidence. The default path is offline: it performs no network or model calls.

!!! note "Current status"

    Temporal Memory is included in the 0.1.0 release candidate. Its local SQLite
    contracts and storage boundaries are recorded in accepted RFC-0009 and
    ADR-0015. Current command and workspace limitations are described on this
    page; contract acceptance does not imply published packages or a guarantee
    that future releases will read every database format without migration.

## Requirements and privacy boundary

- Python 3.11 or later.
- SQLite 3.42 or later with FTS5.
- A host-owned runtime directory and configuration outside the repository.
- Explicit workspace and scope authorization supplied by the host.

The runtime rejects symbolic links. It creates a private root or tightens an
existing empty root to mode `0700`; a nonempty root must already be private.
Files are private local storage, not per-record encryption. Do not place runtime
databases, request files, credentials, or user content in version control.

## Configure an offline runtime

Create `/absolute/private/config.json`:

```json
{
  "root": "/absolute/private/path/memory-runtime",
  "workspace": "demo",
  "scopes": ["notes"]
}
```

This selects the built-in `personal-v1` ontology. The runtime configuration is
trusted host policy: paths, workspace, authorized scopes, ontology, and providers
must never come from model-supplied Tool arguments.

## Remember and recall an Episode

Create `/absolute/private/remember.json`:

```json
{
  "scope": "notes",
  "content": "Alice moved to Paris on January 1.",
  "event_time": "2026-01-01T00:00:00+00:00",
  "idempotency_key": "alice-move-001"
}
```

Store it:

```bash
capicove memory --config /absolute/private/config.json remember \
  --request /absolute/private/remember.json
```

Then create `/absolute/private/recall.json`:

```json
{
  "scope": "notes",
  "text": "Alice Paris",
  "types": ["episodic", "semantic"],
  "states": ["accepted"],
  "top_k": 10,
  "max_characters": 12000,
  "max_tokens": 4000
}
```

Recall cited context:

```bash
capicove memory --config /absolute/private/config.json recall \
  --request /absolute/private/recall.json
```

Recall is lexical by default, not general natural-language question answering.
An empty query explicitly browses current authorized memories. Dense retrieval is
an optional host integration and never provisions an embedding model or index.

## Record a time-aware fact

An assertion distinguishes world time (`valid_from` and `valid_to`) from the time
Capicove observed the evidence (`event_time`):

```json
{
  "scope": "notes",
  "subject_name": "Alice",
  "subject_type": "person",
  "predicate": "city",
  "object": {"kind": "string", "value": "Paris"},
  "valid_from": "2026-01-01T00:00:00+00:00",
  "event_time": "2026-01-02T00:00:00+00:00",
  "evidence_text": "Alice reported moving to Paris on January 1.",
  "idempotency_key": "alice-city-001"
}
```

```bash
capicove memory --config /absolute/private/config.json assert-fact \
  --request /absolute/private/assertion.json
```

A later fact can close the earlier validity interval. To correct an error instead
of recording a later change, pass the earlier fact ID as `supersedes`. Reusing an
idempotency key with identical logical input returns the committed receipt;
changing the input returns `write_conflict`.

Recall can set `valid_at` and `known_at` independently. A historical `known_at`
cannot see evidence committed later. Returned citations identify immutable
Evidence and exact character spans.

## Extraction, retention, and recovery

Set `"extractor": "structured-json-v1"` in the host configuration to extract
explicit facts from structured JSON Episodes. Run `work` to process a bounded
batch; Capicove does not start a scheduler or background worker implicitly.

- Standard admissions expire after 90 days; pinned admissions do not expire
  automatically.
- `restricted` sensitivity is rejected before admission.
- `archive` hides a target from recall while preserving audit history.
- `erase` irreversibly denies and scrubs an entire scope while preserving audit
  commitments. Retry the same request and idempotency key after an uncertain
  timeout.
- `doctor` verifies the ledger. `read_only_safe` blocks writes and requires an
  operator to investigate or restore a trusted backup.

Back up `memory.sqlite3` and `evidence.sqlite3` with a consistent SQLite backup
procedure or while all clients are closed. Preserve the current privacy-authority
directory during restore; restoring an older copy can violate erasure authority
and therefore fails closed.

See the [Memory CLI reference](../cli/memory.md) and
[Memory API reference](../api/memory.md).
