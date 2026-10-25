# Offline workspace search

Capicove can import your local Markdown notes and images, build an immutable
SQLite FTS5 index generation, and answer `capicove search` and
`capicove get-evidence` with exact, cited evidence. Everything runs locally. No
model, service, download, or network access is involved.

!!! note "Current status"
    This is the accepted [RFC-0018](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0018-offline-workspace-retrieval.md)
    slice ([ADR-0024](https://github.com/happyBayes/capicove/blob/main/docs/adr/0024-offline-workspace-retrieval.md)).
    It runs on Linux and macOS. On Windows it fails closed until the no-follow
    file checks it relies on have an equally strong Windows implementation.

## Requirements

- Python 3.11 or later, with an SQLite build that includes FTS5 (the standard
  CPython builds do). `capicove setup doctor` reports `missing_dependency` when
  FTS5 is absent; there is no silent fallback.
- An owner-private workspace directory (mode `0700`) that holds the host config
  and the index.
- One or more existing local source directories. Sources are read, never modified.

## Configure the workspace

The quickest start creates the workspace and a ready configuration in one step.
Each `--source-root ALIAS=DIR` names an existing directory to search; repeat it
for more roots:

```bash
capicove setup init /home/me/cove --source-root notes=/home/me/notes
```

This writes `/home/me/cove/host.json` with the `write` profile, the two
retrieval Tools, `index_root` at `/home/me/cove/index`, a 1 MiB
`max_source_bytes`, and no network access. It does not read your notes or build
an index, and its `next_commands` output lists the commands below. Invalid
roots fail before anything is created.

To write the configuration yourself instead, create it with mode `0600`, for
example `/home/me/cove/host.json`:

```json
{
  "schema_version": 2,
  "server_name": "capicove-local",
  "workspace_id": "personal",
  "actor_id": "owner",
  "environment_id": "local",
  "workspace_root": "/home/me/cove",
  "profile": "write",
  "permissions": ["evidence:read", "evidence:search"],
  "enabled_tools": ["capicove.get_evidence", "capicove.search"],
  "trace_store": null,
  "capabilities": {
    "retrieval": {
      "schema_version": 1,
      "source_roots": {"notes": "/home/me/notes"},
      "index_root": "/home/me/cove/index",
      "corpus_ref": "corpus:personal-notes",
      "retrieval_profile_ref": "local-default",
      "max_source_bytes": 1048576
    }
  },
  "network": {"enabled": false, "source_hosts": []}
}
```

The `retrieval` object accepts exactly these six fields:

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1`. |
| `source_roots` | 1 to 16 lowercase aliases mapped to absolute, existing, local directories without symlinks. |
| `index_root` | Absolute directory inside `workspace_root`, owner-private when it exists, disjoint from every source root. |
| `corpus_ref` | Nonblank corpus identity owned by the host. |
| `retrieval_profile_ref` | Exactly `local-default`. |
| `max_source_bytes` | Per-file limit from 1 byte to 16 MiB. |

Check readiness. This never creates the index or reads sources:

```bash
capicove setup doctor --config /home/me/cove/host.json
```

## Import sources

Importing is an explicit operator write and needs the `write` or `admin` profile.
Each import names a configured root alias and a normalized path below it:

```bash
capicove source import --runtime-config /home/me/cove/host.json \
  --root notes --relative-path travel/paris.md
capicove source import --runtime-config /home/me/cove/host.json \
  --root notes --relative-path photos/eiffel.png
```

The importer opens every path component without following symlinks and verifies
the file did not change while it was read. It accepts UTF-8 Markdown (`.md`,
`.markdown`) and PNG, JPEG, GIF, or WebP images identified by their content
bytes. It records the source's SHA-256, size, and media type. Re-importing
unchanged bytes reports `"changed": false`.

Images are recorded as evidence only. They are never lexical hits, and no
caption-to-image linking is inferred.

## Retire a source

To drop a note from future builds, retire it. The file itself does not need to
exist any more:

```bash
capicove source forget --runtime-config /home/me/cove/host.json \
  --root notes --relative-path travel/paris.md
```

The command prints `{"source_ref": ..., "forgotten": true}`, or `false` when the
source was not in the inventory; repeating it changes nothing. Nothing changes
for searches until the next `index build`, which leaves the source out. Retiring
every source and building again activates a valid empty generation. Importing the
same path later makes it eligible again. The import ledger keeps every import and
retirement event; the source file and earlier generations are never modified.

Import, forget, and build take one lock per index, so concurrent commands run one
after another; a build always works from the inventory as it was when it started.

## Build and activate a generation

```bash
capicove index build --runtime-config /home/me/cove/host.json
```

The build re-reads every imported source and checks its hash against the import.
If a file changed since its import, the build stops with
`source … changed since import; re-import it` and the previously active
generation stays in service. A deleted file fails the same way until you
`forget` it. Otherwise the build writes a new index in private
staging, validates its counts and FTS5 integrity, and then switches the active
generation atomically. Identical inputs produce the same generation ID.

The output lists the generation ID, counts, and one `source_evidence` entry per
source. Use those evidence IDs to open image evidence.

## Search and open evidence

```bash
capicove search "Paris bakery" --runtime-config /home/me/cove/host.json
capicove get-evidence ev-… --runtime-config /home/me/cove/host.json \
  --section metadata --section lineage
```

Search pins the active generation for the whole call. It supports the canonical
corpus, artifact, media-type, and text-origin filters; continuation cursors and
the `full` detail level are rejected as invalid input. Hits contain a bounded
snippet and an opaque, generation-bound evidence ID.

The `metadata` section of a text chunk contains its exact `text`, the
`start_char`/`end_char` range in the source, the source reference, and its
SHA-256. The `lineage` section names the generation, manifest hash, and import
and chunking versions. Evidence comes only from the active generation's index:
it never reopens a source path. After a new generation is activated, older
evidence IDs fail closed, as do IDs from another workspace or whose stored text
no longer matches its recorded hash.

## Serve over MCP

The same configuration serves both Tools over MCP stdio. Search and Evidence are
read-only and work with the `read` profile too:

```bash
capicove mcp serve --config /home/me/cove/host.json
```

The MCP Tools `capicove_search` and `capicove_get_evidence` use the same
handlers as the CLI. Neither startup nor a query builds an index, downloads
anything, or opens the network.

## Limits

- Tokenization is Unicode word-based (`unicode61`). Recall for languages written
  without spaces, such as Chinese or Japanese, is limited.
- A chunk's exact text is capped so the `metadata` section stays within the
  Evidence Tool's 4 KiB section limit; long paragraphs are split.
- Generations are kept on disk for lineage. Evicting old generations is not yet
  automated.
- A build reads sources one at a time and writes units in batches, so its memory
  stays bounded by `max_source_bytes` rather than the total size of the notes.
- The import ledger uses format version 2. Earlier releases cannot read it and
  fail instead of silently ignoring retirements.
