# Standalone local Wiki

The local Wiki turns reviewed Markdown notes into an inspectable, offline
workspace with immutable source snapshots, user-editable pages, deterministic
validation, and a disposable SQLite search index. It does not connect to Temporal
Memory and is not canonical Capicove knowledge.

## Install and initialize

Install YAML support, then create a new Wiki beneath a workspace:

```bash
python -m pip install "capicove[wiki]"
capicove wiki init ./knowledge-workspace
```

Initialization refuses an existing `wiki/`. It stages and validates a complete
layout before publication, so a normal failure leaves no partial destination.

## Ingest and validate a note

The source must be one readable UTF-8 Markdown regular file. The supplied path
must not contain symbolic-link components.

```bash
capicove wiki ingest ./knowledge-workspace ./notes/local-first.md
capicove wiki lint ./knowledge-workspace
```

Ingest preserves the exact bytes at `wiki/raw/<sha256>.md` and creates a cited
page under `wiki/pages/`. Repeating the same source is idempotent. Capicove never
replaces an existing raw capture or page; a conflicting destination fails closed
or receives a bounded digest suffix. If page publication succeeded but logging
failed, repeating the ingest repairs the missing log entry.

Create-only publication uses a hard link from a private staging file in
`wiki/_migration`. The local filesystem containing `wiki/_migration`, `wiki/raw`,
and `wiki/pages` must therefore support hard links within that filesystem. If it
does not, ingest fails closed with an explicit error, removes the private
`.create-*` file, and publishes no raw capture or page. Capicove does not fall back
to copying, replacing, or overwriting a destination.

For the generated page only, ingest removes one leading UTF-8 byte order mark and
normalizes CRLF or CR line endings to LF before reading source frontmatter. The
raw capture remains byte-for-byte identical to the supplied file.

Treat the whole Wiki as sensitive user data. Do not add a generated workspace to
version control unless every source and page was deliberately reviewed for
publication.

## Build and query

Build only after lint is clean:

```bash
capicove wiki build ./knowledge-workspace
capicove wiki query ./knowledge-workspace "local evidence" --top-k 10
capicove wiki doctor ./knowledge-workspace
```

`index.md` is a deterministic human index. Its Markdown link targets
percent-encode each Wiki-relative page path while keeping `/` separators.
`_derived/search.sqlite3` is a local, disposable FTS5 projection and can be rebuilt
from validated pages. Build and query do not use the network, models, telemetry,
or external databases.

Build tracks completion with `_migration/build.pending`. A failed build may
leave partially replaced disposable projections, but `doctor` reports unhealthy
and `query` refuses them until a complete rebuild succeeds. Re-run `build`; do
not remove the pending marker to bypass recovery. This is recoverable publication,
not a filesystem transaction spanning the database, catalog, and log.

Build, query, doctor, and index-health checks coordinate through the existing
local Wiki lock. Query and the final doctor health snapshot hold that lock across
validation and index inspection, so cooperating builds cannot split one result
across two states. A symbolic link at the derived-index path is reported as not
built and unhealthy. Query and index-health checks must be able to open
`_migration/writer.lock` for reading and writing, even though they do not change
durable Wiki content. Acquiring the lock may clean abandoned internal publication
temporary files. Version 1 does not promise operation on a read-only mount,
network filesystem, or against non-cooperating writers that replace directories
while an operation is running.

Version 1 search is lexical and uses SQLite's `unicode61` tokenizer. It is not
semantic or arbitrary substring search and has no language-specific CJK word
segmenter. Continuous CJK text may require querying the complete indexed token or
adding spaces at authoring time. The 128-token request bound is counted with this
same tokenizer; input such as `_` that produces no searchable token is rejected.

## Durable and operational files

- `raw/`, `pages/`, `graph/`, `wiki.toml`, `SCHEMA.md`, and `log.md` are durable.
- `_derived/` contains rebuildable indexes.
- `_migration/` contains private operation state, including the writer lock.
- Graph JSONL files are reviewed interchange records, not automatically accepted
  knowledge or memory events.

See the [Wiki CLI reference](../cli/wiki.md) and
[Python API](../api/wiki.md).
