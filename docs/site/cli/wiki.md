# Wiki commands

The `wiki` command manages one standalone `wiki/` directory beneath the supplied
workspace. Every successful command emits deterministic JSON.

```text
capicove wiki init WORKSPACE [--date YYYY-MM-DD]
capicove wiki ingest WORKSPACE SOURCE [--date YYYY-MM-DD]
capicove wiki scan WORKSPACE
capicove wiki lint WORKSPACE
capicove wiki build WORKSPACE [--date YYYY-MM-DD]
capicove wiki query WORKSPACE QUERY [--top-k 1..100]
capicove wiki doctor WORKSPACE
capicove wiki migrate-v2 WORKSPACE [--apply]
```

`--date` defaults to the current local date. It is primarily useful for
deterministic imports, tests, and migrations. `query` accepts at most 2,048
characters, 8,192 UTF-8 bytes, 128 tokens from the index's SQLite `unicode61`
tokenizer, and 100 results. Input that produces no tokenizer term is rejected.

`migrate-v2` previews the legacy-workspace migration unless `--apply` is supplied.
Back up the workspace and inspect the preview before applying a migration.

## Exit status

- `0`: the operation succeeded, or lint/doctor found no errors.
- `1`: lint or doctor completed and reported an unhealthy Wiki or index.
- `2`: invalid input, a missing optional dependency, or an operation failure.

Operation errors use a JSON `error` object with a stable category such as `wiki_exists`,
`wiki_not_found`, `wiki_unsafe_path`, `wiki_dependency_unavailable`,
`wiki_format_invalid`, `wiki_validation_failed`, `wiki_index_unavailable`,
`wiki_invalid_request`, or `wiki_operation_failed`. Command-line syntax errors are
reported by the standard argument parser on stderr with exit status `2`.

Run `lint` before `build`. Run `build` again after editing or ingesting pages if
`query` or `doctor` reports a missing, corrupt, incompatible, or stale index.
An interrupted or failed build also requires a successful rebuild before queries
resume. Keep its `_migration/build.pending` recovery marker in place. Query and
index-health checks require read/write access to the existing Wiki lock file;
they may clean internal temporary files but do not modify durable Wiki content.
Doctor reports lint, regular non-symlink index presence, and index health from one
coordinated final snapshot for a valid layout; an invalid layout remains a normal
unhealthy report.

See the [standalone local Wiki guide](../guides/local-wiki.md) for the complete
workflow and privacy boundary.
