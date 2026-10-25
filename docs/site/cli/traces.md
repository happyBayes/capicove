# Trace export CLI

The `traces` command exports complete Agent operation trace episodes to portable,
model-neutral JSONL and validates or inspects an export package.

```bash
capicove traces export --help
capicove traces validate DIRECTORY
capicove traces inspect DIRECTORY
```

Export requires an explicit local SQLite trace store and a new output directory.
Use the filters shown by `capicove traces export --help` to select bounded complete
episodes. Validation checks package structure, hashes, and referenced media by
default; use `--no-verify-media` only when media verification must be skipped.

See the [Agent operation trace export guide](../guides/agent-trace-export.md) for the
complete workflow and privacy considerations.

## Workspace host mode

With `--runtime-config PATH`, the three commands invoke the configured
`capicove.trace.*` Tools instead of opening paths directly, and print the
canonical result envelope as JSON:

```text
capicove traces inspect  --runtime-config PATH (--store ALIAS | EXPORT_ID)
capicove traces validate --runtime-config PATH EXPORT_ID [--no-verify-media]
capicove traces export   --runtime-config PATH --store-alias ALIAS --output-root ALIAS
                         --idempotency-key KEY [filters] [--include-incomplete]
                         [--include-restricted] [--media-mode reference|copy]
```

`--store`, `--output`, and `--media-root` paths cannot be combined with this mode;
the configuration fixes every store, export root, and media root. See the
[operator guide](../guides/operator-jobs.md#trace-inspection-and-export).
