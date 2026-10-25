# Operator jobs, traces, and models

A version-2 workspace configuration can expose durable batch jobs, trace
inspection and export, and model plan/verify/install as host Tools. MCP
(`capicove mcp serve`), HTTP (`capicove http serve`), and the configured CLI
(`--runtime-config`) all use the same bindings, roots, and permissions. Only the
separately started `capicove jobs worker` process executes queued jobs.

!!! note "Current status"
    These capabilities implement the accepted
    [RFC-0016 amendment 1](https://github.com/happyBayes/capicove/blob/main/docs/rfcs/0016-operator-jobs-traces-and-models.md#amendment-1-workspace-host-configuration-2026-10-09)
    and require Linux or macOS (batch video operations require Linux, like the
    other host media Tools). Synthetic-provider tests do not qualify real model
    performance or license rights.

## Batch jobs

### Configure

Add `capabilities.batch` to the host configuration:

```json
{
  "schema_version": 1,
  "input_roots": {"photos": "/home/me/photos"},
  "output_roots": {"results": "/home/me/cove/batch-results"},
  "registry_root": "/home/me/cove/jobs",
  "operations": ["image_size_filter", "video_split"],
  "provider_fingerprints": {
    "image_size_filter": "<64 hex>",
    "video_split": "<64 hex>"
  },
  "max_items": 10000,
  "worker": {"concurrency": 2, "lease_seconds": 120, "poll_seconds": 2}
}
```

- Input roots are existing directories without symlinks; they are never written.
- Output roots and the registry root are owner-private directories strictly
  inside `workspace_root`, disjoint from each other and from the inputs. They are
  created on the first enqueue, never at startup.
- `operations` is a closed allowlist: `image_aspect_ratio_filter`,
  `image_shape_filter`, `image_size_filter`, `image_subplot_filter`,
  `video_filter`, `video_split`. Each needs a host-owned provider fingerprint;
  change it when the provider changes so cached results are not reused.
- `video_filter` and `video_split` also need `capabilities.video`; the batch
  input roots act as its read roots, and its pinned FFmpeg/FFprobe, backend,
  staging root, and input-size bound apply. Filter criteria are limited to
  `metadata` and `motion`.

Select the Tools and permissions:

| Tool | Permission | Profile |
| --- | --- | --- |
| `capicove.batch.plan`, `capicove.batch.status` | `batch:read` | read |
| `capicove.batch.enqueue`, `capicove.batch.cancel` | `batch:write` | write |

### Plan, enqueue, and run

```bash
capicove jobs plan --runtime-config host.json --input-root photos \
  --operation image_size_filter --parameters '{"max_size": "8MiB"}' \
  --provider-fingerprint <64 hex>
capicove jobs enqueue --runtime-config host.json --input-root photos \
  --operation image_size_filter --parameters '{"max_size": "8MiB"}' \
  --provider-fingerprint <64 hex> --output-root results \
  --plan-id plan:<sha256> --idempotency-key nightly-2026-10-09
capicove jobs worker --config /home/me/cove/host.json
```

`plan` hashes every source and returns a deterministic `plan:<sha256>` with a
preview of at most 100 items; it writes nothing. `enqueue` recomputes the plan and
returns `plan_changed` if any source changed, then durably records the job before
returning `job:<24 hex>`. Repeating the same key returns the same job; using the
key for a different plan returns `idempotency_conflict`.

If no worker is running, the job stays `queued` and `status` says so. Start the
worker in a terminal or a service manager you control; MCP clients cannot start or
stop it. Use `--once` to drain the queue and exit. The first Ctrl-C or SIGTERM
finishes the current job; a second one aborts it.

```bash
capicove jobs status --runtime-config host.json job:<24 hex> --offset 0 --limit 100
capicove jobs cancel --runtime-config host.json job:<24 hex> --expected-revision 1
```

`status` pages items in plan order with their state, attempts, failure code, and
(for succeeded items) the Tool result. `cancel` is revision-checked: a queued job
becomes `cancelled`; a running job becomes `cancelling` and the worker stops
before starting more items. Completed outputs are never undone.

### Recovery

If a worker dies, its lease expires and the next worker marks the job
`interrupted` after verifying the checkpoints. Resume it explicitly:

```bash
capicove jobs resume --config /home/me/cove/host.json job:<24 hex> --expected-revision 4
```

Resume rechecks every source; a changed source fails with `source_changed`
instead of reusing stale work. A job whose operation, provider fingerprint, or
root alias was later removed or changed in the configuration is never re-routed:
the worker reports `job_skipped` once and leaves it durable.

## Trace inspection and export

```json
{
  "schema_version": 1,
  "stores": {"local": "/home/me/cove/traces/traces.sqlite3"},
  "export_roots": {"exports": "/home/me/cove/trace-exports"},
  "media_root": null
}
```

| Tool | Permission | Notes |
| --- | --- | --- |
| `capicove.trace.inspect` | `trace:read` | `{"store": "local"}` counts Episodes; `{"export_id": "export:…"}` inspects a package |
| `capicove.trace.validate` | `trace:read` | Validates a package in a configured export root |
| `capicove.trace.export` | `trace:export` | Writes a new idempotent package |

Results contain counts, validation codes, and opaque export IDs, never trace
content or paths. Exports exclude restricted Episodes unless the profile is
`admin`, `trace:export_restricted` is granted, and the request sets
`include_restricted`. Copy media mode additionally needs `trace:copy_media` and a
configured `media_root`. Retrying the same key returns the same package; changing
the filter for that key returns `idempotency_conflict`.

```bash
capicove traces inspect --runtime-config host.json --store local
capicove traces export --runtime-config host.json --store-alias local \
  --output-root exports --idempotency-key weekly
capicove traces validate --runtime-config host.json export:<32 hex>
```

## Model plan, verify, and install

```json
{
  "schema_version": 1,
  "destinations": {"models": "/home/me/cove/models"},
  "install_allowlist": []
}
```

`capicove.model.plan` and `capicove.model.verify` (`model:read`) never use the
network. A plan returns the pinned repositories, revisions, files, hashes, source
hosts, and a `plan_digest` bound to this workspace and destination alias. Verify
hashes the installed files and reports verified, missing, and mismatched counts.

```bash
capicove models plan --runtime-config host.json paddleocr --profile book-balanced \
  --destination models
capicove models verify --runtime-config host.json paddleocr --profile book-balanced \
  --destination models
```

Installation is never authorized by a request. To install, review the plan, then
edit the configuration: select `capicove.model.install`, use the `admin` profile
with `model:install`, enable `network` with the plan's source hosts, and add the
exact entry:

```json
{
  "plan_digest": "<64 hex>",
  "plugin": "paddleocr",
  "profile": "book-balanced",
  "destination": "models",
  "source_hosts": ["huggingface.co"]
}
```

```bash
capicove models install --runtime-config host.json --plan-digest <64 hex> \
  --destination models
```

The host recomputes the plan and refuses a different digest (`plan_changed`).
The plugin downloads with your existing local credentials; tokens never appear in
arguments or logs. The result reports success only when every planned file
verifies. Private transfers require explicit authorization for the exact
repository, destination and file scope.
