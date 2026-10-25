# RFC-0016: Bounded Operator Jobs, Trace Access, and Model Installation

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers
- Review scope: New Agent-facing operator Tool contracts, durable job receipts,
  and administrative security boundaries. Acceptance does not approve a remote
  listener, arbitrary process execution, or unrestricted network access.
- Related decisions: [ADR-0003](../adr/0003-tool-runtime-trace-rsi-boundaries.md),
  [ADR-0012](../adr/0012-agent-trace-recording-export-boundaries.md).

## Summary and ownership

The [RFC-0023 model factory amendment](0023-unified-model-factory.md) generalizes
installer discovery through `capicove.model_installers`, retaining legacy parser
installers, Tool contracts, version-1 configuration, plan digests and installation
authority. Offline inference profiles are separately configured and grant no
download permission.

Expose existing directory batch, trace inspection/export, and model management
through bounded canonical Tools and equivalent CLI commands. `capicove.batch`
continues to own per-item planning, checkpoints, and results. A small
workspace-owned job registry owns only durable scheduling, worker leases,
idempotency, and job status. `capicove.traces` and `capicove.export` retain trace
and package semantics. Model installers remain capability-owned plugins; this
RFC adds an operator authorization wrapper, not a second installer.

The ordinary MCP host never launches a background daemon, installs a model,
deletes inputs, copies restricted traces, or enables network access on startup.
A disconnected MCP client cannot make a queued or running job unobservable: job
identity and status are durably committed before `enqueue` returns, and work is
performed only by a separately started local worker process. No generic shell
command, Python import path, URL, path root, provider, or credential may be
supplied through a Tool request to expand startup authority.

## Closed version-1 Tool catalog

All listed IDs have contract version 1, closed JSON input schemas, canonical
Tool result envelopes, and host-owned identity. Existing CLI syntax remains
available; new commands delegate to the same public services. Read-only Tools
may be selected in a normal profile. Write Tools require explicit writable
roots and permissions. Administrative Tools require an administrative profile,
their exact Tool ID, and operation-specific authorization.

| Tool ID | Required selectors | Side effect and permission |
| --- | --- | --- |
| `capicove.batch.plan` | Configured input-root alias, closed operation ID, parameters, provider fingerprint | Bounded read; `batch:read` |
| `capicove.batch.enqueue` | Plan ID, idempotency key | Durable queue write; `batch:write` |
| `capicove.batch.status` | Job ID, offset, limit (at most 100 items) | Read; `batch:read` |
| `capicove.batch.cancel` | Job ID, expected revision | Durable cooperative cancellation request; `batch:write` |
| `capicove.trace.inspect` | Configured trace-store alias or export ID | Bounded read; `trace:read` |
| `capicove.trace.validate` | Configured export ID, media-verification flag | Bounded read; `trace:read` |
| `capicove.trace.export` | Filter, configured output-root alias, media mode, idempotency key | New local package write; `trace:export`, administrative when restricted or copying media |
| `capicove.model.plan` | Installed plugin ID, profile ID, configured destination alias | Read manifest and plan; `model:read` |
| `capicove.model.verify` | Installed plugin ID, profile ID, configured destination alias | Read hashes and receipt; `model:read` |
| `capicove.model.install` | Immutable plan digest, destination alias | Network/local asset write; `model:install`, administrative network profile |

No Tool accepts an absolute path except a relative media reference already
resolved under a startup-authorized root. The host maps aliases to validated
absolute local roots; caller values may only narrow them. Each Tool caps array
counts, string and response byte lengths, concurrency, and deadlines in its
schema and service validator. Pagination uses opaque, scope-bound cursors or
integer offsets with a fixed stable order. Unknown fields and nonfinite values
are rejected before side effects. Error messages do not include raw trace
content, private paths, credentials, or model download tokens.

## Batch planning and durable jobs

`batch.plan` discovers only regular files beneath one configured input root,
using existing modality and extension rules. It records source snapshots and
content hashes, operation/parameter/provider fingerprints, item count, bounded
estimated work, and a deterministic `plan:<sha256>` ID. A plan is immutable and
has no execution side effect. The host rejects operations or providers not
selected at startup. `enqueue` atomically records the plan identity, authorized
output root, scoped idempotency key, and job revision before returning
`job:<24 lowercase hex>`. The same key and plan return the same job; a different
plan with that key returns `idempotency_conflict`.

The job registry has schema version 1 and states `queued`, `running`,
`cancelling`, `interrupted`, `succeeded`, `failed`, and `cancelled`. It stores
workspace, plan hash, operation fingerprint, selected roots by immutable alias
and hash, creation/update times, revision, worker lease, bounded counters,
terminal manifest hash, and sanitized error code. It does not store credentials,
source content, or a copy of per-item results. Existing BatchStore checkpoints
remain the item authority. A worker uses a compare-and-swap lease; on process
loss or expiry, startup marks the job `interrupted` and releases the lease only
after verifying the checkpoint. Explicit resume with the same job ID rechecks
source snapshots and provider fingerprint. Changed items become
`source_changed`; they are never silently reused. Cancellation is cooperative,
idempotent, revision-checked, and cannot undo completed output. A terminal job
cannot be restarted under the same ID.

`capicove jobs worker --config PATH` is an operator-started local process using
the same bounded configuration. MCP EOF does not start, stop, or own that worker.
If no worker is running, an enqueued job remains `queued` and status explains
the missing worker. The worker has fixed concurrency and operation allowlists;
it cannot execute arbitrary user code. Publication of a terminal manifest is
recoverable: partial artifacts and a pending receipt are not reported as
success, and restart verifies the manifest hash before marking `succeeded`.

## Trace read and export policy

`trace.inspect` and `trace.validate` accept only configured stores or export
roots and return bounded metadata, validation findings, and content-free status.
They do not expose raw message bodies or source paths. `trace.export` reuses the
accepted RFC-0006/ADR-0012 JSONL format and `TraceExporter`; it never mutates
recorded Episodes. Its default `reference` media mode excludes restricted
Episodes. Including restricted Episodes requires the administrative profile,
`trace:export_restricted`, an exact configured workspace scope, and an explicit
`include_restricted=true` request. `copy` media mode additionally requires a
configured media root and `trace:copy_media`. The output is a new directory
under the selected private export root; existing destinations are never
overwritten. An idempotent retry validates and returns a matching completed
manifest, or reports a recoverable pending export. No Tool can export to an
arbitrary client path or remote endpoint.

## Model plan, verification, and installation policy

`model.plan` returns an immutable manifest digest, plugin and profile IDs,
pinned upstream repository/revision, source hosts, file paths, sizes, SHA-256
values, license/provenance metadata, destination alias, and estimated bytes.
`model.verify` reads an existing installation without downloading anything and
returns missing, mismatched, and verified file counts plus receipt identity.
`model.install` accepts only the digest of a plan generated for the same
workspace and destination. The administrative startup configuration must
allowlist that exact plan digest, plugin/profile, destination, source hosts,
and network access. A new plan digest requires an explicit operator edit and
restart; a Tool argument cannot approve it. The installer uses approved local
credentials without
placing tokens in Tool arguments, source files, or logs. It pins immutable
revisions, verifies every file hash and redistribution/usage rights recorded in
the reviewed manifest, writes a receipt, and atomically publishes the qualified
directory. A failed or partial download remains unqualified and cannot be
silently selected by runtime composition. No fallback to a personal or public
Hugging Face repository is permitted; private repository transfers
still require the separate exact-target authorization described in AGENTS.md.

## Security, compatibility, and acceptance

The default catalog may contain `batch.plan`, `batch.status`,
`trace.inspect`, `trace.validate`, `model.plan`, and `model.verify` only when
their local resources are configured. `batch.enqueue` and `batch.cancel` are
explicit write-profile Tools. Restricted trace export, media copying, and model
installation are never included in the default profile. Model installation
requires an explicitly network-enabled administrative profile; a client
argument cannot turn network access on. Destructive video duplicate deletion,
Memory scope erasure, and Wiki migration are not authorized by this RFC.

Existing batch item state, trace JSONL format, model plugin contracts, and CLI
commands retain their meaning. The new job registry is versioned, private
operational state; unknown versions fail closed. No automatic data migration,
public remote worker API, service manager, or telemetry is introduced.

Acceptance tests must cover durable status after client disconnect and host
restart, no-worker diagnosis, lease loss, repeated idempotency keys, changed
sources, interrupted publication, bounded pagination, foreign workspace/root
rejection, restricted trace denial, copied-media authorization, missing model
files, hash mismatch, denied network and wrong plan digest. CLI/MCP results and
installed-wheel behavior must agree on each claimed OS. A synthetic provider
test does not qualify real model performance or license rights. Complete
`make quality`, `make test`, `make build`, and `make docs-check` gates remain
required for release.

## Alternatives considered

- Run long batch work inside one MCP call: rejected because a disconnected
  client or host exit would leave unobservable partial work.
- Expose the existing CLI argument parser as an MCP Tool: rejected because
  arbitrary paths, plugin names, and network confirmations would widen startup
  authority.
- Put TraceExporter or model installer logic into the common MCP host:
  rejected because it would cross capability ownership boundaries.

## Amendment 1: workspace host configuration (2026-10-09)

- Status: Accepted by the maintainer on 2026-10-09.
- Scope: the closed version-2 host capability objects that hold the "configured"
  roots and aliases this RFC refers to, the profile permission ceilings, the
  `capicove jobs` commands, and batch video operations. Tool IDs are unchanged.

### `capabilities.batch` (schema version 1)

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1` |
| `input_roots` | 1–16 lowercase aliases → absolute existing directories without symlinks; read-only sources |
| `output_roots` | 1–16 aliases → absolute owner-private directories strictly inside `workspace_root`, disjoint from inputs |
| `registry_root` | Absolute owner-private directory strictly inside `workspace_root` holding the job registry and checkpoints |
| `operations` | Non-empty subset of `image_aspect_ratio_filter`, `image_shape_filter`, `image_size_filter`, `image_subplot_filter`, `video_filter`, `video_split` |
| `provider_fingerprints` | Object mapping every listed operation to a lowercase SHA-256; host-owned |
| `max_items` | Integer 1 through 100,000 per plan |
| `worker` | `{"concurrency": 1–8, "lease_seconds": 30–3600, "poll_seconds": 0.5–60}` |

Video operations additionally require `capabilities.video` in the same
configuration. They use its pinned executables, backend, staging root, and input
bound (RFC-0021, ADR-0026); batch input roots act as the read roots, and split
outputs go to a per-item directory inside the selected batch output root through a
pinned directory handle. Filter criteria are limited to `metadata` and `motion`.

`batch.plan` takes `input_root`, `operation`, `parameters`, and
`provider_fingerprint` (which must equal the configured value). `batch.enqueue`
takes the same selectors plus `output_root`, `plan_id`, and `idempotency_key`; the
host recomputes the plan and rejects a changed plan before any write.
`batch.status` takes `job_id` with an `offset` and `limit` (at most 100 items);
`batch.cancel` takes `job_id` and `expected_revision`.

### `capabilities.trace` (schema version 1)

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1` |
| `stores` | 1–16 aliases → absolute existing private trace SQLite files |
| `export_roots` | 1–16 aliases → absolute owner-private directories strictly inside `workspace_root` |
| `media_root` | Absolute existing directory or `null`; required only for `copy` media mode |

### `capabilities.model` (schema version 1)

| Field | Rule |
| --- | --- |
| `schema_version` | Exactly `1` |
| `destinations` | 1–16 aliases → absolute owner-private directories strictly inside `workspace_root` |
| `install_allowlist` | Array of `{"plan_digest", "plugin", "profile", "destination", "source_hosts"}`; empty unless installation is intended |

### Permissions and profiles

| Profile | Adds |
| --- | --- |
| read | `batch:read`, `trace:read`, `model:read` |
| write | `batch:write`, `trace:export` |
| admin | `trace:export_restricted`, `trace:copy_media`, `model:install` |

`capicove.model.install` additionally requires `network.enabled`, which already
requires the admin profile, and its plan's source hosts must appear in
`network.source_hosts`.

### Commands

`capicove jobs plan|enqueue|status|cancel --runtime-config PATH` invoke the batch
Tools through the same host bindings as MCP; `capicove jobs worker --config PATH`
is the operator-started worker. `capicove traces inspect|validate|export` and
`capicove models plan|verify|install` gain a `--runtime-config PATH` mode with the
same rule.

### Implementation notes (2026-10-09)

These refine the accepted amendment without changing its Tool IDs or fields.

- **Platform.** The batch, trace, and model host capabilities are POSIX-only in
  this release, like the other workspace capabilities with private roots.
- **Root identity.** A job binds each root by alias and
  `sha256("capicove.batch.root.v1\0<alias>\0<absolute path>")`; changing the path
  behind an alias makes existing jobs unroutable instead of silently redirecting
  them.
- **Plan selectors.** `batch.plan` also accepts an optional `output_root`; it is
  required for `video_split`, whose split outputs are part of the operation
  fingerprint. Video parameters are validated against the video Tool input
  schema at plan time, and split uses the backend from `capabilities.video`
  rather than a request value.
- **Operation descriptors.** `batch.enqueue` writes a create-only, owner-private
  descriptor `registry_root/operations/<operation digest>.json` holding the
  operation ID and canonical parameters. The worker rebuilds the operation from
  it and requires the recomputed digest to equal the job's recorded fingerprint;
  a removed operation, changed provider fingerprint, removed alias, or edited
  descriptor leaves the job durable and skipped, never re-routed.
- **Worker.** `capicove jobs worker --config PATH` validates the whole selected
  catalog as MCP startup does, requires `batch:write`, pins video executables
  for its lifetime, recovers expired leases first, then runs the oldest routable
  queued job. The first SIGINT/SIGTERM finishes the current job; a second one
  aborts and leaves the lease to expire. `--once` exits when idle. Events are
  JSON lines on stderr.
- **Resume.** Expired leases become `interrupted` (unchanged registry
  semantics). The operator-only `capicove jobs resume --config PATH JOB_ID
  --expected-revision N` requeues a job after checkpoint and source verification;
  changed sources fail with `source_changed`. It is not an MCP Tool.
- **Status.** `batch.status` returns items in plan order from the job's own
  checkpoint database (items never claimed report `pending`), each with a
  failure code and, when succeeded, the cached Tool result. A `queued` job
  carries a diagnostic naming the worker command; an `interrupted` job names
  `jobs resume`.
- **Trace Tools.** `trace.inspect` takes exactly one of `store` or `export_id`;
  `trace.validate` takes `export_id` and `verify_media` (default `true`);
  `trace.export` takes `store`, `output_root`, `idempotency_key`, `media_mode`,
  the existing `TraceQuery` filters, and `include_incomplete` /
  `include_restricted`. Export IDs are `export:<32 hex>` and are resolved only
  inside configured export roots. Restricted and copy denials happen before any
  store is opened or root created.
- **Model Tools.** Destinations are `<destination root>/<plugin>/<profile>`.
  `plan_digest` is the SHA-256 of the canonical plan (workspace ID, plugin,
  profile, profile version, destination alias, manifest digest, models, files,
  source hosts, size), so it binds the workspace and destination. Plans never
  expose local paths. `model.verify` hashes every planned file without network
  access and reports verified, missing, and mismatched counts; symbolic links
  count as mismatched. `model.install` recomputes the plan, requires the exact
  allowlisted digest and source hosts, installs through the plugin, and reports
  failure unless the post-install verification is complete.
