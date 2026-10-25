# RFC-0017: Explicit Multi-Capability Workspace CLI/MCP Host

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers
- Review scope: Version-2 startup configuration, workspace authority,
  capability composition, and setup commands. Acceptance does not approve
  individual capability Tools not covered by their own accepted contracts.
- Related decisions: [ADR-0003](../adr/0003-tool-runtime-trace-rsi-boundaries.md),
  [ADR-0004](../adr/0004-tool-resource-requirements.md).

## Summary

Introduce one explicit, offline-first workspace configuration that composes
approved capability handlers for the installed-wheel CLI and an MCP stdio host.
Canonical Tool handlers remain the business boundary. Capability factories own
providers, root validation, authorization, tracing policy, and shutdown; the
common host owns configuration parsing, identity, selected Tool registry,
transport, and lifecycle. No Tool argument may choose a provider, workspace,
permission, filesystem root, model store, or network policy.

The existing version-1 image MCP JSON format and its behavior remain valid
byte-for-byte. The version-2 format is opt-in and is never inferred from a
working directory or an environment variable. `capicove setup init WORKSPACE`
creates a private local workspace and an explicit read-only starter config;
`capicove setup doctor --config PATH` reports readiness without downloading,
building an index, starting a service, invoking a model, or changing source
media. A normal MCP startup never probes a remote endpoint or acquires assets.

## Version-2 configuration envelope

The host reads one bounded UTF-8 JSON file (at most 64 KiB) from an explicitly
supplied absolute path. Duplicate or unknown keys, nonfinite numbers, symbolic
links, and unsupported schema versions fail before capability composition. The
closed top-level fields are:

| Field | Version-2 rule |
| --- | --- |
| `schema_version` | Exact integer `2`. |
| `server_name` | Lowercase local identifier, 1 to 128 characters. |
| `workspace_id`, `actor_id`, `environment_id` | Nonblank host-owned identities, each at most 128 UTF-8 bytes. |
| `workspace_root` | Absolute local directory; private for any write profile. |
| `profile` | Exactly `read`, `write`, or `admin`. |
| `permissions` | Unique bounded permission strings; subset of the profile's reviewed ceiling. |
| `enabled_tools` | One or more unique canonical `capicove.*` Tool IDs; exact versions are resolved at startup from accepted catalogs. |
| `trace_store` | Absolute private local path or null; each capability's trace policy must permit it. |
| `capabilities` | Closed map from approved capability ID to its capability-owned configuration object with exact integer `schema_version`. |
| `network` | Closed object with `enabled` boolean and a sorted unique source-host allowlist; `enabled=false` outside `admin`. |

The common host does not accept an opaque catch-all configuration dictionary.
Each capability declares its own closed configuration schema, version, root
roles, optional extras, provider identities, and readiness check in an accepted
contract. Unknown capability IDs, fields, versions, providers, or selected
Tools fail startup. A capability object cannot override the envelope's identity,
permissions, workspace root, network setting, or trace destination. New fields
require a reviewed version change. Credentials are referenced through approved
local authentication and are never stored in this file.

The profile ceiling is monotonic: `read` allows local read Tools only; `write`
adds explicitly enabled local mutations beneath configured roots; `admin` may
add separately reviewed destructive operations, restricted exports, or network
installation. Selecting a higher profile alone enables nothing. A Tool is
registered only when its ID is listed, its exact version and required permissions
match, its factory completes, and all declared local resources are ready. The
default catalog is read-only. A client-supplied `confirm`, path, endpoint, or
permission cannot promote a Tool into another profile.

## Setup and product journeys

`capicove setup init WORKSPACE` accepts one explicit absolute path. It creates
an empty private workspace and a version-2 config with `profile=read`,
`network.enabled=false`, `trace_store=null`, and only deterministic local image
inspection enabled against a new `inputs/` directory. It does not scan existing
files, open a database, install a plugin, create an index, or make a network
request. Existing nonempty or symlinked roots are rejected; existing empty
private roots are reused only with explicit `--use-existing`. The output lists
the config path and the next commands needed to enable other capabilities.
Config writes are create-only, atomic, and owner-private. `init` never rewrites
an existing config or changes its authority silently.

`capicove setup doctor --config PATH` validates config security, Tool selection,
local roots, required extras/binaries, local model manifests and hashes, index
generation receipts, and provider readiness through bounded, read-only checks.
It returns structured `ready`, `not_configured`, `missing_dependency`,
`asset_mismatch`, `unsafe_path`, or `unavailable` entries with exact local
remediation commands. It never prints source content, secrets, or token values.
An unconfigured optional capability does not break unrelated ready Tools. The
exit code is zero only if every selected Tool is ready; otherwise it is nonzero.

The first required offline journey is: install a wheel outside the source tree,
initialize a workspace, explicitly import Markdown and an image through approved
operator commands, build a local generation, search it, open source-linked
Evidence, change one source, rebuild, restart, and repeat Search/Evidence through
MCP. A configured multimodal journey adds explicitly provisioned PDF, audio,
video, parser, and model assets. An ordinary query or MCP startup never triggers
the import/build/download steps. Existing bare CLI commands that require injected
handlers gain `--runtime-config PATH` rather than guessing a provider.

## Runtime and transport boundary

The host composes capability factories in a deterministic order and keeps an
`ExitStack` for their process lifetime. A factory consumes only the immutable
host authority and its own validated capability configuration, returns public
Tool bindings, and registers cleanup before a later factory starts. If any
factory fails, all previously opened resources close and no partial MCP catalog
is served. `mcp catalog` validates selected local resources without opening
trace databases, creating workspace state, invoking model code, or starting an
MCP server. A catalog response contains exact names, Tool IDs/versions, input
and output schemas, side-effect annotations, provider identity, and
configuration readiness. It validates local manifests/files and selected
resources, not model quality or runtime qualification; it does not claim a Tool
is ready if its required local provider is absent.

The CLI and MCP use the same handler and startup policy. The MCP transport
imports its optional SDK only for `serve`, reserves stdout for protocol data,
sanitizes unexpected failures, bounds synchronous concurrency, and drains active
work on cancellation or EOF. Provider diagnostics go to stderr or the selected
sanitized trace store. Memory source content and other restricted payloads use
capability-specific content-free trace projection; generic tracing is not
silently substituted. A Tool result carries the canonical envelope and evidence
references, and business output is unchanged across direct Handler, CLI, and
MCP invocation. An explicit admin job worker under RFC-0016 is independent of
the stdio server lifetime.

## Filesystem, platform, and network policy

The CPU profile is required on Linux, macOS, and Windows. POSIX startup rejects
symlink components, wrong file types, other-user-writable host configs, and
nonprivate writable roots. Windows startup uses native handle/owner/ACL checks,
rejects reparse points, alternate data streams and network/removable drives
where the profile requires a fixed local drive, and checks private inheritance
for SQLite sidecars and output files. A fallback based only on `Path.resolve()`
or POSIX mode bits is not sufficient. Source files are opened beneath their
selected roots and revalidated against their immutable hashes; outputs cannot
alias input files or escape through a symlink. Unsupported security APIs fail
closed before a capability is advertised. Cross-platform support is claimed
only after installed-wheel acceptance on all three OSes; an unqualified
capability blocks the full-profile release rather than being silently omitted.

`network.enabled=false` is the default and forbids automatic HTTP clients,
model downloads, plugin installation, external APIs, and telemetry. An admin
configuration may allow specific source hosts for a reviewed operation, but a
Tool request cannot add a host or enable network access. Downloaded assets are
pinned to immutable revisions and verified hashes. Team-managed Hugging Face
transfers still require exact repository/type and file-scope authorization under
AGENTS.md; this RFC grants no blanket transfer authority.

## Compatibility, failure semantics, and acceptance

Version-1 image configurations continue to select only their existing image
profile with unchanged fields, six deterministic providers, names, and error
behavior. Version 2 is an additive host format; there is no automatic migration
or implicit discovery. Existing CLI commands and capability-specific standalone
MCP creators remain available until their separately reviewed deprecation.
Unknown or future config versions fail with an actionable error. Invalid
configuration never produces a partially authorized server. Readiness failure
does not silently substitute a provider, backend, profile, model, or index.

Acceptance requires byte-compatible version-1 catalog/stdio tests, version-2
round trips for every selected approved capability, scope/path/permission
negative tests, startup failure cleanup, no startup network or hidden optional
imports, stdout protocol isolation, restart and evidence identity checks, and
installed-wheel CPU tests on Linux, macOS, and Windows. Hardware acceleration
is required only for explicitly supported hardware profiles. Required skipped,
failed, or blocked QA cases prevent a full-function release claim. `make
quality`, complete `make test` with the repository coverage threshold, `make
build`, and `make docs-check` must pass before release.

## Alternatives considered

- Extend version-1 image fields in place: rejected because existing users need
  a stable configuration and image-only startup semantics.
- Auto-discover capabilities and models: rejected because it creates hidden
  network, dependency, privacy, and startup side effects.
- Let each transport compose its own providers: rejected because CLI/MCP
  authorization, evidence identity, and shutdown would drift.

## Amendment 1: offline search starter (2026-10-08)

- Status: Accepted by the maintainer on 2026-10-08, as phase 2 of the
  release-readiness repair. Requested and approved in the implementation
  session; no other contract changes.
- Scope: one repeatable `setup init` option. The version-2 envelope, the
  six-field retrieval configuration of
  [RFC-0018](0018-offline-workspace-retrieval.md), and the default starter are
  unchanged.

`capicove setup init WORKSPACE [--use-existing] [--source-root ALIAS=DIR ...]`

Without `--source-root`, the output is exactly the starter described above.
With one or more `--source-root` options, the starter also enables offline
search:

- `profile` is `write`, because importing sources is an operator write under
  RFC-0018; `network.enabled` stays `false` and `trace_store` stays `null`.
- `permissions` are `evidence:read`, `evidence:search`, and `image:read`;
  `enabled_tools` are `capicove.get_evidence`, `capicove.image.filter.shape`, and
  `capicove.search`.
- `capabilities.retrieval` holds the given aliases and directories,
  `index_root` = `WORKSPACE/index`, `corpus_ref` = `corpus:<workspace name>`,
  `retrieval_profile_ref` = `local-default`, and `max_source_bytes` = 1 MiB.
- Relative directories are anchored at the caller's working directory, like
  `WORKSPACE`. Duplicate aliases, a missing `=`, or any value RFC-0018 rejects
  fail with exit status 2 before any directory or file is created.
- `next_commands` lists `setup doctor`, `source import` (with `RELATIVE_PATH`
  as a placeholder for the first alias), `index build`, and `search` (with a
  `QUERY` placeholder).

The option only stats the named directories; it does not read or list them.
It does not create the index, open a database, or make a network request.
Retrieval remains POSIX-only, so on other platforms the option fails closed
with exit status 2 and the default starter remains available.
