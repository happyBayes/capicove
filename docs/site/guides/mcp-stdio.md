# Run a local MCP stdio host

The configured host supports explicitly selected image, Memory, offline retrieval,
video and audio Tools, operator batch jobs, trace access and model management.
Version 2 can compose the supported Tools in one workspace; the version-1 image
configuration remains supported. Book and other standalone services retain their
own entry points and configuration. Runtime requirements are listed in the
[capability matrix](../getting-started/capabilities.md). Memory MCP has a Linux/macOS
development test; Windows runtime qualification and sanitized MCP tracing remain
separate qualifications, so this page does not claim full cross-platform support.

The image profile supports Linux, macOS, and Windows with native file security
APIs; the Memory profile currently fails closed on Windows. On Windows, use an
absolute path on a fixed local drive and a filesystem
that exposes owner and access-control information. UNC and mapped network drives
are rejected. If the required APIs or checks are
unavailable, `catalog` and `serve` stop before opening the configuration file.

Install the MCP extra explicitly in your environment:

```bash
uv pip install 'capicove[mcp]'
```

For a source checkout, use `uv sync --frozen --group dev --extra mcp`.
No models, network services, or credentials are required for the local image
profile. Help and base package imports do not load the optional MCP SDK.

## Configure one bounded workspace

For a new offline workspace, start with an explicit absolute path:

```bash
capicove setup init /absolute/private-workspace
capicove setup doctor --config /absolute/private-workspace/host.json
capicove mcp catalog --config /absolute/private-workspace/host.json
```

`setup init`, `setup doctor`, and `mcp catalog` also accept relative paths and
anchor them at the current working directory. `mcp serve` still requires an
absolute `--config` path because MCP clients may launch it from any directory;
the printed `next_commands` always use absolute paths.

`init` creates an owner-private workspace, empty `inputs/`, and a create-only
`host.json`. Its JSON output includes `next_commands` as argument arrays, so
paths containing spaces remain unambiguous on every supported OS. The
version-2 starter configuration enables only
`capicove.image.filter.shape`, with read-only authority and no network access.
Add `--source-root ALIAS=DIR` to also enable offline search; see
[Offline workspace search](offline-retrieval.md#configure-the-workspace).
`doctor` checks selected Tool composition without creating Memory storage or
downloading assets. Existing nonempty workspaces are never overwritten; use
`--use-existing` only for an empty owner-private directory. On Windows, an
inherited ACL that is not private is rejected; configure a private directory
before retrying. Additional capability configuration is an explicit operator
edit, not automatic discovery. The version-2 host composes explicitly selected
image, Memory, offline retrieval (`capicove.search`, `capicove.get_evidence`;
see [offline workspace search](offline-retrieval.md)), local video and audio
Tools (see [workspace video and audio Tools](workspace-media.md)), and operator
batch jobs, trace access, and model management (see
[operator jobs, traces, and models](operator-jobs.md)). Each capability still
needs its own local assets and permissions; an absent provider does not enable a
fallback or block unrelated Tools.

The following manual example documents the unchanged version-1 image format.

Create a UTF-8 JSON file with the following fields. Replace the example input
directory with an existing absolute directory containing images you authorize
the client to read. Use a resolved path without symlinks (including parent
directories). The config itself must be a regular file, at most 64 KiB, with no
duplicate or unknown keys. No config is discovered from the working directory.
The file must belong to the process user. On Linux and macOS it must not be
group- or world-writable. Use mode `0600` if it contains private workspace
identifiers or paths; `0644` is accepted because read access alone cannot change
startup authority. On Windows its owner must match the process user or that
token's trusted SYSTEM/Administrators default owner, and its access-control
list must not grant other users write access. Keep its parent
directories under operator control. Symlinks, junctions, and other reparse
points in the configuration path are rejected.

```json
{
  "schema_version": 1,
  "server_name": "capicove-local",
  "capability": "image",
  "workspace_id": "local-images",
  "actor_id": "local-owner",
  "environment_id": "local",
  "permissions": ["image:read"],
  "enabled_tools": ["capicove.image.filter.shape"],
  "trace_store": null,
  "capability_config": {
    "read_roots": ["/absolute/authorized-images"]
  }
}
```

All fields are required. Identity values are nonblank strings of at most 128
characters; server/profile names are lowercase identifiers. Permissions and Tool
IDs cannot repeat. The image profile accepts only `read_roots` (1–32 distinct
existing directories). It uses existing provider defaults and rejects unknown
provider settings. Available Tool IDs are:

- `capicove.image.filter.shape`
- `capicove.image.filter.size`
- `capicove.image.filter.aspect-ratio`
- `capicove.image.filter.subplot`
- `capicove.image.phash`
- `capicove.image.lsh-search`

Only explicitly selected Tools are exposed. Unsupported Tools, profiles, missing
permissions, and invalid roots fail startup; they do not produce a partial catalog.

For local Memory, select `"capability": "memory"`, an absolute dedicated
`root`, and a closed list of authorized `scopes`. The host-owned `workspace_id`
is the Memory workspace. For example, replace the image-specific fields in the
JSON above with:

```json
{
  "capability": "memory",
  "permissions": ["memory:read", "memory:write"],
  "enabled_tools": [
    "capicove.memory.remember_episode",
    "capicove.memory.recall"
  ],
  "capability_config": {
    "root": "/absolute/private/memory-workspace",
    "scopes": ["notes"]
  }
}
```

Keep the other required version-1 fields, including `"trace_store": null`.
The Memory profile accepts the existing ontology fields and the built-in
`extractor` setting, but does not load arbitrary provider factory code. The
selected root and its parents must not contain symlinks; the memory directory
must be private. `catalog` checks the selection without creating storage. `serve`
opens the repository once and closes it on shutdown. Tool arguments cannot
select another workspace or scope. `erase_scope` is never enabled by this example
and requires an explicit Tool selection and `memory:erase` startup permission.
Unlike the image profile, this Memory profile currently rejects a non-null
`trace_store` until content-free MCP tracing is verified.

## Verify and start

```bash
capicove mcp catalog --config /absolute/host.json
capicove mcp serve --config /absolute/host.json
```

`catalog` returns one deterministic JSON object with Tool names, versions, exact
schemas, and side-effect annotations. It does not start a server, invoke a Tool,
or create a trace database. With the image example above it lists exactly
`capicove_image_filter_shape`.

`serve` waits for an MCP client over stdin/stdout. It is not an interactive text
prompt and does not open a network port. Use an absolute executable path in a
client configuration, for example:

```json
{
  "mcpServers": {
    "capicove": {
      "command": "/absolute/venv/bin/capicove",
      "args": ["mcp", "serve", "--config", "/absolute/host.json"]
    }
  }
}
```

On Windows, the executable is typically `C:\\absolute\\venv\\Scripts\\capicove.exe`.
JSON paths use doubled backslashes, including `read_roots` and the `--config`
argument when written inside a client JSON file.

After initialization, call `tools/list`, then call `capicove_image_filter_shape`
with `{"sources": [{"path": "/absolute/authorized-images/sample.png"}]}`.
Expect a canonical structured result with `status: "succeeded"`, dimension
metrics, evidence references, and lineage. A file outside the configured roots
returns `permission_denied`. Tool arguments cannot widen roots or permissions.

## Diagnostics, tracing, and shutdown

In the dedicated CLI process, Python diagnostics, native stdout writes, and
inherited subprocess stdout go to stderr. A separate stream carries catalog JSON
or MCP protocol. On Windows this routing covers both the C runtime stdout
descriptor and the Win32 standard output handle. Embedders supplying a custom
output stream retain Python-only redirection and must manage native output
themselves. This is diagnostic routing, not isolation from hostile code that
deliberately accesses the protocol stream.
Startup failures return exit code 2; missing SDK errors instruct the
operator to install `capicove[mcp]`. Invalid Tool arguments and unexpected handler
errors return sanitized MCP errors without echoing submitted data.

Tracing is off by default. To enable existing sanitized Tool tracing, set
`trace_store` to an absolute database path beneath an existing owner-only
directory. On Linux and macOS use mode `0700`; on Windows the directory,
database, and SQLite sidecar files must have that same trusted owner policy and
a private access-control list. Do not use the global `--trace-store` flag with
the MCP commands. Catalog checks never create the database; serve opens it only
after capability checks pass. The trace store closes its SQLite connections per
operation. On Windows, SQLite keeps temporary trace tables and sorts in memory;
large snapshots may use more memory, and unsupported SQLite builds are rejected.
Trace records and source files stay local; retain or delete them under your
workspace policy after stopping the host.

For a Windows trace directory, create it under your local account and remove
inherited access for other users before starting the host. In PowerShell:

```powershell
$traceDir = Join-Path $env:LOCALAPPDATA 'Capicove\traces'
New-Item -ItemType Directory -Path $traceDir -Force | Out-Null
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
icacls $traceDir /grant:r "*${sid}:(OI)(CI)F"
icacls $traceDir /inheritance:r
```

The host checks the resulting owner and access-control list at startup and
before opening SQLite files. A path that still grants another user read or write
access fails closed.

The host allows at most four concurrent synchronous Tool executions. It
uses shared instances of the six configured providers, whose configuration is
read-only and working state is invocation-local. Each invocation owns its trace
session; SQLite connections are per operation. New providers must establish the
same concurrency property before joining this profile. Cancellation
and EOF do not abandon an already running handler: cleanup waits for it to return.
Deadlines are cooperative, as in the underlying Tools; this host is not a hard
process timeout or an isolation sandbox for arbitrary provider code. Ctrl-C exits
with code 130. No retry, background durable job, or model acquisition is performed.
