# RFC-0020: Authenticated loopback HTTP for the workspace host

- Status: Accepted by the maintainer on 2026-10-06 (package WP-4 of the
  next-stage proposal (internal record; public support status is documented in the capability guide)).
- Date: 2026-10-06
- Owners: Capicove maintainers
- Related: [RFC-0017](0017-workspace-cli-mcp-host-v2.md),
  [ADR-0025](../adr/0025-workspace-loopback-http.md)

## Summary

`capicove http serve --config PATH [--port 0] [--token-file PATH]` serves the
Tools selected by one version-2 workspace configuration over HTTP on
`127.0.0.1` only. It composes the same bindings as `capicove mcp serve` and adds
no Tool, schema, or authority of its own.

## Routes

| Route | Authentication | Behavior |
| --- | --- | --- |
| `GET /v1/health` | none | `{"status": "ok"}` only |
| `GET /v1/tools` | bearer | The selected catalog, identical to `capicove mcp catalog` output |
| `POST /v1/tools/{tool_id}` | bearer | One selected canonical Tool; body = canonical arguments object; response = canonical result envelope |
| `POST /v1/search` | bearer | Alias for `capicove.search`; present only when that Tool is selected |
| `POST /v1/evidence` | bearer | Alias for `capicove.get_evidence`; present only when selected |

Responses: `200` with the canonical envelope for any completed invocation,
including failed Tool results; `400` for a body that is not a JSON object or
fails the Tool's input schema; `401` for a missing or wrong token; `404` for an
unknown or unselected Tool; `413` for a body over 2 MiB, enforced while
streaming and before parsing; `429` when an invocation is already running;
`503` once shutdown has begun; `504` when the worker exceeds the Tool deadline.
Error bodies are `{"error": {"code": ..., "message": ...}}` with stable codes and
never echo request content. No CORS headers are sent.

## Authority and isolation

- Identity, permissions, roots, tracing, and network policy come only from the
  host config. Headers and bodies cannot change them.
- The listener binds `127.0.0.1`. There is no bind-address option; a provider's
  `network.enabled` never authorizes remote serving.
- Invocations run in one worker process that composes the same bindings and is
  started in its own process group. One invocation is admitted at a time; the
  event loop stays free, so health answers during media work.
- Existing host limitations carry over unchanged, including the rejection of
  Memory Tools with `trace_store`.
- Linux only in this version: startup on platforms without the required
  no-follow, process-group, and child-subreaper primitives fails before writing
  anything.

## Token and lock lifecycle

1. Fail with exit 2 before any write when `capicove[api]` is missing or the
   catalog does not compose.
2. Take a non-blocking owner-private lock `<workspace_root>/.http.lock`. A second
   server exits 2 without touching the running server's token.
3. Generate a fresh 256-bit token. Write it to `--token-file` (default
   `<workspace_root>/http.token`), which must resolve strictly inside the
   workspace root, through pinned no-follow directories. An existing file is
   replaced atomically only when it is an owner-private regular file; links,
   foreign owners, and group/world-accessible files are refused. Mode `0600`.
4. After the listener is up, print one JSON readiness record on stderr with
   `event`, `address`, `port`, and `token_file`. The token is never printed or logged.
5. On SIGTERM or SIGINT: stop admission, terminate the worker's process group,
   escalate to SIGKILL after 5 seconds, reap it, close bindings, remove the token
   file only if it still holds this server's token, then release the lock. The
   whole shutdown is bounded by 10 seconds.
6. After SIGKILL, a restart replaces the stale token file safely through step 3.

## Acceptance

The criteria of WP-4 in the next-stage proposal apply. HTTP results are compared
with MCP results under the proposal's transport comparison policy.

## Review amendment (2026-10-06)

The maintainer review of the first implementation required these corrections, now part of this contract:

- The server reads the configuration bytes once at startup and gives those exact
  bytes, plus the workspace root's device and inode, to every worker start. A
  restarted worker can never load a replaced configuration or another workspace.
- The server is a Linux child subreaper. Processes orphaned by a dead worker
  (including FFmpeg or Whisper started in their own sessions) are stopped and
  reaped before a restart and at shutdown. HTTP serve therefore requires Linux.
- The lock and token files are opened without blocking and must be regular files;
  a FIFO planted at either path cannot stall startup or shutdown.
- Signal handlers are installed before the readiness record and kept through
  cleanup, so SIGTERM at any point still removes the token and releases the lock.
- Missing `capicove[api]` dependencies (Starlette, Uvicorn, jsonschema) exit with
  status 2 and installation guidance before any web module is imported.
