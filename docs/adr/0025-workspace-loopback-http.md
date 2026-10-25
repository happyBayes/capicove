# 0025: Serve workspace Tools over loopback HTTP through an isolated worker

- Status: Accepted
- Date: 2026-10-06
- Owners: Capicove maintainers

## Context

The existing `create_search_app` builds a search-only HTTP application with
synthetic identity and permissions, and nothing launches it. RFC-0020 defines a
launchable HTTP transport over the version-2 workspace host.

## Decision

- HTTP reuses `compose_mcp_host` bindings and the shared binding invocation path
  that MCP uses, so validation, context creation, tracing, and result identity
  checks are the same code.
- Tool calls run in one long-lived worker process created in a new process
  group. The HTTP process keeps the catalog, authentication, body limits, and
  admission control. Killing the worker's process group is the only reliable
  way to stop in-flight work, including FFmpeg or Whisper children, within a
  bounded shutdown budget; threads cannot be cancelled.
- The listener is loopback-only with a per-start bearer token in an
  owner-private file and a per-workspace server lock.

## Consequences

- One invocation at a time per server; excess calls get `429` instead of queueing.
- A killed worker is restarted for the next call; its in-flight call reports a
  stable error.
- Windows needs a reviewed job-object equivalent before it can be supported.

## References

- [RFC-0020: Authenticated loopback HTTP for the workspace host](../rfcs/0020-workspace-loopback-http.md)
- [ADR-0023: Compose capabilities from one versioned workspace authority](0023-versioned-workspace-host-authority.md)

## Review amendment (2026-10-06)

Workers receive frozen configuration bytes and the workspace root identity, not a
path. The server enables Linux child-subreaper mode so orphaned media processes
are reaped across worker failure, opens its lock and token without blocking, and
owns SIGTERM/SIGINT from startup through cleanup. HTTP serve is Linux-only.
