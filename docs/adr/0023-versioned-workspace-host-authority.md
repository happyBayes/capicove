# 0023: Compose capabilities from one versioned workspace authority

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers

## Context

The existing image MCP host has a stable version-1 configuration. Other
capabilities have public handlers but no single installed-wheel composition and
authorization boundary shared by CLI and MCP. Implicit discovery would also
make optional dependencies, model assets, and network behavior unpredictable.

## Decision

The common runtime will retain version-1 image
behavior and add an opt-in version-2 workspace envelope. The envelope owns
identity, profile ceiling, enabled Tool IDs, local roots, tracing, and network
policy. Each capability owns a closed versioned configuration, resource checks,
public Tool bindings, and cleanup. Both CLI and MCP will use the same handlers
and startup authority. Unknown or unsafe configuration fails before any Tool is
advertised. Setup initialization creates only a private, read-only offline
starter workspace; readiness checks do not download or execute models.

## Consequences

- New capabilities can be added without changing another capability's private
  implementation or silently changing the version-1 image catalog.
- Startup needs deterministic factory order and rollback of opened resources.
- Cross-platform path and permissions checks are part of the security boundary;
  unsupported Windows checks must fail closed, not fall back to POSIX semantics.
- A full-function release requires installed-wheel acceptance on every claimed
  platform, not just unit tests in the source tree.

## Alternatives considered

- Add optional fields to the version-1 image config: rejected because it could
  change established startup semantics.
- Let each transport compose providers independently: rejected because CLI and
  MCP authorization and evidence identity could diverge.

## References

- [RFC-0017: Explicit multi-capability workspace CLI/MCP host](../rfcs/0017-workspace-cli-mcp-host-v2.md)
- [ADR-0003: Unify Tool Runtime, Trace, and controlled RSI boundaries](0003-tool-runtime-trace-rsi-boundaries.md)
