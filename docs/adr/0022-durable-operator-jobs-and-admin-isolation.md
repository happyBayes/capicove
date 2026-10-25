# 0022: Separate durable operator jobs from MCP host lifetime

- Status: Accepted
- Date: 2026-09-28
- Owners: Capicove maintainers

## Context

Directory batch work can outlive a Tool call, while trace export and model
installation have more sensitive filesystem and network effects than ordinary
read Tools. Treating an MCP stdio session as their lifecycle owner would make
disconnect recovery and authorization unclear.

## Decision

A workspace-owned versioned registry will record
bounded batch plans, durable job identity, worker leases, cancellation, and
status. A separately started local worker will execute only startup-allowlisted
operations, using existing BatchStore checkpoints as per-item authority. MCP
EOF will not start or stop that worker.

Trace export and model installation will remain capability-owned services
behind exact Tool IDs and startup-selected roots. Restricted trace/media export
and model installation require an administrative profile. Model installation
also requires an exact plan digest allowlisted at startup and explicit network
authorization. Tool arguments cannot supply a new path root, source host,
credential, process command, or authorization.

## Consequences

- Job status survives MCP disconnection and host restart; a stopped worker
  leaves a visible queued or interrupted job.
- The registry does not duplicate item results, trace payloads, or model bytes.
- Operators must explicitly configure and start a worker and must restart with
  a reviewed allowlist before installing a new model plan.
- Recovery, source-change, lease-expiry, and partial-publication tests become
  mandatory acceptance gates.

## Alternatives considered

- Execute a long batch in one MCP call: rejected because connection lifetime
  does not provide a durable job identity.
- Expose existing CLI parsers as Tools: rejected because arbitrary paths and
  flags can widen authority.

## References

- [RFC-0016: Bounded operator jobs, trace access, and model installation](../rfcs/0016-operator-jobs-traces-and-models.md)
- [ADR-0012: Establish Agent trace recording and export boundaries](0012-agent-trace-recording-export-boundaries.md)

## Amendment 1 (2026-10-09)

[RFC-0016 amendment 1](../rfcs/0016-operator-jobs-traces-and-models.md#amendment-1-workspace-host-configuration-2026-10-09)
places the operator capabilities in the version-2 workspace host
([ADR-0023](0023-versioned-workspace-host-authority.md)):

- `capabilities.batch`, `capabilities.trace`, and `capabilities.model` are closed
  objects; read, write, and admin profile ceilings gain the RFC permissions.
- Batch video operations reuse `capabilities.video` pinned executables and
  descriptor-bound staging ([ADR-0026](0026-host-media-handoff.md)); split outputs
  go to per-item directories opened through the selected output root.
- The worker routes jobs only through a create-only operation descriptor whose
  recomputed digest must match the job's fingerprint, so configuration edits can
  strand a job but can never re-route it to different authority.
- Explicit resume of an interrupted job is an operator command, not an MCP Tool.
