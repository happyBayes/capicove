# ADR-0004: Declare Tool Resource Requirements

- Status: Accepted
- Date: 2026-08-10
- Owners: Capicove maintainers

## Context

Tool execution environments need a machine-readable declaration of filesystem and
subprocess requirements before an invocation is admitted. Permission strings remain
the policy vocabulary, but they do not give a runtime a structured resource shape.
Hiding these requirements in handler metadata would bypass public contract review.

## Decision

`ToolSpec` contains a frozen `ToolResources` value with `filesystem_read` and
`filesystem_write` booleans plus an ordered tuple of approved subprocess executable
names. `required_permissions` remains the policy vocabulary; `resources` is the
machine-readable execution requirement. Network access remains the existing,
separate `network_access` field.

Executable values are bare names such as `ffmpeg` and `ffprobe`, never paths,
arguments, shell fragments, or globs. Contract version 1 gains the field with a safe
empty default because existing ToolSpecs remain behaviorally unchanged.

## Consequences

- Runtimes can reject an invocation before filesystem or subprocess access.
- Tool contracts expose security-relevant resource changes to compatibility review.
- Executable arguments and path policy remain runtime-owned and are not Tool input.
- Existing ToolSpec constructors remain compatible through the empty default.

## Alternatives Considered

- Encode resources only as permission strings: rejected because execution runtimes
  need structured resource declarations.
- Store requirements in handler metadata: rejected because it is not versioned or
  visible to registries and transports.
- Merge network access into the new value: rejected to preserve the existing stable
  `network_access` contract.

## References

- [Module boundaries](../architecture/module-boundaries.md)
- Video split Tool design (archived development reference)
