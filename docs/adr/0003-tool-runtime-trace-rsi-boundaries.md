# 0003: Unify Tool Runtime, Trace, and controlled RSI boundaries

- Status: Accepted
- Date: 2026-08-09
- Owners: `@TransActionCode`

## Context

Capicove must expose multimodal processing and search capabilities to enterprise
services and AI agents through CLI and MCP. Maintaining separate business logic for
each transport would cause their schemas, errors, permissions, and trace behavior to
diverge.

Capicove must also learn from tool-use history. Repeated successful traces should
produce reusable SOP, Workflow, or Skill candidates, while repeated failures and
corrections should produce Pitfall and recovery candidates. Recursive updates can
amplify errors or expand privileges unless evidence, evaluation, activation, and
rollback are explicit boundaries.

## Decision

Capicove will use one versioned, transport-neutral Tool Contract for CLI, MCP, and
future interfaces. The Tool Runtime resolves a ToolSpec, enforces invocation policy,
calls a capability operation, returns a canonical result envelope, and appends an
immutable trace record.

Agentic RL traces use immutable Episode, Step, and ToolCall records. An Episode
captures its Tool Catalog, environment, and active evolution-asset versions. Trace
schemas store observable inputs, actions, results, evidence, outcomes, and feedback;
they do not require hidden model reasoning.

Recursive self-improvement creates immutable, trace-derived candidates. Mode A is
the default supervised path. Mode B is an explicitly enabled path through sandbox,
shadow, and canary states with automatic quarantine or rollback. Mode B cannot
expand permissions or alter Tool Contracts, persistent formats, dependencies,
privacy, security, or audit boundaries.

The first implementation remains an in-process Python modular monolith. Canonical
contracts use only the standard library, and separation-ready ports allow interface,
execution, trace, and evolution workers to be deployed independently when required.

## Consequences

- CLI and MCP share stable schemas, errors, permissions, and trace semantics.
- Tool-use training exports can retain exact environment and evidence lineage.
- Evolution assets can improve recursively without overwriting source traces or
  previously released versions.
- Automatic promotion requires explicit policy, evaluation, observability, and
  rollback infrastructure.
- Public Tool, Trace, evolution, and persistent schemas require compatibility review.
- The core package retains a small default dependency and deployment footprint.

## Alternatives considered

- Separate CLI and MCP implementations were rejected because behavior and policy
  would drift.
- Immediate Tool Runtime and RSI microservices were rejected because their
  operational cost precedes stable workloads and contracts.
- Fully automatic production updates were rejected because they could silently
  amplify failure patterns or cross trust boundaries.
- A plugin-only core was rejected until Tool, Trace, permission, and evolution
  contracts stabilize.

## References

- Tool Runtime and RSI architecture design (archived development reference)
- [ADR-0001: Python modular monolith](0001-python-modular-monolith.md)
- [ADR-0002: Plugin entry points](0002-plugin-entry-points.md)
