# 0027: Manage Local Models Through Typed Factory Leases

- Status: Accepted
- Date: 2026-10-13
- Owners: Capicove maintainers

## Context

Model provisioning lives in `model_management`, while local BGE, SigLIP2 and
Paddle composition is scattered across capabilities. Operator installation
discovery is coupled to document parsers. A bounded CPU device needs explicit
profile selection and release rather than eager loading of all installed models.

## Decision

Implement the operator-authorized [RFC-0023](../rfcs/0023-unified-model-factory.md)
scope: `capicove.model`, compatibility aliases, immutable offline profiles,
explicit provider registration, typed context-managed factory leases and generic
installer discovery. Retain existing public capability ports and format-specific
asset validators. Installation authority stays separate from inference selection.
The default live instance bound is one; only idle models can be evicted.

This decision records requested implementation scope. It does not record an
independent maintainer review, merge approval or release authorization.

## Consequences

The operator's 2026-10-13 legacy-removal amendment supersedes the initial
compatibility-alias decision above. `capicove.model_management` is removed before
commit; external callers must update their imports to `capicove.model`. Existing
model assets and persisted evidence keep their formats and identities.

Existing receipts and indexes do not migrate. Public adapters preserve model and
preprocessing identities. Optional libraries remain lazy. SDK memory reclamation
is runtime dependent, and typed references must remain inside leases. Canonical
contracts never depend on the factory. Third-party plugin code is trusted installed
code selected by the operator, not a configuration-supplied import expression.

## Alternatives considered

- A rename alone leaves composition and lifetime unmanaged.
- A universal `run()` interface loses embedding, OCR and parser contract checks.
- A global singleton or automatic preload couples workspaces and wastes memory.
- Broadening one asset format allowlist weakens existing security guarantees.

## References

- [RFC-0023](../rfcs/0023-unified-model-factory.md)
- [ADR-0022](0022-durable-operator-jobs-and-admin-isolation.md)
