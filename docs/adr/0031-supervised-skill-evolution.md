# ADR-0031: Versioned supervised Skill evolution

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- Authority: Explicit local implementation request; independent review remains pending.

## Decision

Keep the deterministic promotion primitive compatible. Implement evidence extraction,
proposal generation, evaluation and supervised activation as separate services behind
public ports. The local registry owns SQLite transactions and immutable content;
raw traces remain owned by capicove.traces. Contracts remain dependency-light.
Use compare-and-swap on the active pointer so stale approvals fail. Store review and
pointer changes in the same transaction. Verify content hashes on reads and activation.
Runtime skill use is explicit and recorded; unapproved candidates never enter prompts.
Use Python and SQLite in process; no broker or external service is required.

Before executing a valid evaluation campaign, append an incomplete, non-passing
report to revoke prior approval readiness. Completion appends a new immutable
report only if its start report is still latest, checked inside the same SQLite
write transaction. A superseding attempt owns readiness; older success or failure
cannot replace its outcome. Interrupted execution records completed observations
and an infrastructure-error observation when storage remains available, then
propagates the exception. A hard exit leaves the incomplete report non-passing.

## Consequences

The prior automatic-canary state vocabulary remains compatible but does not authorize
automatic production activation. Evaluation fixtures establish only the behavior they
exercise. Model-backed synthesis and replay require explicit configured providers.
The host retains ownership of cancellation, execution permissions and model selection.
