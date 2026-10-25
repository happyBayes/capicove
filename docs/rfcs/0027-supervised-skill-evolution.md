# RFC-0027: Supervised Skill evolution

- Status: Accepted for operator-requested implementation
- Date: 2026-10-20
- Authorization: The operator explicitly requested implementation of the previously
  reviewed evidence-to-candidate-to-evaluation-to-approval-to-runtime-feedback loop.
  This authorizes local implementation, not independent review, release, or upload.
- Decision: [ADR-0031](../adr/0031-supervised-skill-evolution.md).

## Scope

Implement the supervised workflow described in the August Skill evolution proposal.
Preserve existing contracts and raw Episode records. New versioned local records
hold evidence identities, distilled experience, immutable candidate content,
paired evaluation results, explicit reviews, active pointers and usage events.
Candidate creation never changes active behavior. Approval requires matching expected
state and active baseline, verified content, passing evaluation and an identified
local operator with a reason. Rollback selects a retained compatible version.

All source evidence must come from validated completed trace snapshots. Restricted
records, incomplete chains, unresolved versions and weak evidence fail closed.
Synthesis consumes bounded sanitized examples, never unrestricted store access.
Task failure is attributed before modifying instructions; misuse, infrastructure
failure and uncertainty default to no change. New content cannot expand permissions.

Evaluation starts by persisting a non-passing report before invoking an executor.
Final reports use an atomic expected-latest-report check to prevent an older
concurrent attempt from restoring readiness after a newer attempt starts or fails.
Exceptions retain completed observations without storing raw provider diagnostics;
approval remains unavailable until a current attempt completes successfully.

Evaluation separates synthesis, replay and held-out episodes. Paired baseline and
candidate execution uses a configured evaluator, never provider-reported scores as
proof. A deterministic local adapter supports reproducible behavioral scenarios;
results identify their evaluator and scope rather than claim general model quality.
The administration CLI owns approval; no Agent-callable approval Tool is added.

Runtime loading selects only verified active versions explicitly requested by the
host. Invocation records the actual loaded version and appends observable outcomes
and feedback, making subsequent distillation possible. Optional background review
is bounded, single-flight and cancellable; it creates candidates, never approves.
No network, model downloads, generated-code execution, cross-user sharing, telemetry,
or mutation of released Skill bytes is enabled implicitly.
