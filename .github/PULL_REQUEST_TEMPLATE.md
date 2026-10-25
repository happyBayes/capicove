## Summary

Describe the problem and the outcome of this pull request. Link the issue or RFC.

- Closes:
- RFC:

## Target branch

- [ ] This normal feature, fix, documentation, or maintenance PR targets `dev`.
- [ ] This is a release PR from `dev` (or `release/<version>`) to `main`.
- [ ] This is an urgent hotfix PR from `hotfix/<issue>-<description>` to `main`;
      its back-merge or cherry-pick PR to `dev` is linked below.

Back-merge / follow-up PR:

## Change type

- [ ] Feature
- [ ] Fix
- [ ] Documentation
- [ ] Refactor with no behavior change
- [ ] Build, dependency, or CI maintenance

## Validation

List the exact commands and results used to validate this change.

```text
make quality
make test
make build
```

## Impact assessment

- [ ] Tests cover new or changed behavior.
- [ ] User and contributor documentation is updated.
- [ ] No canonical contract, serialized schema, persistent format, plugin entry
      point, CLI, API, or MCP compatibility is changed.
- [ ] No new runtime dependency or external service is introduced.
- [ ] No network, telemetry, credential, privacy, consent, or data-retention
      behavior is introduced or changed.
- [ ] No source evidence, model weights, generated indexes, credentials, or private
      user data is included.

If any compatibility, dependency, or security statement is unchecked, explain the
impact, migration path, and related RFC/ADR:

## Tool, trace, and evolution impact

- [ ] This change does not alter a Tool Contract or CLI/MCP semantic behavior, or
      equivalent transport contract tests are included.
- [ ] This change does not alter trace schemas, redaction, rewards, feedback, or
      training-export behavior, or its lineage and migration impact is documented.
- [ ] This change does not alter RSI synthesis, evaluation, promotion, canary, or
      rollback behavior, or Mode A and Mode B tests cover the change.
- [ ] This change does not expand filesystem, network, credential, data, subprocess,
      privacy, security, or audit permissions.
- [ ] CLI and MCP parity evidence is included when either enterprise Tool surface
      changes.

Explain every unchecked statement and link the required RFC, ADR, evaluation, or
migration evidence:

## Reviewer notes

Call out the highest-risk files, decisions, and assumptions. Include screenshots or
benchmark results only when they materially help review.
