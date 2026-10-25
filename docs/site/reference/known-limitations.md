# Known limitations

Capicove's preview features require capability-specific qualification. A source
export or successful package build does not establish production readiness.

## Retrieval indexing

The reviewed preview has a reported identity collision when identical content is
imported under different paths. It affects code and unified multimodal generation
builds. Until a regression-tested fix is included in a release, avoid duplicate
source content and do not rely on indexing arbitrary repositories successfully.

Code relation expansion remains opt-in. Real-media retrieval quality, installed
model journeys and platform support require separate measurements with the
selected assets and runtime.

## Remote control

Remote MCP is a controlled preview. Reported defects concern storage accounting
for conflicting upload chunks, consumed approvals after backup restoration, and
artifact commits racing with workspace revocation. Keep remote artifact and
approval/control-plane features disabled until their regression fixes are verified.
Read-only service exposure still requires verified identity-provider integration,
TLS configuration, resource policies and independent security qualification.

## Wiki maintenance

Wiki v3 maintenance is experimental. Back up a workspace before migration; older
versions cannot read migrated workspaces. Agent-generated semantics, human review,
cross-platform recovery and production capacity require separate acceptance.

Update these restrictions only when the release includes the relevant fixes and
their verification evidence. Report security concerns through the repository's
private vulnerability reporting route, not a public issue.
