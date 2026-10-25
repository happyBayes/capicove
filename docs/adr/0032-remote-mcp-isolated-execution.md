# ADR-0032: Remote MCP over shared Tools with isolated application execution

- Status: Accepted for implementation by requester direction on 2026-10-22; not production-qualified
- Date: 2026-10-22
- RFC: [RFC-0028](../rfcs/0028-remote-mcp-service.md)

## Context

Capicove has local stdio MCP and authenticated loopback REST. Startup configuration
owns identity and permissions, and multiple public capabilities lack common-host
bindings. Exposing the REST listener cannot provide authenticated remote MCP or
complete capability coverage.

## Proposed decision

Add an opt-in Linux Streamable HTTP MCP adapter. Delegate identity issuance to a
replaceable OAuth/OIDC provider and validate access tokens as an OAuth resource
server. Bind authenticated application and optional user identity to live policy,
explicit workspace grants, resources and request contexts. Never use one shared
startup actor for all remote requests.

Use the complete AuthorityKey, including delegated user subject, for ownership,
sessions, caches, operation idempotency and worker resource access. Application
isolation alone is insufficient when two users have different resource grants.
Generic operation submission enforces both wrapper and target permissions.
Authorize sources before retrieval, model context and egress, then recheck output.

Reuse canonical Tools, validation, result encoding and public application services.
Keep SDK, web, identity, storage and worker frameworks outside canonical contracts.
Preserve the existing stdio and loopback REST interfaces. Add versioned remote
artifact-reference contracts where local path-based schemas are unsuitable.

Use separate worker security domains, opaque artifact transfer, durable operation
records, idempotent admission, fenced publication and live output authorization.
Administrative MCP has a separate audience and exact-plan approval, preserving
human-supervised Skill activation and existing transfer authorization rules.

The workspace owner serializes policy/source revocation with effect commits and
output release. Revocation acknowledgement means old-epoch permits are fenced;
a prior policy check in another transaction is insufficient. Reserve approval,
idempotency and operation admission together; ambiguous effects never recycle an
approval automatically. Approval binds a distinct authenticated human reviewer,
exact plan, audience, versions, expiry and execution lease. Keep local Tool
schemas intact and assign distinct versioned remote names where I/O or durable
execution semantics change. Tool-based artifact reads are the baseline; MCP
resources are not advertised without their own qualified adapter.

Qualify a single-node Linux profile first. Production completion requires every
supported operation in the frozen capability inventory, including CLI/Python-only
capabilities and optional providers selected for the full release. A transport
preview or a catalog-only test is not completion.

## Consequences

This requires targeted changes to composition, actor checks, generic durable jobs,
artifact I/O and missing capability adapters. It does not require rewriting domain
algorithms or replacing all SQLite stores. Existing single-owner stores retain
their write constraints; cross-application sharing needs a verified storage API.

Stateful sessions require a single owner or tested routing. Multi-node HA, shared
GPU isolation and additional architectures are separately qualified deployment
profiles. Identity outages, stale policies, unavailable providers and ambiguous
mutation outcomes fail closed rather than falling back to broader authority.

## Alternatives

A REST reverse proxy does not speak MCP. A generic stdio bridge adds policy and
lifecycle ambiguity without completing capabilities. A new independent service
implementation would duplicate domain logic and evidence lineage. The selected
adapter approach preserves one implementation with explicit new trust boundaries.

## Review constraints, 2026-10-22

The capability inventory must explicitly include directory discovery and duplicate
move/delete, text/image/video index generation lifecycle, VLM token counting and
bounded structured/tool-call outputs, unified model discovery and all registered
plugin/provider variants. A family-level checkbox cannot substitute for individual
operation/provider evidence. Unsupported advanced modes remain explicit exclusions,
not inferred capability claims.

JWT offline revocation is bounded by accepted token lifetime; bearer credentials
are not proof of possession. Admin review requires recent MFA and fresh validation.
The single-node four-hour restore target can exceed a 99.9% monthly availability
budget; neither a 24-hour soak nor this ADR guarantees that availability. G0-G5
must report their exact test population, failure budget and unqualified features.
This review strengthens the proposal and does not mark it accepted or implemented.

## Acceptance

Acceptance of this ADR must be recorded by a maintainer after reviewing RFC-0028,
its capability matrix and required amendments. This file does not change the
Accepted status or behavior of earlier RFCs/ADRs.

## Phased implementation plan

The detailed R0-R6 implementation plan (archived development reference)
maps all 26 capability families to code scope, dependencies, tests and release
gates. Execution status is recorded in the implementation log; release approval remains separate.

## Implementation authorization, 2026-10-22

The requester instructed execution of the reviewed phased plan, baseline repair,
complete capability/contract reconciliation and adversarial acceptance. This
authorizes implementation of this decision. It does not attest to completed
implementation, independent code-owner review, Linux qualification or release
approval, and does not authorize commits, publication or production deployment.
Capability-specific contracts remain subject to their explicit versioned
amendments and tests before exposure. See the
execution log (internal record; public support status is documented in the capability guide).
