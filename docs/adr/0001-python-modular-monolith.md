# 0001: Start as a Python modular monolith

- Status: Accepted
- Date: 2026-08-08
- Owners: `@TransActionCode`

## Context

Capicove spans evidence storage, multimodal processing, retrieval, knowledge, Wiki,
agent memory, and export. These capabilities need explicit boundaries, but the
project begins with one primary engineer and must remain easy to install locally.
Premature services would add deployment, networking, consistency, and operational
cost before workload boundaries are validated.

## Decision

Capicove will start as a Python-first modular monolith using a `src` package layout.
Capabilities own their contracts and application operations. Infrastructure is
connected through ports and adapters. CLI, HTTP, and MCP interfaces share the same
application operations.

The default runtime remains local and in-process. Boundaries must permit later
separation of stateless interfaces, workers, and stores without changing canonical
models.

## Consequences

- Local installation and debugging remain simple.
- Cross-capability changes can be delivered atomically.
- Boundary checks and review discipline are required to prevent a coupled monolith.
- Distributed reliability is designed into ports and idempotency rules but is not
  implemented before a measured need.

## Alternatives considered

- Microservices were rejected for the initial phase because their operational cost
  would precede stable capability boundaries.
- A monorepo of separately published internal packages was rejected because it
  creates version coordination without current user value.
