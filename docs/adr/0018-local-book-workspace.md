# 0018: Use an immutable local workspace for book workflows

- Status: Accepted
- Date: 2026-09-02
- Owners: Capicove maintainers

## Context

PDF parsing, Agent task exchange, repeated submissions, and Wiki publication span
multiple processes and may be interrupted. The workflow needs resumability without
a database dependency or an inference-provider dependency.

## Decision

Each authorized workspace stores content-addressed PDF bytes, one immutable parser
representation and receipt, one deterministic structure, immutable tasks,
append-only submission versions, and a generated Wiki projection. JSON and JSONL
records use closed codecs and atomic replacements. User paths are resolved before
entry and never appear in Agent tasks.

The caller's Agent reads tasks and submits structured results through CLI or MCP.
Capicove validates dependencies and citations before appending a submission.
Publication reads only accepted workspace records and cannot invoke a model.

## Consequences

- Work survives process restarts and can be inspected without a service.
- Content hashes and append-only versions preserve provenance and replayability.
- One local process at a time may mutate a book workspace in version 1.
- Remote stores, concurrent writers, distributed queues, and automatic model calls
  require separate change control.

## Alternatives considered

An embedded database was rejected for the first delivery because immutable files
and append-only JSONL satisfy the bounded workflow. In-memory orchestration was
rejected because it loses resumability and review evidence.

## References

- RFC-0012
- RFC-0013
- ADR-0017
