# RFC-0012: Book Structure and Agent Task Contracts

- Status: Accepted
- Date: 2026-09-02
- Owners: Capicove maintainers

## Summary

Capicove turns immutable document evidence into a deterministic BookStructure and
bounded, model-neutral Agent tasks. It never chooses or invokes a language model.
High-confidence structure passes directly; only low-confidence local subtrees
require review.

## Contracts

BookStructure contains ordered BookSection records with stable IDs, parent and
level, exact and normalized titles, half-open PDF page ranges, heading block IDs,
construction method, confidence, reason codes, and warnings. Section ranges are
monotonic and stay inside parsed pages. A heading always resolves to source
evidence.

SummaryTask contains a stable task ID, section ID, objective, bounded evidence,
allowed block citation IDs, dependency task IDs, output contract version, and an
idempotency key. Tasks form an acyclic bottom-up graph. Paths, credentials,
unrelated sections, and full books are excluded.

SummarySubmission contains an abstract, key points, and atomic claims. Each factual
claim selects at least one allowed citation block. The workspace rejects unknown,
stale, dependency-incomplete, or uncited submissions. Resubmission appends a new
version linked to its predecessor and never mutates evidence or an earlier
submission.

## Compatibility and privacy

Contract version 1 is provider-neutral and JSON-compatible. Caller-supplied model
and provider fields are optional provenance only. PDF bytes, parsed text, tasks,
and submissions are confidential by default and remain in the authorized local
workspace unless the caller explicitly sends a bounded task to an Agent.

## Failure semantics

Unresolved structure blocks strict publication. Invalid submissions return stable
validation errors without recording partial state. A failed page, missing block,
stale task, incomplete dependency, or citation outside the task allowance cannot
be hidden by a generated summary.
