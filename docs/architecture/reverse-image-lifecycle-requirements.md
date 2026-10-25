# Reverse image lifecycle requirements

These requirements preserve the proposed Stage 2 design boundary from the
development plan. They are not implementation approval. The
[retrieval RFC](../rfcs/0002-multimodal-retrieval-contracts.md) is authoritative;
ADR-0020 remains Proposed.

## Prepare the Intent and Generation Contract Revision for Review

The offline personal profile requires review of the complete intent, generation,
authorization and lifecycle contracts before activation. It must not imply
enterprise capacity qualification or activate an unfinished research adapter.

## Lease and schema concurrency

A future `def renew(` operation must compare the original `expected_expires_at`
and the `owner_session_epoch`, along with the owner `application_id`, in one
transaction. A stale owner cannot extend a replacement session's lease.

The store checks `PRAGMA user_version` before reading or changing persisted
state. A newer schema version fails closed. Every migration is transactional;
rollback must leave the previous valid schema and ownership data usable.

## Source authorization

`InvocationAuthorizationContext` contains server-owned `allowed_source_root_refs`;
transport input cannot grant or widen source access. `open_authorized` returns an
`AuthorizedImageHandle` through descriptor-relative, no-follow traversal. Decode,
hash and verification must operate on the same opened object, not reopen a path
after authorization. These requirements preserve the source-race boundary even
when the original source path is replaced during a request.
