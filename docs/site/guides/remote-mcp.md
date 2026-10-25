# Remote MCP endpoint (controlled preview)

Expose selected **read-only** workspace Tools to authenticated external MCP
applications over Streamable HTTP. Every request carries an OAuth/OIDC access token
that is verified against one enrolled issuer. Sessions are bound to the complete
authority (issuer, tenant, application, subject, workspace grant and audience), and
each call intersects deployment, application, token, user and workspace permissions.

!!! warning "Preview boundary"
    Only explicitly composed read Tools are served, plus the optional artifact and
    operation-control Tools below; workspace mutations, durable job submission and
    administrative operations are not exposed. No production security sign-off,
    multi-gateway coordination exists yet, and no served Tool runs in an isolated worker. Interoperability
    with real identity providers has not been qualified. The repository acceptance
    restrictions are listed in [known limitations](../reference/known-limitations.md).

## Files

All three documents are private owner-only files (mode `0600`) in directories no
other identity can write. They are opened with no-follow descriptors.

- **Endpoint policy** (`schema_version: 1`): public URL ending in `/mcp`, the enrolled
  issuer and applications, deployment permissions and one explicit grant per
  principal (`client_id`, `subject_kind`, `subject`, `workspace_binding_id`,
  `grant_id`, `epoch`, and application/user/workspace permissions).
- **Workspace host**: an ordinary version-2 workspace host file whose enabled read
  Tools are served. Its `workspace_id` must equal every grant's
  `workspace_binding_id`, and its `environment_id` must equal the policy's.
- **Launch configuration**:

```json
{
  "schema_version": 1,
  "policy_path": "/etc/capicove/remote-policy.json",
  "workspace_config": "/srv/capicove/host.json",
  "epoch_store": "/srv/capicove/remote/epochs.sqlite3",
  "listen": {"host": "127.0.0.1", "port": 8443},
  "verifier": {"kind": "jwks", "url": "https://issuer.example/jwks", "subject_kind": "human"},
  "json_response": false,
  "publish_resource_metadata": true
}
```

`verifier.kind` may instead be `introspection` with `client_id` and an absolute
private `secret_file`; secrets are never accepted inline. The optional
`control_store` (an absolute, owner-only directory) enables the artifact and
operation Tools described below; omit it to keep the endpoint read-only. The optional
`revocation_journal` (an absolute path outside the backed-up stores) records every
workspace revocation; on start the epoch store reapplies it, so restoring an older
backup cannot revive a revoked grant. The listener must be
loopback (`127.0.0.1` or `::1`). Terminate TLS in a reverse proxy that forwards the
public `Host` unchanged and does not add `Forwarded`/`X-Forwarded-*` headers.

## Check, list and serve

```bash
capicove mcp remote-check --config /etc/capicove/remote-launch.json
capicove mcp remote-catalog --config /etc/capicove/remote-launch.json
capicove mcp serve-http --config /etc/capicove/remote-launch.json
```

`remote-check` validates the launch, policy, workspace and grant binding and builds
the catalog without opening a socket or contacting the identity provider. It prints
digests, Tool names and schema digests, the grant count and the verifier kind. It
never prints client registrations, subjects, roots or secrets. `serve-http` repeats
the check, then serves on the loopback listener.

The endpoint always offers `capicove_remote_readiness` (permission
`remote:readiness`), which reports the same sanitized readiness without invoking
models. When `publish_resource_metadata` is true it also serves RFC 9728 protected
resource metadata at `/.well-known/oauth-protected-resource/mcp`, and 401 challenges
point to it.

## Artifact and operation Tools (optional)

With `control_store` set, each grant additionally sees the Tools its permissions
allow, bound to that grant's complete authority (issuer, tenant, application,
subject, workspace, grant and audience):

| Tool | Permission | Purpose |
| --- | --- | --- |
| `capicove_artifact_begin_upload` | `artifact:write` | Reserve quota for `size`/`sha256`/`media_type`; the same `idempotency_key` returns the same upload |
| `capicove_artifact_write_chunk` | `artifact:write` | Numbered base64 chunks of 262,144 decoded bytes (the last may be shorter) |
| `capicove_artifact_upload_status` | `artifact:write` | Received chunk indices and digests |
| `capicove_artifact_abort_upload` | `artifact:write` | Release an incomplete upload |
| `capicove_artifact_commit_upload` | `artifact:write` | Verify size, hash and declared media signature, then publish |
| `capicove_artifact_stat`, `capicove_artifact_read_chunk` | `artifact:read` | Reference and bounded base64 reads |
| `capicove_artifact_delete` | `artifact:delete` | Tombstone and release quota |
| `capicove_operation_status`, `_result`, `_cancel` | `operation:submit` | Inspect or cancel owned operations |

Other subjects receive `not_found` for every foreign ID. Each store call rechecks
the grant's revocation epoch. Defaults: 1 GiB per artifact, 10 GiB per
application, 100 GiB per tenant, uploads expire after 24 hours. No operation
submission Tool is offered yet: no remote mutation targets are registered.

## Limits

Each process enforces a 2 MiB request and response bound, 128 concurrent ingress
requests, 32 per application, two streams per session, 1,000 sessions in total and
64 per application. Sessions expire after 15 minutes idle or one hour total; expired
sessions return 404 and release their capacity immediately. These are per-process
limits, not cross-gateway quotas.
