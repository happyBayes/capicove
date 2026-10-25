# Executable reverse-image search

These query-only entry points use the same `capicove.search.image` handler:

```text
capicove image search --config HOST.json --arguments QUERY.json
capicove image serve --config HOST.json
```

The second command serves `capicove_search_image` over **local MCP stdio**.
It does not open an HTTP port or implement remote authentication. One process
represents one trusted local principal; do not share it between untrusted users.
Existing text search (`capicove.search` / `capicove_search`) is unchanged.
Video search is not implemented by these commands.

## Prerequisites

From the repository checkout, install the locked CPU and MCP environment:

```bash
uv sync --frozen --extra retrieval-cpu --extra mcp
uv run --no-sync capicove image search --help
uv run --no-sync capicove image serve --help
```

Use a supported POSIX system and real absolute paths, without symlink components.
Before querying, an administrator must provision the exact two ONNX bundles and
complete the [build, qualification, and activation lifecycle](../guides/image/reverse-search.md).
The workspace, generation directory, lifecycle database, qualification directory,
and qualification key must match that deployment. Publication alone is insufficient.
The command never invents a qualification, downloads models, or activates an index.
The team's private Hugging Face upload is a separate, unfinished delivery step;
this interface does not claim real model availability or semantic accuracy.

## Trusted host configuration

Create an owner-controlled JSON file such as `/srv/capicove/host.json` using the
following complete schema. Paths and the digest are deployment values, not a
ready-made installation. Replace the digest with the SHA-256 of the exact local
provisioning-manifest bytes; see its [schema and verification rules](../guides/image/reverse-search.md#offline-two-bundle-provisioning).

```json
{
  "schema_version": 1,
  "workspace_id": "personal",
  "actor_id": "local-owner",
  "generation_root": "/srv/capicove/generations",
  "lifecycle_database_path": "/srv/capicove/lifecycle.sqlite3",
  "model_root": "/srv/capicove/models",
  "provisioning_manifest": "reverse-image-provisioning.json",
  "provisioning_sha256": "<actual 64-character lowercase SHA-256>",
  "source_roots": {"photos": "/srv/capicove/photos"},
  "qualification_root": "/srv/capicove/qualifications",
  "qualification_key_file": "/srv/capicove/secrets/qualification.key",
  "cursor_key_file": "/srv/capicove/secrets/cursor.key",
  "permissions": ["retrieval:search"],
  "cpu_threads": 2
}
```

The schema rejects unknown or duplicate keys. `cpu_threads` is an integer from
1 to 32; `source_roots` contains 1 to 32 existing directories. An empty directory
can authorize an inline-only deployment without granting access to other files.
All query access uses the `local-only` data policy. Artifact references and external
mounts are not configured by this host and fail closed. A query cannot supply a
workspace, actor, extra permission, model path, or signing key.

Both signing-key files contain **32–64 raw bytes**, not hexadecimal text. They must
be distinct. Reuse the administrator's actual qualification key; replacing it
invalidates existing qualification verification. Keep the cursor key stable across
restarts to preserve cursor signatures. Never paste either key into command-line
arguments, query JSON, documentation, or version control.

```bash
chmod 600 /srv/capicove/host.json
chmod 600 /srv/capicove/secrets/qualification.key /srv/capicove/secrets/cursor.key
```

Startup rejects group/other-writable configuration, group/other-readable keys,
symlink inputs, non-regular JSON/key files, oversized inputs, and missing storage.
Keep configuration and its parent directories accessible only to trusted owners;
the local OS account is the authentication boundary, not a network credential.

## CLI request and response

Save this as `/srv/capicove/query.json`, pointing to an image below an authorized root:

```json
{
  "query": {
    "type": "image",
    "source": {"type": "local_path", "path": "/srv/capicove/photos/query.png"}
  },
  "targets": ["image"],
  "data_policy_ref": "local-only"
}
```

```bash
uv run --no-sync capicove image search \
  --config /srv/capicove/host.json --arguments /srv/capicove/query.json
```

For inline images, replace `source` with an object containing `type: "inline_bytes"`,
`base64` (standard Base64 of the image bytes), and `media_type: "image/png"`.
The canonical request limits still apply, including the 4 MiB inline payload limit.
No URL fetching or implicit artifact resolver is available.

Stdout contains one JSON Tool envelope, with `status` and `result`; successful
results contain `hits` and `warning_codes`. Preview success retains
`retrieval_preview`. Follow any returned signed cursor using the canonical search
request contract; changing or tampering with it is rejected.

| Exit code | Meaning |
| --- | --- |
| 0 | Tool succeeded (a valid empty result is still success) |
| 1 | Canonical Tool rejection or search failure; inspect the JSON envelope |
| 2 | Parser, startup, configuration, or transport failure; sanitized stderr |
| 130 | Keyboard interrupt |

Missing activation is not silently repaired. Check the workspace and administrator
activation record. For startup failure, check local file permissions, absolute
paths, model provisioning digests/receipts, key lengths, storage, and installed
extras locally. Diagnostic output intentionally omits private paths and exception
details. `--trace-store` is explicitly unsupported for these two commands.

## MCP client configuration

Configure a stdio-capable client to launch the installed executable. Adapt the
outer configuration format to your client; the process command and arguments are:

```json
{
  "command": "/absolute/path/to/checkout/.venv/bin/capicove",
  "args": ["image", "serve", "--config", "/srv/capicove/host.json"]
}
```

After MCP initialization, `tools/list` exposes only `capicove_search_image` with its
input and output schemas. Send the same query object as `tools/call.arguments` and
`capicove_search_image` as its name. Stdout is reserved for protocol messages.
Normal EOF shuts down the host; admitted synchronous searches finish under their
deadlines before the runtime closes. At most four calls are admitted concurrently;
cancelling a client wait does not free its worker slot early.

## Reproduce without private models or images

Run the executable integration test from this checkout:

```bash
uv sync --frozen --no-default-groups --group model-test --extra retrieval-cpu --extra mcp
uv run --no-sync pytest --no-cov \
  tests/model/test_reverse_image_onnx_runtime_vertical.py
```

The test authors three-channel synthetic ONNX models and tiny PNG fixtures in a
temporary directory, creates matching locks/receipts/provisioning, builds and
explicitly activates a synthetic preview, then closes the administrator runtime.
It starts the CLI in a new process and a real MCP stdio client/server session,
checks tool discovery and returned hit parity, and exercises query rejection.
No private media, Hugging Face access, model download, or production qualification
is involved. Synthetic report hashes are test-only and must never be reused as
real qualification evidence. This proves executable wiring, not model quality.
