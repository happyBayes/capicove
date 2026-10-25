# Releasing Capicove

All implemented capabilities ship through one core distribution and five optional
plugins. This procedure qualifies and publishes that scope; it is not a feature
roadmap. The public repository is `happyBayes/capicove`; source publication uses
reviewed snapshots with independent history where development is internal. The first prepared
version is `0.1.0`. Preparing artifacts does not make them available on PyPI.

## Local release gate

Use a reviewed checkout and Python 3.11 or later. Preserve local user changes.
Install the development, documentation and release tools, then run:

```bash
uv sync --group dev --group docs --group release
make quality
make test
make test-qualification
make docs-check
make build
uv run --group release python scripts/release_packages.py \
  --version 0.1.0 --build --output /tmp/capicove-0.1.0-release
uv run --group release twine check --strict /tmp/capicove-0.1.0-release/*.whl \
  /tmp/capicove-0.1.0-release/*.tar.gz
```

The output directory must be new and empty. The package inspector requires one
wheel and one source archive for each of these distributions:

- `capicove`
- `capicove-execution-spark`
- `capicove-processor-paddleocr`
- `capicove-retriever-elasticsearch`
- `capicove-retriever-opensearch`
- `capicove-retriever-faiss`

It verifies package names, versions, MIT license files, dependency metadata and a
restricted payload inventory. It rejects path traversal, links, unexpected files,
model weights and oversized members, and writes `SHA256SUMS` and
`release-manifest.json`. Do not add datasets, credentials, model weights or local
qualification output to the source distribution or Git history.

Test the actual wheels outside the checkout with fresh environments and caches:

```bash
for profile in base interfaces generation knowledge spark paddle elasticsearch opensearch faiss; do
  uv run --group security python scripts/release_smoke.py \
    --dist /tmp/capicove-0.1.0-release --profile "$profile" --audit \
    --report "/tmp/capicove-install-$profile.json"
done
```

These tests check ordinary dependency resolution, installed imports, CLI execution,
local image processing, Wiki ingestion/retrieval, source indexing, MCP catalog,
plugin entry points, native FAISS search and Torch-free chat-template rendering
as applicable. They do not prove that
external services or model-dependent execution are provisioned. Record those
separate results using [local model qualification](local-model-qualification.md)
and each plugin's native integration tests. OpenSearch tests can use an explicitly
provisioned 3.6.0 service at `CAPICOVE_TEST_OPENSEARCH_URL=http://127.0.0.1:PORT`;
otherwise they start the pinned Docker image. The native endpoint must be a
disposable test service because the tests create and remove their own indexes. The `knowledge` profile installs the CPU runtime dependencies and tests the native
knowledge CLI lifecycle without downloading model weights. Real model qualification
is separate. Never count skipped tests as passed
runtime qualification. Verify the documented Java, FFmpeg, PaddlePaddle, GPU and
local-model requirements for each advertised deployment platform.

The root Security workflow audits all locked extras and groups without advisory
exclusions. Clean-install audits cover each plugin's independently resolved runtime
dependencies. Run the secret detector over tracked and newly added files, inspect
Git history before public visibility, and review third-party licenses separately.
A clean source archive does not certify that the repository's history is public-safe.

## Version and migration checks

The core version lives in `src/capicove/__about__.py`; each plugin declares its
matching version and compatible core requirement. Update plugin metadata and both
lock files together. The inspector rejects mismatched versions and direct URL
runtime dependencies. Use a changelog entry with the tested compatibility matrix.

For 0.1.0, BGE requires sentence-transformers 5.6.x. Its preprocessing identities
end in `st5.6-v2`. Rebuild indexes created with `st5.2-v1`; old and new vectors must
not share an embedding space. MCP requires SDK 1.30 or later within the 1.x line.

## Maintainer account configuration

These are external account actions, not settings that a local source change can
activate. Verify them before authorizing publication:

1. Confirm ownership of all six PyPI project names. Configure a pending or existing
   Trusted Publisher for each project: the exact new public repository owner/name,
   workflow filename `release.yml`, environment `pypi`. Do not create API tokens.
2. Set the repository variable `CAPICOVE_PUBLISH_REPOSITORY` to the exact new
   public `happyBayes/capicove`. Leave it unset in internal development repositories;
   publication fails closed until the explicit destination matches. Update package
   URLs and documentation links to that public destination before its first commit.
   Create the GitHub `pypi` environment with required reviewers, prevent self-review
   where available, and restrict deployments to reviewed `v*` tags. Protect `main`
   and release tags against unreviewed modification. OIDC trust must match the exact
   environment and workflow; local YAML cannot enforce account protection settings.
3. Enable private vulnerability reporting and verify the link in `SECURITY.md` with
   an authorized account. Confirm the public repository name and complete source,
   history, rights and license reviews before changing repository visibility.
4. Connect Read the Docs to the new public repository using `.readthedocs.yaml`. Verify a
   strict documentation build and the versioned documentation URL before announcing
   the release. Do not describe an unverified remote site as deployed.

## Publish reviewed artifacts

The **Release** workflow is manual. First run it with `publish=false`; it reuses
all CI and Security gates, builds the twelve artifacts once, then installs and
audits them on Python 3.11–3.14 (interfaces) and Python 3.11 (plugin profiles).
Review this evidence and the separately provisioned native/model qualification.
A green installation job alone is insufficient for an all-runtime claim.

After maintainer authorization, create the reviewed `v0.1.0` tag and dispatch the
same workflow from that tag with version `0.1.0` and `publish=true`. Publication
requires the protected `pypi` environment. Only that job receives OIDC permission;
it downloads the checked artifacts, verifies their hashes, and publishes the core
before the five plugins. It does not rebuild distributions or use a stored token.
If one project upload fails, inspect PyPI's exact existing filenames and hashes
before retrying; PyPI versions cannot be overwritten. Do not silently change or
rebuild an already published version.

Read back all six PyPI versions and artifact hashes, test installations from the
public index, then create the GitHub release with the changelog, checksums and
artifact manifest. Repository visibility changes, tag pushes, uploads and release
announcements require separate explicit authorization. This workflow does not
perform those account setup or visibility actions for you.

## Separate development and public history

Keep internal development records in their original repository. Export a reviewed
file snapshot and validate the extracted source before creating independent public
history. Do not push internal branches or tags to publish a snapshot. Follow
[public source snapshots](public-source.md) for the allowlist, draft inspection,
export commands and account configuration.
