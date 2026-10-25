# Personal reverse-image search

Capicove's Stage 2 reverse-image search is an offline, single-node preview for a
personal collection of at most 10,000 independent images. The caller declares the
semantic intent as `instance`, `concept`, or `balanced`; Capicove does not run a
server-side intent classifier.

## Install and provision

Install the CPU runtime and MCP transport only where you intend to run search:

```bash
python -m pip install "capicove[retrieval-cpu,mcp]"
```

You must separately provision two verified, already-local ONNX image-encoder
bundles and the images to index. Capicove does not download weights, ingest remote
URLs, copy user media into the package, or enable network access or telemetry.

### Offline two-bundle provisioning

Preview startup uses one content-addressed provisioning manifest below an
explicit absolute model root. The manifest contains exactly two ordered entries:
`instance`, then `concept`. Each entry fixes one model-lock key, installation
directory, Capicove receipt, and bundle-manifest digest. Every recorded path is a
safe relative POSIX path; absolute paths, `..`, URL/file schemes, symlinks, missing
files, duplicate lanes, and alternate receipt names fail closed.

The closed schema is:

```json
{
  "entries": [
    {
      "bundle_manifest_sha256": "<sha256 of bundles/instance/manifest.json>",
      "installation_relative_path": "bundles/instance",
      "lane": "instance",
      "model_key": "<exact key in the model lock>",
      "receipt_relative_path": "bundles/instance/capicove-model-receipt.json",
      "receipt_sha256": "<sha256 of the exact receipt bytes>"
    },
    {
      "bundle_manifest_sha256": "<sha256 of bundles/concept/manifest.json>",
      "installation_relative_path": "bundles/concept",
      "lane": "concept",
      "model_key": "<exact key in the model lock>",
      "receipt_relative_path": "bundles/concept/capicove-model-receipt.json",
      "receipt_sha256": "<sha256 of the exact receipt bytes>"
    }
  ],
  "model_lock_relative_path": "locks/reverse-image-models.toml",
  "model_lock_sha256": "<sha256 of the exact TOML lock bytes>",
  "provisioning_id": "personal-reverse-image-v1",
  "schema_version": 1
}
```

The angle-bracket values above are explanatory placeholders. Encode the real
document with the exact Python JSON codec `sort_keys=True`,
`separators=(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`; append one
newline and then encode as UTF-8. Therefore non-ASCII identifiers use JSON
`\uXXXX` escapes in the stored bytes. Startup policy separately pins the SHA-256
of those exact bytes, avoiding a self-hash field.

The referenced TOML is a complete image-model lock accepted by
`load_manifest`. Each installed receipt must have been created for that exact lock
digest and exact normalized model entry. Provisioning then performs four checks in
order: provisioning-manifest digest, model-lock digest and parse, receipt digest
and lock match, and verified bundle-manifest/assets. Both complete embedding spaces
must also be distinct. A digest never selects a compatible fallback model.

Pass the pinned local declaration to the runtime composition root:

```python
from pathlib import Path, PurePosixPath

from capicove.runtime import build_reverse_image_runtime
from capicove.runtime.reverse_image import ReverseImageModelProvisioning

model_provisioning = ReverseImageModelProvisioning(
    root=Path("/srv/capicove-models"),
    manifest_relative_path=PurePosixPath("reverse-image-provisioning.json"),
    manifest_sha256="<64 lowercase hexadecimal characters>",
)

runtime = build_reverse_image_runtime(
    # Existing workspace, storage, authorization, and CPU-budget arguments remain required.
    model_provisioning=model_provisioning,
    # ...
)
```

Provisioning verification is standard-library-only and opens no network or model
runtime. It completes before generation or lifecycle state is created. The exact
receipt digest from provisioning remains pinned in each verified installation;
the build path re-verifies both bundle receipts and artifacts before staging or
decoding the first image, so tampering after startup also fails before media
reads.

Content-addressed provisioning currently requires POSIX directory-descriptor
support: `dir_fd`, `O_DIRECTORY`, `O_NOFOLLOW`, `O_CLOEXEC`, and `O_NONBLOCK`.
Every component of the absolute model root must be a real directory rather than a
symlink. On platforms without those primitives (including current Windows Python
builds), provisioning raises a controlled error with remediation guidance. The
pre-existing direct `instance_bundle` and `concept_bundle` composition path remains
available for already-verified bundles until an equally strong Windows handle
implementation is provided. That compatibility path has no provisioning-manifest
identity and therefore cannot supply Task 20 provisioning-bound scale evidence.
It must not synthesize such an identity from bundle or receipt digests.

Capicove intentionally does not register a candidate model as a packaged default.
An administrator must select, license, provision, and qualify exact instance and
concept bundles; model weights, local receipts, and the provisioning manifest are
deployment artifacts and must not be committed.

## Administrative lifecycle

Search is unavailable until an administrator completes each explicit step:

1. Build derived pHash and semantic representations from authorized local images.
2. Validate the catalog, indexes, parameters, and target-representation manifest.
3. Publish one immutable generation without activating it.
4. Issue a signed local preview qualification for that exact publication.
5. Register and explicitly activate the qualified generation using compare-and-swap.

Build, validation, publication, qualification, and activation are deliberately
separate. A build or publication never changes the active alias. Generated indexes,
model weights, source media, and qualification keys are runtime artifacts and must
not be committed.

## Query surfaces and limits

The implemented Stage 2 surfaces are the Python retrieval API, Tool
`capicove.search.image`, and MCP tool `capicove_search_image`. The executable
[CLI and MCP host](../../cli/image-search.md) exposes `capicove image search`
and `capicove image serve` for already provisioned and activated indexes. The existing
text Tool `capicove.search` and MCP `capicove_search` remain unchanged. Earlier
draft image clients must migrate to the image-specific names and import their
handlers from `capicove.tools.image_search` or `capicove.mcp.image_search`.

Every call receives host-created authorization that fixes the workspace,
permission, data policy, and exact allowed source roots. Request fields can narrow
that authority but cannot add a root, mount, provider, model, host, URL, credential,
or index generation.

Each response page contains at most 100 hits. A signed cursor can continue through
the active personal profile's bounded candidate depth. The personal profile reads
the complete filtered candidate depth before weighted rank fusion, so pagination
does not change the first-page ranking. A cursor becomes stale if the pinned
generation, lane outcomes, or bounded ranking changes. Every successful result is
visibly marked with the `retrieval_preview` warning because semantic no-answer
calibration and real-model quality gates are not yet formal.

## Current non-goals

- video, keyframe, and video-segment search;
- enterprise or cross-node indexes;
- automatic media/model acquisition;
- live mutation of an active generation;
- a calibrated accept-or-abstain threshold for non-empty semantic retrieval; and
- formal quality or performance claims for a real 10,000-image corpus.
