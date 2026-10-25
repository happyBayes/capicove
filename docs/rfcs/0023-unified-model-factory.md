# RFC-0023: Unified Local Model Factory

- Status: Accepted
- Date: 2026-10-13
- Owners: Capicove maintainers
- Authorization: The project operator requested implementation of the proposed
  package migration, typed factory, configuration and lifecycle plan on 2026-10-13.
  This records implementation scope, not a completed non-author review or merge
  approval. Normal maintainer review remains required before merging.
- Related decision: [ADR-0027](../adr/0027-unified-model-factory.md).

## Scope and boundaries

`capicove.model` owns local asset selection, verification, provisioning and model
factory lifecycle. Capability-owned adapters continue to implement public text
embedding, aligned image/text embedding, image OCR and document parser ports.
There is no universal inference request or result. Canonical evidence contracts
remain independent from model libraries and factory configuration.

Model factory descriptors must import optional runtimes only when inference is
used. Factory construction validates configuration and descriptors without model
loading, network access, SDK imports or plugin auto-discovery. Builtins compose
public capability entry points; they never import another capability's private
adapter. Profile acquisition verifies complete local assets with the existing
format-specific validators. Paddle does not expand the safetensors validator's
file allowlist.

## Closed startup configuration version 1

The separately loaded JSON/TOML object has exactly `schema_version`, `profiles`,
`bindings`, and optional `max_instances`. `schema_version` is integer 1.
`max_instances` defaults to 1 and is bounded to 1–16. There are 1–64 profiles and
1–128 bindings. Each profile requires `provider`, `capability`, `root` and
`manifest_sha256`, with optional `device` (default CPU) and `options` (default
empty). Names and binding aliases are bounded identifiers, never Python import
paths. Roots are absolute local paths without linked ancestors or traversal.
Missing assets fail on acquisition. Digests are lowercase SHA-256 selected by
the operator, not trusted simply because a local receipt contains them.

Options are bounded scalar mappings. Each builtin rejects unknown keys and
invalid ranges. Devices are explicit CPU, MPS or CUDA selections; availability
is checked by the selected adapter when used. There is no implicit device or
network fallback. Binding values must name configured profiles; a lease also
checks the expected capability before creating anything.

| Provider | Capability | Options | Trusted digest |
| --- | --- | --- | --- |
| `bge-m3` | `text_embedding` | `max_tokens` 1–8192 (default 512), `batch_size` 1–64 (default 1) | Converted safetensors asset inventory |
| `siglip2` | `image_text_embedding` | None | Verified installation asset inventory |
| `paddleocr` | `image_ocr` | `enable_mkldnn` boolean (default false); CPU only | Locked book-balanced profile manifest |
| `paddle-document` | `document_parser` | Same Paddle policy | Same three-model manifest |

This configuration adds no workspace-host fields or inference Tools. Existing
`capabilities.model` remains installation authority. Standalone Book and video
filter commands accept `--model-config` and `--model-binding`; conflicting local
provider/root/device flags are rejected. Video host mode rejects model factory
flags rather than widening its existing model-filter contract. Graph preparation
accepts an explicitly supplied factory and separate parser/OCR binding aliases.
Retrieval and graph embedding callers can borrow the typed encoder and use its
complete embedding space through existing public services.

## Identity and lifetime

Profile identity hashes provider, capability, local root, trusted assets, device
and options. Names are aliases, not identity. Factory cache keys additionally
bind provider version. Existing adapters retain their evidence and embedding
space identities; factory configuration does not overwrite those identities or
relabel vectors as compatible. New model/preprocessing/dimension/normalization
identities require new indexes; existing strict space checks still apply.

A process-owned `ModelFactory` supplies context-managed typed leases. Instances
are created lazily and reused only for the exact profile/provider identity.
Inference leases for one instance are serialized; waiting leases retain their
reference and cannot be evicted. At capacity only idle instances may be evicted.
If all slots are leased, acquisition fails explicitly. Close rejects active
leases and disposes every idle instance, reporting aggregated cleanup failures.
Borrowed references must not escape a lease. The slot bound is not an RSS quota;
SDK allocators may retain memory. There is no global singleton, background loader,
model worker or automatic preload.

## Plugin and installation compatibility

Trusted startup code can explicitly register a `ProviderDefinition` or load one
installed `capicove.model_providers` entry point by ID. A descriptor binds the
provider ID, version, capability, create/validate/dispose callbacks and contract
version 1. Duplicate names, mismatched IDs, unsupported contract versions and
incompatible returned ports are rejected. Configuration cannot cause arbitrary
module imports or automatic entry-point loading.

`capicove.model_installers` generalizes the existing installer protocol independently
of document parsing. A selected installer retains `model_plan(profile, destination)`
and `install_models(..., confirmed=...)` and RFC-0016's immutable file plan shape.
When no dedicated entry point exists, legacy document parser installers remain
available. Duplicates or malformed dedicated entries fail without fallback.
Paddle exposes both entry-point groups. CLI/MCP/HTTP use the same installer loader;
Tool IDs, schemas, permissions, plan digests and allowlists are unchanged.

BGE's explicit conversion script and image downloader remain available through
the new package. Converted output hashes depend on verified conversion results
and are recorded separately; they are not invented as pinned download hashes.
New conversion installers require a suitable output-verification contract before
they can claim a real shared-host installation journey.

The initial implementation decision was to make existing
`capicove.model_management` submodules alias canonical modules for
one compatibility cycle, preserving class identity and patch targets. Legacy
packaged lock files remain byte-identical. Receipts, model roots, source locks,
installed indexes and download command names are unchanged. Compatibility removal
needed a separately approved breaking-change decision. The amendment below
supersedes that compatibility-cycle decision.

### Legacy removal amendment (2026-10-13)

The operator explicitly removed `src/capicove/model_management` and requested
committing the resulting migration on local `dev`. The compatibility package and
its resource copies are removed immediately. Scripts and plugins must import
`capicove.model`; old imports fail. Locked manifest bytes, installed asset roots,
receipts, indexes and download command names remain unchanged. Tests assert the
legacy package is absent from source and installed wheels. This authorization
does not record non-author review or authorize a remote push or release.

## Validation and deferred work

Qualify strict configuration, wrong capability/port rejection, lazy imports,
identity isolation, lease serialization, capacity, cleanup, explicit plugin
discovery, canonical resources and legacy-package absence, installed-wheel usage and real sequential
CPU BGE/SigLIP2/Paddle operation. Run project quality, full tests, docs and build.
CLIP, audio and LLM provisioning remain deferred by operator instruction. Native
macOS/Windows qualification and broader shared-host model inference contracts are
not implied by this package migration.
