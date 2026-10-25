# Document graph qualification

This procedure qualifies one exact, local-only document-to-graph configuration.
Passing the repository test suite establishes structural behavior; it does not
establish extraction, entity-resolution, diagnostic, or report quality for a
particular model. A provider is qualified only after an independently labeled
held-out corpus, a reviewed numeric evaluation profile, and the exact runtime
identities have been recorded together.

## Claim boundary

The graph core is a **text-origin pipeline**. Local Paddle PDF/image preparation
uses the landed public parser and image ports to supply derived text and retained
source evidence. Full source-aware PDF/OCR locator adaptation remains deferred.
General PDF/OCR quality, ASR,
discontinuous locators, model quality, throughput, and latency remain unqualified
until separately measured. Temporal Memory and Wiki writers are deliberately not
part of this capability.

Do not commit personal evaluation data, model weights, generated indexes,
credentials, or raw model responses. Keep the labeled corpus, protected artifacts,
receipts, SQLite catalogs, Parquet snapshots, and evaluation reports in an
owner-only, scope-specific local qualification root.

## Install the explicit adapter set

Use only the extras needed by the selected run:

```bash
uv sync --extra graph-models --extra text-embedding \
  --extra graph-community --extra graph-data
```

The extras mean:

| Extra | Qualified surface | Pinned range |
| --- | --- | --- |
| `graph-models` | local causal LLM and tokenizer adapters | Torch 2.x, Transformers 4.48–5.x |
| `text-embedding` | local standard BERT/CLS Sentence Transformers BGE adapter | Sentence Transformers 5.2.x, Torch 2.x, Transformers 4.48–5.x |
| `graph-community` | NetworkX Louvain adapter | NetworkX 3.6.x |
| `graph-data` | typed Parquet publication and verified reads | PyArrow 25.x |

Installing an extra does not select, download, or qualify a model. Network and
telemetry remain disabled by policy; dependency installation itself is a separate,
explicitly authorized operation.

## Provision and verify local assets

`LocalModelAssets` is the asset manifest contract. There is intentionally no Hub
identifier, automatic manifest-path loader, or download fallback. The host loads a
reviewed manifest from its own protected configuration and constructs:

```python
from pathlib import Path

from capicove.model_management.local_assets import LocalModelAssets

llm_assets = LocalModelAssets(
    root=Path("/absolute/private/models/graph-llm"),
    files=tuple(reviewed_llm_file_hashes),  # sorted (relative_path, sha256) pairs
    manifest_sha256=reviewed_llm_manifest_sha256,
)
tokenizer_assets = LocalModelAssets(
    root=Path("/absolute/private/models/graph-tokenizer"),
    files=tuple(reviewed_tokenizer_file_hashes),
    manifest_sha256=reviewed_tokenizer_manifest_sha256,
)
bge_assets = LocalModelAssets(
    root=Path("/absolute/private/models/bge"),
    files=tuple(reviewed_bge_file_hashes),
    manifest_sha256=reviewed_bge_manifest_sha256,
)

for assets in (llm_assets, tokenizer_assets, bge_assets):
    assets.verify()
```

Each root must be an existing absolute path with no symlink component. The closed
inventory contains only regular, singly linked, approved data files; every digest
and the digest of the sorted manifest are rechecked. Executable model code,
`adapter_config.json`, remote-code configuration, traversal, unexpected files,
unsafe weight maps, and non-safetensors LLM weights are rejected. Protect roots
against concurrent mutation for the entire adapter lifetime.

Build the local generation profile with an explicit device (`cpu`, `mps`, or an
exact `cuda:N`), context window, output reservation, and revision. The adapter
always uses `local_files_only=True`, `trust_remote_code=False`, safetensors, fixed
system/user roles, temperature zero, and no cloud fallback:

```python
from capicove.graph_build._adapters.local_generation import (
    LocalStructuredGenerator,
    LocalTextTokenizer,
)
from capicove.graph_build.profiles import LocalGenerationProfile

local_generation = LocalGenerationProfile(
    assets=llm_assets,
    tokenizer_assets=tokenizer_assets,
    model_revision="sha256:" + llm_assets.manifest_sha256,
    tokenizer_revision="sha256:" + tokenizer_assets.manifest_sha256,
    device="cpu",
    context_limit=8192,
    max_new_tokens=2048,
)
generator = LocalStructuredGenerator(local_generation)
tokenizer = LocalTextTokenizer(local_generation)
```

The BGE profile must pin the complete `EmbeddingSpaceRef`, prefixes, verified
token limit, local bundle revision, output dimension, L2 normalization, and cosine
metric. The initial adapter supports the standard BERT + CLS Sentence Transformers
layout, optionally followed by Normalize; a BGE family name alone is insufficient.
Use `bge_preprocessing_version(...)` to derive the exact preprocessing identity,
then compose `BgeTextBatchEncoder` and the public `TextEmbeddingService`:

```python
from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval import TextEmbeddingService
from capicove.retrieval._adapters.bge import (
    BgeTextBatchEncoder,
    BgeTextEmbeddingProfile,
    bge_preprocessing_version,
)

preprocessing = bge_preprocessing_version(
    bge_assets,
    query_prefix=reviewed_query_prefix,
    document_prefix=reviewed_document_prefix,
    max_tokens=reviewed_bge_max_tokens,
)
space = EmbeddingSpaceRef(
    space_id=reviewed_space_id,
    provider_id="local-bge",
    model_revision="sha256:" + bge_assets.manifest_sha256,
    preprocessing_version=preprocessing,
    dimension=reviewed_bge_dimension,
    normalization=VectorNormalization.L2,
    distance_metric=DistanceMetric.COSINE,
    query_modalities=frozenset({QueryModality.TEXT}),
    target_modalities=frozenset({TargetType.TEXT_CHUNK}),
)
bge_profile = BgeTextEmbeddingProfile(
    assets=bge_assets,
    space=space,
    query_prefix=reviewed_query_prefix,
    document_prefix=reviewed_document_prefix,
    max_tokens=reviewed_bge_max_tokens,
)
bge_encoder = BgeTextBatchEncoder(bge_profile, device="cpu", batch_size=32)
embedding_service = TextEmbeddingService(bge_encoder, space=space)
```

## Offline preflight

Run these checks before loading Torch or Transformers:

1. Verify each `LocalModelAssets` inventory and record its absolute protected
   configuration locator and `manifest_sha256` (do not publish the local path).
2. Verify that the generation profile identities exactly match the accepted
   `InferencePolicy`; verify the BGE `EmbeddingSpaceRef` matches both the identity
   and retrieval space selected by `GraphBuildProfile`. Verify that the trusted
   source-provenance resolver classifies every current document and carries a real
   derivation reference for every generated input.
3. Record Python implementation/version, Capicove revision, OS, device, available
   memory, and the exact NetworkX/PyArrow/Torch/Transformers/Sentence Transformers
   versions.
4. Set the host runtimes to offline mode as defense in depth, and monitor the
   qualification process for network access. The Capicove adapters themselves do
   not acquire assets.
5. Verify the authorized held-out corpus and independent annotation digests before
   starting. The generator that produced predictions must not certify its own
   output.

This checkout was inspected on 2026-09-14. No graph LLM/BGE environment locator
was configured, the default macOS Capicove model directory was absent, and no
authorized labeled corpus was present in the repository. Therefore adapter unit
tests can run here, but a real local-model smoke or model-quality claim cannot be
made from this checkout alone.

## Required deterministic matrix

Run the real library adapters and the complete repository gates:

```bash
uv run --extra graph-data --extra graph-community pytest \
  tests/integration/graph_build tests/e2e/test_document_graph_pipeline.py \
  --no-cov -q

uv run --extra graph-models --extra text-embedding pytest \
  tests/unit/graph_build/test_local_generation.py \
  tests/unit/retrieval/test_bge_adapter.py --no-cov -q

make quality
make test
make build
make docs
```

The first command exercises real NetworkX and PyArrow. The second exercises
adapter logic with authored test doubles; it is not a real-model smoke and does
not certify quality. Inspect the wheel and sdist for graph schemas and
`NETWORKX_LICENSE.txt`, and verify that importing `capicove` and
`capicove.graph_build` without optional SDKs does not import Torch, Transformers,
Sentence Transformers, NetworkX, or PyArrow.

## Quality run

On the authorized corpus, run the exact `GraphBuildService` composition described
in the [operator guide](../site/guides/document-graph-pipeline.md). Retain the
immutable predicted identifiers and independently authored labels needed by
`evaluate_graph_run`.

The evaluation corpus must have a held-out split and independent labels for:

- entities by base/event/diagnostic ontology layer and by Chinese,
  mixed-language, table, reflection, feedback, and reference-book stratum;
- within-snapshot merges, cross-snapshot alignment decisions and abstentions;
- repeated versus distinct event occurrences;
- diagnostic evidence support;
- report statement entailment and counter-evidence coverage.

It must also include fixed-token chunking and name/type-only grouping predictions
on the same split. Create a reviewed `EvaluationProfile` with numeric gates before
viewing final results, then call:

```python
from capicove.graph_build.evaluation import evaluate_graph_run

report = evaluate_graph_run(predicted_run, labeled_corpus, evaluation_profile)
```

Retain the resulting `QualificationReport` with corpus/profile/model hashes,
runtime identities, counts, denominators, strata, timing, token/call/cache counts,
baseline comparisons, and pass/fail reasons. A metric without a denominator is
`None`; it is never converted to 1.0. Failed structural checks always prevent
qualification. Changing a prompt, schema, tokenizer, model, BGE preprocessing,
embedding space, partition profile, threshold, runtime version, or labeled split
requires a new profile and requalification.

## Initial bounded profile

These are implementation bounds, not measured capacity or latency claims:

| Area | Initial default or hard bound |
| --- | --- |
| Text input | 1,000,000 code points per representation; at most 50,000 tokenizer calls and 8,000,000 counted code points during chunking |
| Chunking | target 800 tokens, core maximum 1,200, non-core context budget 100 |
| Inference | output reservation 4,096 tokens; at most 2 concurrent units, 3 calls/unit, 1 repair/unit; 120 s call and 3,600 s build deadlines |
| Extraction | 64 entities and 128 relationships/unit; 1,048,576 request bytes; 262,144 response bytes; 4,096 text and 8,192 quote code points; 64 spans/record |
| Resolution | top-k 10; 2,000 pairs/batch; default pool 2,000; hard pool cap 10,000 entities |
| Diagnostics | 128 assertions/bundle and 128 bundles/build |
| Communities | max leaf size 50, hierarchy depth 3, support cap 20; Louvain resolution 1.0, threshold `1e-7`, max level 10, seed 42 |
| Reports | 128 assertions, 32 findings, 256 child statements, 65,536 response bytes, 32,768 code points, 100 statements, 2,000 code points/statement |
| Retrieval projection | batch size 32 by default, maximum 256; ranges are exhaustive and never silently truncated |

Actual supported corpus size, runtime, memory, latency, and quality are
**measured-or-unqualified** values. Record measurements for the exact host rather
than deriving a claim from these safeguards.

## Qualification record checklist

- Graph manifest schema version, every one of the 17 table schema identities, and
  all stage coverage/issue codes.
- Full `GraphBuildProfile` fingerprint and closed subprofile versions.
- LLM/tokenizer/BGE manifest digests, model revisions, context/token limits,
  prefixes, embedding space, device, and generation profile.
- NetworkX algorithm/adapter/package/Python identity, Louvain parameters, node and
  edge ordering, and projection fingerprint. Replay is only promised within the
  recorded version; it is not promised across NetworkX versions.
- Input/baseline snapshot IDs, scope, authorization revision, run ID, graph
  snapshot ID, source-provenance resolver/classification fingerprint, retrieval
  binding ID, exact retrieval generation IDs and index format version.
- Held-out corpus, annotation, evaluator, reviewer approval, fixed baseline, and
  report hashes; raw counts and undefined denominators as well as aggregate rates.
- Structural matrix, recovery matrix, package inspection, docs build, and any
  unverified adapter, origin, platform, or performance limitation.
