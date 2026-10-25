# Configure the local model factory

Operator-deployed vision language models use the `multimodal_generation`
capability and `factory.vlm(binding)`. See [VLM registration and serving](vlm.md)
for generic vLLM/OpenAI endpoint profiles, image/video input, Python, CLI and HTTP.

Text reranking is also available through the `text_reranking` capability and
`factory.text_reranker(binding)`. See [local text reranking](reranker.md) for the
pinned BGE reranker installer, Python binding, CLI, and HTTP endpoint.

`capicove.model` manages verified local models through named profiles and typed
leases. It supports BGE-M3 dense embeddings, SigLIP2 image/text embeddings, and the
Paddle three-model inventory for image OCR and document parsing. Optional SDKs
remain unloaded until the selected feature is used. Configuration never downloads
a model. Use the existing explicit [BGE/Paddle setup](local-bge-paddle.md) and
[SigLIP2 setup](image/siglip2-local-retrieval.md) first.

## Profiles and capability bindings

Save an operator-owned `models.toml`. Replace the root and digest placeholders
with your verified installation values; placeholders are deliberately invalid.
The BGE and SigLIP2 digests identify installed asset inventories. Paddle's digest
identifies its locked `book-balanced` profile manifest, verified against all three
installed models and their receipt.

```toml
schema_version = 1
max_instances = 1

[profiles.dense]
provider = "bge-m3"
capability = "text_embedding"
root = "/absolute/models/bge-m3"
manifest_sha256 = "REPLACE_WITH_TRUSTED_BGE_ASSET_SHA256"
device = "cpu"
[profiles.dense.options]
batch_size = 1
max_tokens = 512

[profiles.visual]
provider = "siglip2"
capability = "image_text_embedding"
root = "/absolute/models/siglip2-base-patch16-224"
manifest_sha256 = "REPLACE_WITH_TRUSTED_SIGLIP2_ASSET_SHA256"
device = "cpu"

[profiles.ocr]
provider = "paddleocr"
capability = "image_ocr"
root = "/absolute/models/paddleocr/book-balanced"
manifest_sha256 = "REPLACE_WITH_TRUSTED_PADDLE_PROFILE_SHA256"
device = "cpu"
[profiles.ocr.options]
enable_mkldnn = false

[profiles.pdf]
provider = "paddle-document"
capability = "document_parser"
root = "/absolute/models/paddleocr/book-balanced"
manifest_sha256 = "REPLACE_WITH_TRUSTED_PADDLE_PROFILE_SHA256"
device = "cpu"

[bindings]
"retrieval.text" = "dense"
"graph.embedding" = "dense"
"retrieval.visual" = "visual"
"graph.ocr" = "ocr"
"video.ocr" = "ocr"
"graph.parser" = "pdf"
"book.parser" = "pdf"
```

JSON with the same fields is also supported. Unknown fields, unknown provider
options, wrong capabilities, remote model IDs used as roots and invalid digests
are rejected. A configuration can contain profiles for SDKs absent from the
current environment; only acquiring those profiles requires their dependencies.

## Use existing typed services

```python
from pathlib import Path

from capicove.model import ModelFactory, load_configuration
from capicove.retrieval import TextEmbeddingService

with ModelFactory(load_configuration(Path("models.toml"))) as factory:
    with factory.text_encoder("retrieval.text") as encoder:
        service = TextEmbeddingService(encoder, space=encoder.space)
        vectors = service.encode(
            ("Solar panels generate electricity.",),
            space=encoder.space,
            role="document",
        )
```

For aligned embeddings borrow `factory.image_text_encoder("retrieval.visual")`
and call the existing `embed_images`, `embed_texts` or `rank` methods. For graph
OCR/PDF preparation, pass `model_factory=factory` to
`DocumentGraphConfiguration`; defaults are `graph.ocr` and `graph.parser`, with
configurable `ocr_binding` and `parser_binding`. Source hashes and parser/OCR
lineage remain in the existing evidence contracts.

The factory rejects a binding with the wrong capability before loading assets.
Model, preprocessing, dimensions, normalization and metric still identify an
embedding space. A changed space requires rebuilding its index.

## Local CLI composition

External clients can call an installed BGE-M3 through
[`capicove models serve bge-m3`](bge-m3-http.md), using the OpenAI embeddings
protocol and the same verified dense adapter.

Standalone image OCR serving is available through
[`capicove models serve paddleocr`](paddleocr-http.md). It uses an explicit
verified installation directory and a provider-independent FastAPI boundary.

```bash
capicove book prepare /absolute/books /absolute/document.pdf \
  --model-config /absolute/models.toml --model-binding book.parser \
  --evidence-only

capicove video filter --input /absolute/video.mp4 \
  --criteria /absolute/criteria.json \
  --model-config /absolute/models.toml --model-binding video.ocr --output json
```

Model configuration owns provider, model root and device. Do not combine it with
conflicting local model flags. `--profile` / `--profile-version` on Book continue
to select the parsing contract; the current Paddle profile is `book-balanced`,
version 2. Its scanned-PDF restrictions remain in effect. Video host mode
(`--runtime-config`) retains its existing model-filter restrictions and rejects
`--model-config`.

## Lifetime and extension

Borrowed model references stay inside their `with` blocks. Leases on the same
instance are serialized. One live instance is the default; an idle instance can
be released when a different profile is acquired, while active instances cannot
be evicted. If every permitted slot is active, acquisition raises
`ModelCapacityError`. A slot bound is not a guaranteed memory quota; runtimes can
retain allocator memory. Close the factory after the operation.

Trusted application startup can register `ProviderDefinition` objects with a
`ProviderRegistry`, or explicitly call `registry.load_plugin(provider_id)` for
one installed `capicove.model_providers` entry point. Configuration itself never
imports arbitrary classes or automatically discovers plugins. Descriptors must
use contract version 1 and return the corresponding public inference port.

Installers use `capicove.model_installers`; existing parser installers continue
to work as a compatibility fallback. The shared CLI/MCP/HTTP plan/verify/install
Tools retain their permissions and exact plan allowlists. BGE conversion keeps
its explicit script and separate output receipt; a real shared-host BGE install
journey has not been added by the factory migration.

## Migrate Python imports

Use `capicove.model` instead of `capicove.model_management`. The operator removed
the legacy package before this migration was committed; old imports now fail.
Update scripts and plugins to the new namespace. Locked model files retain their
original bytes. Existing model directories, receipts and indexes need no move.
The downloader command remains `capicove-download-image-models`.

## CPU qualification on this device

On 2026-10-13, installed core/plugin wheels were exercised outside the source
checkout using previously provisioned, hash-verified assets and offline runtime
settings. The Linux device has four logical CPUs, approximately 7.75 GiB RAM,
no swap and no usable GPU. Each model process ran sequentially.

| Journey | Result | Whole smoke duration | Peak process RSS |
| --- | --- | --- | --- |
| BGE-M3 factory → existing text embedding service | Chinese query ranked the expected English document first among two; 1024 dimensions | 69.84 s | 2048 MiB |
| SigLIP2 factory → image/text embeddings | Four English/Chinese queries ranked the expected authored image first among two; 768 dimensions | 51.41 s | 1362 MiB |
| Paddle factory → graph image OCR → PDF parsing | Both authored lines recognized; one PDF page parsed; original source hash verified; idle OCR instance released before parser acquisition | 68.76 s | 3166 MiB |
| Book CLI with `--model-config` | One section and one structure review task; zero summary tasks in evidence-only mode | Not separately measured | Not separately measured |
| Video CLI with `--model-config` | Three sampled frames passed OCR-area filtering; recognized line-box union ratio 0.17504 | Not separately measured | Not separately measured |
| Configured CLI → shared `model.plan` / `model.verify` bindings | Three Paddle models and 18 planned files verified offline; zero missing or mismatched files | Not separately measured | Not separately measured |

Durations include factory acquisition, verification and cold runtime loading;
they are not warm query latency or throughput measurements. The BGE smoke used
batch size 1 and 128-token inputs; BGE/SigLIP2 used two Torch CPU threads. Paddle
retained its bounded CPU policy. These authored fixtures establish composition
and evidence integrity, not production retrieval accuracy, Chinese OCR quality,
full Elasticsearch video search or native macOS/Windows qualification. CLIP,
audio and LLM models remain deferred. Local reproducibility scripts, configuration
and reports are under `/tmp/capicove-model-qualification`; that temporary directory
is excluded from source control and is not a durable release artifact.

The isolated Paddle environment initially lacked the optional host validation
dependency. `jsonschema==4.26.0`, matching the project lock, was installed from the
existing offline cache before the shared binding check. This verified plan and
read-only verification through the configured CLI; it did not perform a new
download/install or an MCP/HTTP transport session with real model assets.

Before legacy package removal, `make test` passed with 5,996 tests, 25 skips and
85.33% coverage; `make quality`, `make docs-check` (72 HTML pages) and `make build`
passed. The separate Paddle plugin suite passed 90 tests. Lifecycle regression
coverage also verifies that an evicted provider is collectible before creating
its replacement, rather than retained by a suspended lease generator.

After operator removal of the legacy package, the affected model, CLI, runtime,
graph and architecture regressions passed 303 tests, and the installed-wheel
check passed. Quality, documentation and build gates were rerun successfully.
The full-suite count and coverage above belong to the pre-removal run.
