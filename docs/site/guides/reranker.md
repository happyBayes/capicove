# Local text reranking

Capicove can install and serve `BAAI/bge-reranker-v2-m3` through Python, CLI,
and HTTP. Reranking scores query/document pairs after candidate retrieval.
It does not create embeddings or retrieve documents on its own.

The initial provider uses upstream float32 safetensors and the fast tokenizer,
without conversion or remote model code. The license is Apache-2.0. The pinned
revision is `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`:
[upstream model card](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e).

## Install explicitly

```bash
uv pip install 'capicove[model-serving]'
capicove models install bge-reranker-v2-m3 --profile default \
  --destination /absolute/path/bge-reranker-v2-m3 --dry-run
```

Authorize the pinned 2,288,173,057-byte download:

```bash
capicove models install bge-reranker-v2-m3 --profile default \
  --destination /absolute/path/bge-reranker-v2-m3 --yes
```

The installer stores five pinned runtime files under
`DESTINATION/bge-reranker-v2-m3/`, verifies every SHA-256, and atomically writes
an adjacent `DESTINATION.receipt.json` with provenance. This layout matches the
configured `model.plan`, `model.install`, and `model.verify` Tools. An existing
empty destination is supported. The inventory digest is
`5c059b4359a3e16d3cc3d9d37e09caee5804d8f3b1086b94fb6974429ab4fb64`.
An existing model must match the complete inventory and is never silently
replaced. Repeating installation repairs a missing or interrupted receipt without
downloading verified weights again. Receipt symlinks, hard links and nonregular
files are rejected; a failed write preserves the previous complete receipt.

Legacy flat model directories remain readable by Python, CLI, and HTTP. Running
the explicit installer against a verified flat directory migrates it to the
standard subdirectory without downloading. The original files remain intact
until the verified copy and receipt are published; only unchanged original
inventory files are then removed. Keep assets outside source control and prevent
concurrent writes during installation or model loading. Network downloads occur
only in the explicit install command when verified local assets are unavailable.

## Serve HTTP

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
capicove models serve bge-reranker-v2-m3 \
  --model-directory /absolute/path/bge-reranker-v2-m3 \
  --device cpu --batch-size 4 --max-tokens 512
```

The default listener is `127.0.0.1:8768`. Startup hashes the complete inventory
before and after loading and completes a warmup before opening the listener.
CPU weights are copied into owned memory to detach safetensors file mappings.
Subsequent requests use resident weights and tokenizer without rehashing files.
The next acquisition verifies disk changes. `--model-manifest-sha256` is optional
because this provider pins its inventory in the package; an explicitly supplied
digest must match.

```bash
curl -sS http://127.0.0.1:8768/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "What does a giant panda eat?",
    "documents": [
      "HTTP transfers web pages.",
      "Giant pandas mainly eat bamboo.",
      "Tokyo is the capital of Japan."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

`POST /v1/rerank` is a Capicove text reranking endpoint, not an OpenAI embeddings
operation. Its supported request fields are:

| Field | Meaning |
| --- | --- |
| `model` | `BAAI/bge-reranker-v2-m3` or `bge-reranker-v2-m3` |
| `query` | Nonblank text |
| `documents` | 1–128 nonblank text strings, in original candidate order |
| `top_n` | Optional integer from 1 to candidate count; default: all candidates |
| `return_documents` | Optional boolean; default: false |

The response contains `model`, `model_revision`, `score_type: "raw_logit"`,
`results`, and `usage.total_tokens`. Each result retains its original zero-based
`index` and `relevance_score`. If requested, `document.text` contains unchanged
input text. Scores descend; ties retain input order. Raw logits may be negative
and are not calibrated probabilities or comparable across models. Usage counts
actual unpadded pair tokens, including repeated query and special tokens.

`GET /health`, `GET /v1/models`, and `GET /v1/models/{model}` expose readiness and
identity. Request bodies are limited to 2 MiB, each query/document pair to the
configured token limit, and a request to 32,768 unpadded pair tokens. Long inputs
are rejected, never silently truncated. The default pair limit is 512; operators
can set 1–8192. Qualify larger batch and token limits on the deployment host.

One inference is admitted at a time; excess calls return 429. Health remains
responsive during inference. Disconnected requests retain their slot until
native work finishes. A failed thread submission or cancellation before the
worker starts returns the slot, and any late worker skips inference. Invalid
input returns 400, an unknown model 404, oversized
bodies 413, and inference failures 500 with no runtime path or request content
in the error. Device or model fallbacks never occur automatically.

For non-loopback binding, set a bearer key through your secret manager and pass
`--api-key-env CAPICOVE_RERANKER_KEY`. All routes except `/health` then require
the `Authorization: Bearer ...` header. Use the host's TLS boundary for remote
exposure. Serving disables HTTP access logs.

## Run the CLI

Save the same request to `request.json`; `model` is optional in the CLI:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
capicove models rerank \
  --model-directory /absolute/path/bge-reranker-v2-m3 \
  --input request.json
```

Use `--input -` for standard input. The CLI prints the same result JSON as HTTP,
closes the model, and returns nonzero on failure. For repeated calls, use the
resident HTTP service to avoid process startup and model reload.

## Use the model factory

Existing version-1 model configurations can select the new typed capability:

```toml
schema_version = 1

[profiles.ranker]
provider = "bge-reranker-v2-m3"
capability = "text_reranking"
root = "/absolute/path/bge-reranker-v2-m3"
# Public pinned asset inventory digest, not a credential.
manifest_sha256 = "5c059b4359a3e16d3cc3d9d37e09caee5804d8f3b1086b94fb6974429ab4fb64" # pragma: allowlist secret
device = "cpu"

[profiles.ranker.options]
batch_size = 4
max_tokens = 512

[bindings]
"retrieval.rerank" = "ranker"
```

```python
from pathlib import Path

from capicove.model import ModelFactory, load_configuration

with ModelFactory(load_configuration(Path("/absolute/path/models.toml"))) as factory:
    with factory.text_reranker("retrieval.rerank") as reranker:
        result = reranker.rerank(
            "What does a giant panda eat?",
            ("HTTP transfers web pages.", "Giant pandas mainly eat bamboo."),
            top_n=1,
        )
        print(result.results[0].index)  # 1
```

Other capabilities can depend on the public `TextReranker` port exported from
`capicove.retrieval`. Pass candidate texts in order and map result indices back
to original candidates to retain evidence and lineage. Search callers explicitly
compose candidate retrieval followed by reranking; existing search defaults
remain unchanged. Third-party programs can call HTTP without importing Capicove
or model SDKs.
