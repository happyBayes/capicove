# Serve BGE-M3 with the OpenAI embeddings protocol

Run an installed BGE-M3 model as an HTTP service for Python, JavaScript or other
clients. It returns 1024-dimensional, L2-normalized dense vectors through the
same verified adapter as Capicove retrieval. Sparse and ColBERT outputs are not
exposed.

## Install and select the model

```bash
python -m pip install 'capicove[api,text-embedding]'
```

This installs runtimes, not weights. Provision the pinned, converted safetensors
bundle using the [BGE-M3 installation workflow](local-bge-paddle.md). Retain
`bge-m3.receipt.json` next to `bge-m3/` and the independently trusted
`asset_manifest_sha256` from installation. Do not automatically accept a new
digest from a replacement receipt. Existing verified installations can be reused.
Acquisition is a separate explicit network opt-in. Serving never downloads models
or contacts a cloud API.

## Start the service

Replace the absolute path and digest with the selected installation's values:

```bash
capicove models serve bge-m3 \
  --model-directory /absolute/models/bge-m3 \
  --model-manifest-sha256 REPLACE_WITH_TRUSTED_ASSET_SHA256 \
  --device cpu --port 8766
```

The default address is `127.0.0.1:8766`. Startup verifies the inventory and warms
one persistent model before listening. Subsequent requests reuse it; normal
shutdown closes it. Select `--device cuda:0` or `mps` only with its installed,
working runtime. There is no automatic device fallback. Defaults are
`--batch-size 8` and `--max-tokens 512`; the token limit can be set up to 8192
within verified model limits. Batch size controls model microbatches, not the
number of texts permitted in an HTTP request.

## Call from another program

```bash
curl http://127.0.0.1:8766/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-m3","input":["Solar energy","\u592a\u9633\u80fd"],"encoding_format":"float"}'
```

The response has `object: "list"`, `model: "BAAI/bge-m3"`, indexed `data` items
with `object: "embedding"` and a 1024-element `embedding`, plus
`usage.prompt_tokens` and `usage.total_tokens`. Usage counts actual BGE tokens,
including special tokens and excluding padding. Indices preserve input order
and duplicates. `bge-m3` is also accepted as a request model alias.

The official OpenAI Python SDK can call the endpoint. Install `openai` in the
calling program's environment:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8766/v1", api_key="local")
response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["Solar energy", "\u592a\u9633\u80fd"],
)
vectors = [item.embedding for item in response.data]
```

With no server key configured, `local` is an SDK placeholder. With authentication
configured, supply the actual server key through the client environment.

## Protocol and bounds

| Surface | Behavior |
| --- | --- |
| `POST /v1/embeddings` | One string or 1–128 nonblank strings |
| `encoding_format` | `float` by default, or `base64` of little-endian float32 bytes |
| `dimensions` | Omit or set to 1024; other values are rejected |
| `user` | Optional bounded string; ignored and not stored |
| `GET /v1/models` | Lists the selected model |
| `GET /v1/models/BAAI/bge-m3` | Returns its model descriptor |
| `GET /health` | Readiness, model name and dimension; no key required |

This implements the text-input subset of the
[OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create).
Token-ID arrays are rejected because another model's token IDs do not define
BGE-M3 input. Send original text. No input or vector is silently truncated.
Requests are limited to 2 MiB, 128 texts, the configured per-input token limit
and 300,000 summed tokens. Errors use an OpenAI error envelope: invalid requests
400, unknown models 404, oversized bodies 413, busy inference 429 and runtime
failures 500. One request performs inference at a time; health remains responsive.
Input text and keys are not logged or persisted by this service.

## Optional authentication and remote binding

Set a private bearer key in an operator-controlled environment variable, then
reference its name instead of putting the value on the command line:

```bash
capicove models serve bge-m3 \
  --model-directory /absolute/models/bge-m3 \
  --model-manifest-sha256 REPLACE_WITH_TRUSTED_ASSET_SHA256 \
  --host 0.0.0.0 --port 8766 \
  --api-key-env CAPICOVE_BGE_API_KEY
```

`CAPICOVE_BGE_API_KEY` must already contain the selected nonempty key. Every
route except `/health` requires `Authorization: Bearer <key>`. Non-loopback
binding is refused without a key. Restart to rotate it. This is a standalone
single-owner model endpoint; it provides no TLS or workspace Tool permissions.
An exposed deployment must provide TLS termination and its network access policy.
The existing [`capicove http serve`](../cli/http.md) remains a separate Tool host.
