# Local SigLIP2 retrieval on CPU

Use the official Google SigLIP2 model to debug image/text embeddings and ranking
through Capicove's standalone local API. This path needs an explicitly downloaded
model and a CPU PyTorch runtime. The Linux CPU run on 2026-10-12 passed the
authored image checks and an installed-wheel replay; see the
[device evaluation](../../development/siglip2-cpu-evaluation.md) for results and
scope. The separate [video retrieval workflow](../video/retrieval.md) now has
local real-service acceptance; this model API smoke test alone does not start
Elasticsearch or requalify that composed workflow.

The API accepts image uploads and text strings and returns model identity,
768-dimensional normalized embeddings, or ranked text candidates. Downloading
requires an explicit network opt-in; serving and inference use local files.

## Prepare the runtime

Run these commands from a Capicove checkout with Python 3.11 and `uv`. They
create an isolated environment outside the source repository. Dependency
installation accesses the package indexes; run it only when authorized.

The measured device already has this environment and model. Reuse them and
continue at offline inference. For a fresh installation, choose an unused
environment path instead of recreating an environment that another process uses.

```bash
export CAPICOVE_SIGLIP2_ENV=/tmp/capicove-siglip2-env
export CAPICOVE_MODEL_HOME=/tmp/capicove-models
export HF_HOME=/tmp/capicove-hf-home

uv venv --python python3.11 "$CAPICOVE_SIGLIP2_ENV"
uv pip install --python "$CAPICOVE_SIGLIP2_ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.13.0+cpu'
uv pip install --python "$CAPICOVE_SIGLIP2_ENV/bin/python" \
  'transformers==5.15.0' 'tokenizers==0.22.2' 'huggingface-hub==1.33.0'
uv pip install --python "$CAPICOVE_SIGLIP2_ENV/bin/python" \
  -e '.[model-serving]'

export PATH="$CAPICOVE_SIGLIP2_ENV/bin:$PATH"
hf version
```

These inference-library pins match the measured run. Other dependencies resolve
within the package's declared ranges; record their versions before comparing
results. `model-serving` supplies the PyTorch/Transformers API runtime;
`models-cpu` alone supplies ONNX execution and does not install PyTorch.

For a package-install replay, build the wheel with `make build`, install that
wheel into the isolated environment, and invoke it outside the source directory
with `PYTHONPATH` unset. The measured replay loaded Capicove from the environment's
`site-packages`, with the same local model and inference-library versions.

## Inspect and download the locked model

The existing `scripts/download_image_models.py` and installed
`capicove-download-image-models` command already select the official
`google/siglip2-base-patch16-224` preset. Its lock entry is in
`src/capicove/model/image_models.toml`.

| Setting | Locked value |
| --- | --- |
| Model key | `siglip2-base-patch16-224` |
| Repository and type | `google/siglip2-base-patch16-224`, model |
| Revision | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| License | Apache-2.0 |
| Format | Original Transformers safetensors checkpoint; no conversion or quantization |

Inspect the pinned download and verification commands without network access:

```bash
capicove-download-image-models \
  --model siglip2-base-patch16-224 \
  --destination "$CAPICOVE_MODEL_HOME" \
  --dry-run
```

After authorizing this exact official upstream model and local destination, run:

```bash
capicove-download-image-models \
  --model siglip2-base-patch16-224 \
  --destination "$CAPICOVE_MODEL_HOME" \
  --max-workers 2
```

The downloader runs official `hf download` at the immutable revision, checks the
complete inventory through `hf cache verify`, writes `capicove-model-receipt.json`,
and publishes the model directory. A subsequent download command rehashes an
existing installation before reporting `available`. The measured installation
contained nine upstream files totaling 1,539,458,338 bytes, approximately 1.54 GB,
excluding the receipt.

The command prints `model_manifest_sha256`. Store this value in trusted host
configuration outside the writable model directory when the controlled download
returns `status=installed`. Runtime composition must use that trust anchor; do
not derive it from an untrusted receipt or treat a later `status=available`
response as a new trust decision.

Select an authorized model repository and exact file scope explicitly. This
guide uses the official upstream download; it does not authorize uploads,
mirroring, or an automatic fallback between repositories.

## Start offline inference

Keep the environment and model-root variables from the previous steps. Set the
digest from the controlled download output. This value belongs to the measured
installation, not to an arbitrary replacement model:

```bash
# Public model file-manifest SHA-256, not a credential.
export CAPICOVE_SIGLIP2_MODEL_MANIFEST_SHA256=272cc96aa5009e2da573fe42e8379f6c0779af11f6b9fee2d9fffdfc67d8fdff # pragma: allowlist secret

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
OMP_NUM_THREADS=4 \
"$CAPICOVE_SIGLIP2_ENV/bin/python" scripts/serve_siglip2.py \
  --model-directory "$CAPICOVE_MODEL_HOME/siglip2-base-patch16-224" \
  --model-manifest-sha256 "$CAPICOVE_SIGLIP2_MODEL_MANIFEST_SHA256" \
  --device cpu --host 127.0.0.1 --port 8765
```

Startup checks the receipt and model file hashes, then loads only local files
with remote model code disabled. A missing model or mismatched hash fails startup;
it does not trigger a download. Wait for Uvicorn's startup message before querying.

In another terminal:

```bash
curl --fail http://127.0.0.1:8765/health
```

Expected fields include `status: "ok"`, the locked model ID and revision, and
`device: "cpu"`. The service has no root-page UI: use `/health` and the API routes
below. Press Ctrl-C in the server terminal to stop it. Keep the listener on
loopback; this API is separate from the workspace host's authenticated `http serve`.

## Run an authored image query

This example creates a synthetic image outside the repository. It requires the
prepared environment and the running server; it downloads no fixture. In the
second terminal, set `CAPICOVE_SIGLIP2_ENV` to the prepared environment path.

```bash
export CAPICOVE_SIGLIP2_ENV=/tmp/capicove-siglip2-env
"$CAPICOVE_SIGLIP2_ENV/bin/python" - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

root = Path('/tmp/capicove-siglip2-example')
root.mkdir(mode=0o700, exist_ok=True)
image = Image.new('RGB', (224, 224), 'white')
ImageDraw.Draw(image).ellipse((40, 40, 184, 184), fill='red')
image.save(root / 'red-circle.png')
PY

curl --fail http://127.0.0.1:8765/v1/similarity \
  -F 'image=@/tmp/capicove-siglip2-example/red-circle.png' \
  -F 'texts=["a red circle on a white background","a blue square on a white background","a green triangle on a white background","a black star on a white background"]'
```

The measured run ranked the red-circle description first. Each match contains
`text`, `rank`, `logit`, `match_probability`, and `candidate_probability`.
`match_probability` is the model's sigmoid score; `candidate_probability` is
normalized across the supplied candidates. Neither is a general accuracy measure.

For embedding inference:

```bash
curl --fail http://127.0.0.1:8765/v1/embeddings/image \
  -F 'image=@/tmp/capicove-siglip2-example/red-circle.png'

curl --fail http://127.0.0.1:8765/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"texts":["a red circle on a white background"]}'
```

Both responses include model ID/revision, device, `dimension: 768`, and normalized
vectors under `embeddings`. These endpoints encode queries; building and querying
a persisted index requires an explicitly composed retrieval runtime.

## Check the project integration

With the model server running, use the repository's development test environment:

```bash
CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT=http://127.0.0.1:8765 \
uv run pytest --no-cov -q \
  plugins/retriever-elasticsearch/tests/integration/test_local_model_images.py
```

This opt-in test exercises `LocalSiglip2QueryEncoder` through real loopback HTTP,
checking image-query/preview consistency, model identity, dimension, and
normalization. It does not start Elasticsearch or qualify the whole video pipeline.

The [device evaluation](../../development/siglip2-cpu-evaluation.md) also records
eight queries against a persisted NumPy exact visual index, reopen checks, API
checks, and an installed-wheel replay. For the separate video composition, follow
[experimental video retrieval](../video/retrieval.md).

## Preserve or remove local assets

The example environment, model, Hub cache, and generated images live under `/tmp`.
They may be removed by OS cleanup. For continued use, choose an approved durable
model directory outside the source repository during download, retain the external
trust anchor, and revalidate the installation after any move. Regenerate disposable
fixtures and indexes as needed. Do not commit weights, caches, fixtures, raw local
reports, credentials, or generated indexes.
