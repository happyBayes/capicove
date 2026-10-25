# SigLIP2 CPU retrieval evaluation

On 2026-10-12, the official locked SigLIP2 model passed a local functional smoke
test on this Linux CPU device. Four authored geometric images produced eight
correct English/Chinese Top-1 results through the source runtime, the installed
wheel, and the real HTTP query encoder. This establishes a local debugging path
for these fixtures; it does not qualify real-world semantic recall, production
latency, or shared workspace video Search.

See [download and configuration steps](../guides/image/siglip2-local-retrieval.md)
to provision the model and run the existing API and integration test.

## Device and tested runtime

| Item | Measured configuration |
| --- | --- |
| Source revision | `ec7651ec5956a0cbbddcebc65a52d51e55a4a950` |
| OS | Linux 6.8.0-138-generic, x86_64, glibc 2.39 |
| CPU presented by the VM | `DO-Regular`; four logical CPUs available to the process |
| Memory | 8,131,776 KiB physical memory reported; no swap |
| Device and threads | CPU; four PyTorch intra-op threads, `OMP_NUM_THREADS=4` |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cpu |
| Transformers / tokenizers | 5.15.0 / 0.22.2 |
| Hugging Face CLI and Hub library | 1.33.0 |
| NumPy / Pillow | 2.4.6 / 12.3.0 |
| OpenCV headless | 4.14.0.94 |
| FastAPI / Uvicorn | 0.143.0 / 0.54.0 |

The model runtime used `/tmp/capicove-siglip2-env`. PyTorch used the CPU variant
of the repository's locked 2.13.0 release; Transformers, tokenizers, and Hub
versions matched their lock entries. Other packages resolved within declared
ranges, so this was not an exact copy of every package in the development lock.
Inference used `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_HUB_DISABLE_TELEMETRY=1`.

## Model identity and verification

| Item | Value |
| --- | --- |
| Repository and type | `google/siglip2-base-patch16-224`, model |
| Immutable revision | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| License and provenance | Apache-2.0; original official Google checkpoint |
| Format | Transformers safetensors; no conversion or quantization |
| Preprocessing identity | `transformers-siglip2-processor-v1` |
| Embedding dimension | 768 |
| Verified upstream files | 9 |
| Downloaded size | 1,539,458,338 bytes, excluding the installation receipt |
| Local directory | `/tmp/capicove-models/siglip2-base-patch16-224` |

The packaged downloader completed `hf cache verify` against the exact repository
and revision, including missing/extra-file checks, before writing its receipt.
Runtime composition rehashed the installation against the externally retained
file-manifest digest. Architecture, training provenance, and intended use are
described in the
[pinned official model card](https://huggingface.co/google/siglip2-base-patch16-224/blob/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md).

Public artifact identities, not credentials:

```text
model_manifest_sha256 = 272cc96aa5009e2da573fe42e8379f6c0779af11f6b9fee2d9fffdfc67d8fdff
source_manifest_sha256 = c24c9dffdc8add4c17000af130e958f24345645397adc7443a64b4ba33f349b2
model.safetensors.sha256 = 612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b
installation_receipt_sha256 = c978d883589ca6556f5be98d2406daf00770c9d7de50bef6ec9df5b3b4d079b7
tested_wheel_sha256 = 180243cacd60ee8a7e5d531c9f624012404a0541cc3b3d87bc1fa46768a3f9cd
```

The wheel hash identifies the package tested before this documentation change,
not a later rebuild with different distribution metadata.

## Fixtures and retrieval method

The PNG fixtures were generated locally with Pillow 12.3.0 before inference.
Each was RGB, 224 by 224 pixels, with a white background. Expected answers were
defined by authored color and geometry, independently of model output.

| Fixture | Drawing geometry |
| --- | --- |
| `red-circle` | Red ellipse with bounding box `(40, 40, 184, 184)` |
| `blue-square` | Blue rectangle with bounding box `(45, 45, 179, 179)` |
| `green-triangle` | Green polygon: `(112, 30)`, `(30, 190)`, `(194, 190)` |
| `black-star` | Black ten-vertex polygon around `(112, 112)`, starting at `-pi/2`, advancing `pi/5`, with alternating radii 83 and 35 |

Each English query was `a <color> <shape> on a white background`. Chinese inputs
are listed below; they were authored test inputs, not translated model outputs.
The candidate corpus contained exactly these four images, with no additional
distractors.

`Siglip2RetrievalService` produced L2-normalized image and text embeddings. Image
vectors were published into `NumpyExactVisualRetriever` as four keyframe units in
one pinned generation. Exact inner product ranked text queries. Validation checked
the stored generation before activation and after closing and reopening the index.

Source and installed-wheel runs also checked image self-retrieval, the API's
health/ranking/image-embedding/text-embedding routes, vector consistency between
direct service calls and the API, and blank-text rejection with HTTP 422. The
wheel replay ran from `/tmp`, with `PYTHONPATH` empty and Capicove imported from
the isolated environment's `site-packages`.

For the real HTTP path, `scripts/serve_siglip2.py` listened only on
`127.0.0.1:8765`. `LocalSiglip2QueryEncoder` encoded the eight queries and searched
the stored image vectors. Its image-query and preview paths matched within
absolute tolerance `1e-6`. The temporary server stopped cleanly after testing.

## Retrieval results

All eight queries returned the expected image first. Source and wheel runs
produced the same scores below; real HTTP returned the same Top-1 IDs. Scores
are normalized-vector inner products, not probabilities.

| Language | Expected and returned Top-1 | Top-1 score | Runner-up score |
| --- | --- | --- | --- |
| English | `red-circle` | 0.188819 | 0.074278 |
| English | `blue-square` | 0.191945 | 0.085727 |
| English | `green-triangle` | 0.201854 | 0.066535 |
| English | `black-star` | 0.187060 | 0.078903 |
| Chinese | `red-circle` | 0.157467 | 0.070937 |
| Chinese | `blue-square` | 0.137616 | 0.110574 |
| Chinese | `green-triangle` | 0.166464 | 0.063028 |
| Chinese | `black-star` | 0.161267 | 0.085309 |

| Chinese test input | Expected image |
| --- | --- |
| Red circle on a white background | `red-circle` |
| Blue square on a white background | `blue-square` |
| Green triangle on a white background | `green-triangle` |
| Black pentagram on a white background | `black-star` |

| Check | Outcome |
| --- | --- |
| English text-to-image Top-1 | 4/4 in each execution path |
| Chinese text-to-image Top-1 | 4/4 in each execution path |
| Image self-retrieval | 4/4 in source and wheel runs |
| Dimension and normalization | 768; squared L2 norm within `1e-5` of 1 |
| Persisted index validation and reopen | Passed; English ranking order preserved |
| API/direct-service vector consistency | Passed at absolute tolerance `1e-6` |
| Real HTTP image-query/preview consistency | Passed at absolute tolerance `1e-6` |
| Existing real-model integration test | Passed within the 118-test focused run |

Self-retrieval used each original image as its own query. It checks vector/index
consistency and does not measure similarity between independently captured images.
Color and shape changed together, so the results do not isolate shape
understanding from color recognition.

## Observed CPU timings

| Operation | Source run, seconds | Installed-wheel run, seconds |
| --- | --- | --- |
| Model loading and file verification | 66.0018 | 44.7736 |
| Four images encoded sequentially | 8.5789 | 8.0966 |
| One batch of four English texts | 2.4897 | 1.7883 |
| One batch of four Chinese texts | 2.5575 | 1.6245 |
| One API ranking request through TestClient | 2.5958 | 2.1674 |

Warm real-loopback request timings were:

| Query fixture | English, seconds | Chinese, seconds |
| --- | --- | --- |
| `red-circle` | 0.6390 | 0.6255 |
| `blue-square` | 0.4548 | 0.5750 |
| `green-triangle` | 0.6065 | 0.5117 |
| `black-star` | 0.4185 | 0.5431 |

Each loopback timing includes encoding, response validation, and the local
four-vector score/selection step. The observations support using a resident CPU
model for small debugging requests. Startup and repeated file verification remain
appreciably slower than warm requests.

These were individual functional runs, with the repository's parallel full suite
running during portions of the measurement. CPU contention and filesystem cache
state were not controlled. There were no repeated-load, concurrency, P95/P99,
peak-memory, throughput, or large-corpus measurements; these values are device
observations rather than a performance guarantee.

## Test commands and evidence

The focused run used the started model service:

```bash
CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT=http://127.0.0.1:8765 \
.venv/bin/pytest --no-cov -q \
  tests/unit/model \
  tests/unit/retrieval/test_siglip2_adapter.py \
  tests/unit/retrieval/test_service.py \
  tests/unit/api/test_siglip2_api.py \
  plugins/retriever-elasticsearch/tests/integration/test_local_model_images.py
```

Result: 118 passed in 8.86 seconds. The existing integration test validates the
real image encoder path; the eight semantic queries and wheel replay used
separate local smoke programs outside the repository test suite.

The evaluated source revision passed all four gates with `UV_NO_SYNC=1
UV_OFFLINE=1`: `make quality`, `make docs-check`, `make build`, and `make test`.
The full test run recorded 5,924 passed, 25 skipped, four warnings, 85.30%
coverage, and 1,057.50 seconds. Skipped ASR, native Windows, optional ONNX,
missing local media, and native performance cases remain outside this result.
These are the model-evaluation revision's gate results, not a substitute for
validation of subsequent changes.

Local evidence remains under `/tmp/capicove-siglip2-debug`: `source-report.json`,
`report.json` for the wheel replay, `http-report.json`, `asset-manifest.json`,
`trusted-model.json`, and `checks.json`, plus smoke programs under `/tmp`.
These artifacts may disappear after OS cleanup. The model receipt contains the
complete file inventory and hashes. Only this documentation summary is retained
in Git; weights, images, indexes, and raw local reports are excluded.

## Remaining qualification

The corpus is four simple authored images, not a held-out photographic/video
benchmark. This run does not qualify semantic recall on real media, ANN capacity,
Elasticsearch integration, shared workspace video Search, GPU performance,
native macOS/Windows execution, or general multilingual recall. The standalone
API and Python exact-index path were tested without changing public contracts
or enabling a new workspace provider configuration.
