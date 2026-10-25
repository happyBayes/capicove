# Image Model Installation

Capicove image inference is local-only. It never downloads a model while importing
a module, constructing a Tool, or evaluating an image. Model acquisition is a
separate, explicit operation that uses the official Hugging Face `hf` command.

For a complete SigLIP2 CPU setup, use the
[local retrieval guide](../site/guides/image/siglip2-local-retrieval.md). The
[2026-10-12 device evaluation](../site/development/siglip2-cpu-evaluation.md)
records pinned assets, runtime versions, authored checks, and CPU timings;
it does not qualify shared workspace video Search.

## Prerequisites

Install the model environment needed for the selected execution path, and install
the Hugging Face CLI separately. ONNX CPU inference does not install PyTorch;
the current Hugging Face pipeline adapters use the development environment:

```text
pip install 'capicove[models-cpu]'
pip install 'capicove[models-dev]'  # Export, quantization, and PyTorch adapters.
curl -LsSf https://hf.co/cli/install.sh | bash
```

NVIDIA GPU users install `capicove[models-gpu]`. Model developers combine
`models-dev` with exactly one runtime extra, for example
`capicove[models-cpu,models-dev]`.

Use `hf auth login` before downloading a private or gated repository. Capicove does
not install the CLI, authenticate a user, or enable remote model code.

## Inspect before downloading

Dry-run validates the packaged lock and prints the exact commands without requiring
`hf`, creating directories, or accessing a network:

```text
capicove-download-image-models --profile default --dry-run
capicove-download-image-models --model clip-vit-b32 --dry-run
```

Every command contains a 40-character immutable Hugging Face revision.

## Download the default profile

```text
capicove-download-image-models --profile default
```

The default profile installs the recommended NSFW, watermark, CLIP, BLIP ITM,
region-caption, and multi-image difference-caption models. It excludes the AGPL
aesthetics preset.

To download selected models instead:

```text
capicove-download-image-models --model nsfw-falconsai --model clip-vit-b32
```

The packaged aesthetics model is visible in dry-run output but requires explicit
license authorization:

```text
capicove-download-image-models \
  --model aesthetics-cafe \
  --include-restricted
```

`--include-restricted` means that the user chose to acquire the preset. It is not
legal advice and does not alter the model's AGPL-3.0 terms.

## Destination

The model root is resolved in this order:

1. `--destination PATH`;
2. `CAPICOVE_MODEL_HOME`;
3. the platform cache directory under `capicove/models`.

Examples:

```text
CAPICOVE_MODEL_HOME=/srv/capicove/models \
  capicove-download-image-models --profile default

capicove-download-image-models \
  --model clip-vit-b32 \
  --destination /approved/capicove/models
```

Each preset is stored under `<root>/<model-key>/`. A verified directory contains
`capicove-model-receipt.json`. The receipt records the repository, revision,
license, adapter, preprocessing revision, settings, timestamp, Capicove and `hf`
versions, ordered file hashes, and an aggregate manifest hash. A subsequent command
rehashes the directory before reporting it as already available.

Capicove rejects directories without receipts, mismatched revisions, modified
files, extra files, and symlinks. `--force` can replace only an existing receipt for
the same model key and repository; the new revision is downloaded and verified
before the old directory is replaced.

## Use another aesthetics model

A user manifest completely replaces the packaged catalog for that command. This
keeps model choice explicit and prevents an unreviewed custom model from being
merged into the default profile.

```toml
manifest_version = 1

[[models]]
key = "my-aesthetics"
repository = "my-organization/my-aesthetics-model"
revision = "0123456789abcdef0123456789abcdef01234567" # pragma: allowlist secret
license = "apache-2.0"
license_url = "https://huggingface.co/my-organization/my-aesthetics-model/blob/0123456789abcdef0123456789abcdef01234567/LICENSE"
capability = "image_aesthetics_filter"
adapter = "huggingface-aesthetics"
architecture = "ViTForImageClassification"
preprocess_revision = "my-aesthetics-preprocess-v1"
default = false
restricted = false

[models.settings]
target_label = "aesthetic"
```

Download it with:

```text
capicove-download-image-models \
  --manifest ./my-image-models.toml \
  --model my-aesthetics
```

Custom revisions must be lowercase 40-character Git SHAs. Unknown fields, adapters,
duplicate keys, and moving revisions such as `main` are rejected.

## Construct adapters from trusted installations

The download command prints `model_manifest_sha256` for every installed model.
Store that value in trusted configuration outside the writable model directory.
Reading the digest back from the receipt at runtime would be self-attestation and
does not establish trust in the installed bytes.

Bootstrap this trust anchor only from a controlled command result with
`status=installed`, after the pinned Hugging Face download and verification finish.
For `status=available`, use the digest already stored by the original installation;
do not promote the directory's receipt into trusted configuration.

Create a `VerifiedModelInstallation` from the packaged lock entry and the externally
stored file-manifest digest. Construction rehashes the complete directory. The
adapter verifies it again immediately before lazy model loading and derives its
repository, revision, preprocessing contract, and result lineage from this object:

```python
from pathlib import Path

from capicove.model_management import (
    load_packaged_manifest,
    verify_model_installation,
)
from capicove.processing.image._adapters.huggingface import (
    HuggingFaceClipImageTextSimilarityScorer,
    HuggingFaceImageClassificationScorer,
)

root = Path("/approved/capicove/models")
manifest = load_packaged_manifest()
models = {model.key: model for model in manifest.models}

# Load these values from read-only deployment configuration or a secret manager.
trusted_file_manifests = {
    "clip-vit-b32": "<64-character digest printed by the download command>",
    "nsfw-falconsai": "<64-character digest printed by the download command>",
    "blip-caption-base": "<64-character digest printed by the download command>",
    "smolvlm-256m-instruct": "<64-character digest printed by the download command>",
}


def installed(key: str):
    return verify_model_installation(
        root / key,
        models[key],
        manifest_sha256=manifest.source_sha256,
        expected_model_manifest_sha256=trusted_file_manifests[key],
    )


text_similarity = HuggingFaceClipImageTextSimilarityScorer(installed("clip-vit-b32"))
nsfw_scorer = HuggingFaceImageClassificationScorer(
    installed("nsfw-falconsai"),
    target_label="nsfw",
)
```

The aesthetics, watermark, BLIP matching, and image-pair scorers use the same
verified installation contract. The adapter name in the trusted lock entry must
match the scorer; an arbitrary local directory or independently supplied revision
is rejected.

Difference captions require two independently verified installations:

```python
from capicove.processing.image._adapters.huggingface_caption import (
    HuggingFaceDifferenceCaptionProvider,
)

caption_provider = HuggingFaceDifferenceCaptionProvider(
    installed("blip-caption-base"),
    installed("smolvlm-256m-instruct"),
)
```

The multi-image difference model must support the Transformers
`image-text-to-text` task. A single-image VQA model is not compatible.

## Agent instruction

Give an agent this instruction when you want it to acquire the default models:

```text
Run `capicove-download-image-models --profile default` after confirming that network
access and the destination are authorized. Do not install the Hugging Face CLI,
authenticate, add restricted models, or change the destination without user approval.
Report the installed model directories and locked revisions from their receipts.
```

An agent may run `--dry-run` without network access. A real download consumes
network bandwidth and local storage and therefore requires explicit authorization.

## Local SigLIP2 retrieval API

Install the combined HTTP and PyTorch runtime explicitly:

```text
uv sync --extra model-serving
```

Download the locked Google SigLIP2 model into an approved model root. The command
prints a `model_manifest_sha256`; store that digest in read-only deployment
configuration because the API requires it as the trusted model-file identity.

```text
capicove-download-image-models \
  --model siglip2-base-patch16-224 \
  --destination /approved/capicove/models
```

On Apple Silicon, `--device auto` selects MPS when PyTorch reports it available,
then falls back to CPU. The provider always loads from local files and disables
remote model code.

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --extra model-serving python scripts/serve_siglip2.py \
  --model-directory /approved/capicove/models/siglip2-base-patch16-224 \
  --model-manifest-sha256 <64-character-digest-from-download-output> \
  --device auto \
  --host 127.0.0.1 \
  --port 8000
```

The service exposes:

- `GET /health` for model revision and selected device;
- `POST /v1/similarity` with multipart fields `image` and a JSON-array `texts`;
- `POST /v1/embeddings/image` with multipart field `image`; and
- `POST /v1/embeddings/text` with JSON body `{"texts": [...]}`.

Example image-text ranking request:

```text
curl -X POST http://127.0.0.1:8000/v1/similarity \
  -F 'image=@example.jpg' \
  -F 'texts=["a photo of a cat","a photo of a dog"]'
```

Example text embedding request:

```text
curl -X POST http://127.0.0.1:8000/v1/embeddings/text \
  -H 'Content-Type: application/json' \
  -d '{"texts":["a photo of a cat","\u4e00\u5f20\u732b\u7684\u7167\u7247"]}'
```

Returned image and text vectors are L2-normalized. The similarity response includes
SigLIP2's independent sigmoid match probability and a softmax probability across
the submitted candidate set. Use the latter only when candidates are mutually
exclusive.

For a token stored outside the standard `hf auth` configuration, pass only its
environment-variable name:

```text
capicove-download-image-models \
  --model clip-vit-b32 \
  --token-env PRIVATE_HF_TOKEN
```

The token value is copied to `HF_TOKEN` only in the `hf` subprocess environment. It
is never added to command arguments or receipts.
