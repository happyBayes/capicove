# Deployment and delivery

Use `scripts/deploy_release.py` from the reviewed checkout to prepare a selected
dependency bundle, then use the copy included in that bundle to install on the
target machine. Preparation is the explicit network step; installation uses only
hash-locked local wheels. Weights, private media and credentials are excluded.

## Prepare the bundle

Build the six distributions using the [release procedure](../development/releasing.md).
On a machine with Python and `uv`, select the profiles you need:

```bash
python scripts/deploy_release.py prepare \
  --dist dist/release-0.1.0 \
  --output /srv/capicove-bundle \
  --profile core --profile interfaces --profile knowledge --profile media --profile audio-asr \
  --profile graph-generation --profile reverse-image \
  --device cpu --allow-network
```

The output parent must exist; the destination must be new. The command resolves
dependencies, downloads wheels only, and produces `requirements.lock`,
`wheelhouse/`, `bundle.json` and the installer. It prints the SHA-256 of
`bundle.json`; retain that digest through a trusted channel. The bundle is for
the preparation machine's Python minor version, OS and wheel platform. Prepare
a separate bundle for another target. `cu130` is an explicit alternative to CPU;
device failure never causes fallback.

Additional profile names are `knowledge`, `semantic-text`, `semantic-vision`, `local-generation`,
`background-removal`, `video-ocr`, `video-tags`, `ray`, `ocr-book`, `elasticsearch`,
`opensearch`, `faiss` and `spark`. Profiles select dependencies, not model weights
or a running service. `video-tags` also needs the separately provisioned RAM++
runtime; `spark` requires Java 17. During explicitly authorized preparation, the
builder converts the resolved PySpark source archive into a wheel because upstream
does not publish that wheel. Other dependencies remain binary-only, and target
installation still uses only the sealed wheelhouse. Tesseract is a separate
native executable with explicitly provisioned language data.

## Install and qualify

Use the reviewed digest, an existing parent directory and a new destination:

```bash
python /srv/capicove-bundle/deploy_release.py install \
  --bundle /srv/capicove-bundle --sha256 REVIEWED_BUNDLE_SHA256 \
  --destination /srv/capicove-runtime \
  --native-config /srv/capicove-assets/native.json \
  --llm-config /srv/capicove-assets/generation.json
```

`--native-config` and `--llm-config` are optional explicit selections. The latter
authorizes the authored remote readiness probe after offline package installation.
Supply its credential through the configured environment variable. Never put a
secret in command arguments or the JSON configuration.

Installation first copies and re-verifies the bundle inside the private destination,
then creates an isolated `venv`, a private workspace and authored example
inputs. It checks package consistency, imports a document, builds the SQLite
index and retrieves the document through the installed CLI. Repeat
`--source-root ALIAS=/absolute/directory` to authorize additional source roots;
these are not ingested automatically.

When native media configuration is supplied, the installer verifies every
selected executable/model hash before enabling Tools, generates the existing
audio/video host configuration, and runs actual transcription and video splitting.
When generation configuration is supplied, it calls the selected model and checks
JSON output and exact token accounting. `delivery.json` distinguishes these
results from optional profiles still requiring runtime configuration. A failed
step leaves a `failed` report for diagnosis; it does not declare readiness or
overwrite an existing deployment. Retry into a new directory after correcting
the cause.

Native configuration uses the already supported explicit file references:

```json
{
  "files": {
    "ffmpeg": {"path": "/absolute/ffmpeg", "sha256": "REVIEWED_SHA256"},
    "ffprobe": {"path": "/absolute/ffprobe", "sha256": "REVIEWED_SHA256"},
    "whisper_cli": {"path": "/absolute/whisper-cli", "sha256": "REVIEWED_SHA256"},
    "whisper_vad_cli": {"path": "/absolute/whisper-vad-speech-segments", "sha256": "REVIEWED_SHA256"},
    "model": {"path": "/absolute/ggml-tiny.bin", "sha256": "REVIEWED_SHA256"},
    "vad_model": {"path": "/absolute/ggml-silero-v6.2.0.bin", "sha256": "REVIEWED_SHA256"}
  }
}
```

Replace placeholders with independently reviewed hashes. Preserve the native
executables' shared libraries. An optional hash-pinned `speech` entry selects an
authorized acceptance fixture; otherwise the installer authors silence. This
does not download anything or establish speech-quality accuracy.

## User-hosted compatible LLM

`capicove[graph-remote]` needs a locally provisioned tokenizer, not local LLM
weights. The server uses the OpenAI-compatible `/chat/completions` format; no
vendor SDK or fixed model name is required. The endpoint includes the API base
path, commonly `/v1`.

```json
{
  "schema_version": 1,
  "endpoint": "https://inference.example.org/v1",
  "model": "operator-selected-served-model",
  "model_revision": "operator-attested-immutable-deployment-revision",
  "network_enabled": true,
  "api_key_env":
    "CAPICOVE_LLM_API_KEY",
  "context_limit": 24576,
  "tokenizer": {
    "root": "/absolute/verified-tokenizer",
    "files": {"tokenizer.json": "REVIEWED_FILE_SHA256", "tokenizer_config.json": "REVIEWED_FILE_SHA256", "chat_template.jinja": "REVIEWED_FILE_SHA256"},
    "manifest_sha256": "REVIEWED_INVENTORY_SHA256"
  },
  "template_options": {}
}
```

`files` is the exact regular-file inventory required by the selected tokenizer,
not a universal list of three mandatory files. Use `LocalModelAssets`' existing
manifest format: SHA-256 of the sorted file/hash pairs serialized with
`ensure_ascii=False` and `separators=(",", ":")`. Symlinked Hub cache entries
must be explicitly copied into a verified closed directory. The server and local
tokenizer must use the same chat template and options. A server supporting an
`enable_thinking` template option can select it explicitly; it is not a default
or a requirement for other models.

```bash
capicove setup check-llm --config /absolute/generation.json
```

This command sends only an authored readiness request. It verifies exact local
input-token counts against server usage, selected model identity, bounded output
and successful JSON generation. HTTP 401/403, redirects, truncated/malformed
responses, missing credentials, tokenizer drift and expired deadlines fail
closed. Non-loopback HTTP and URL-embedded credentials are rejected. Remote
model revision is an operator attestation because this API does not verify weight
bytes. Cancellation stops result acceptance immediately; a bounded transport
worker can remain alive until its socket operation exits.

For graph integration, inject `OpenAICompatibleGenerator(profile)` and
`OpenAICompatibleTokenizer(profile)` into the existing `InferenceRunner`. Select
the same identities in `InferencePolicy`. The tokenizer also implements the graph
chunker's `count` and `fingerprint` methods. Authorize evidence egress in the host
before invocation. Existing workspaces stay offline; this configuration does not
expand their network policy. See [document graph composition](document-graph-pipeline.md).

## Operate and release

Use `venv/bin/capicove setup doctor --config workspace/host.json` for startup
configuration checks. Use the generated host file with CLI, HTTP or MCP according
to the [workspace media guide](workspace-media.md). Binding an HTTP server to a
public interface, managing its authentication/TLS, starting external search
services and provisioning model assets remain explicit operator actions.

Retain `bundle.json`, its trusted digest, `delivery.json`, model receipts, service
versions and the real acceptance report together. Rebuild a new bundle and a new
environment for upgrades; preserve evidence and review index compatibility before
activation. This workflow neither commits, tags nor publishes. Source-history
privacy review and registry/repository authorization remain separate release gates.

## Native knowledge workspace

After installing `semantic-text`, `graph-generation`, `ocr-book` and `interfaces`,
use [`capicove kb init`](native-knowledge.md) to connect the existing verified model
factory, compatible generation profile and native audio inventory. It writes one
knowledge configuration and a query-only HTTP/MCP host configuration. Weights and
credentials remain explicitly provisioned; they are not embedded in the bundle.

## Native knowledge delivery

Select `--profile knowledge` to include the native knowledge dependencies and
HTTP/MCP extras. Offline installation initializes a separate `knowledge/` workspace
and verifies import, citation resolution, replacement, restart and revocation using
authored text. The receipt explicitly records lexical acceptance; it does not claim
model inference. Configure verified assets using the [native knowledge guide](native-knowledge.md)
to enable dense retrieval, OCR, speech and graph generation. The profile currently
requires Linux. Add `ocr-book` for the Paddle plugin and provision PaddlePaddle
explicitly for OCR; `audio-asr` and `media` describe separately pinned native tools.
