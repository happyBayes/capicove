# Registered model HTTP services

Serve explicitly installed models through one entry point:

```bash
capicove models list --config /absolute/path/models.toml
capicove models serve --config /absolute/path/models.toml --model speech
capicove models serve --config /absolute/path/models.toml --model qwen35-9b
capicove models serve --config /absolute/path/models.toml --all --port 8769
```

Repeat `--model` to select several instances. `--all` selects all bound profiles.
Set `max_instances` to at least the number selected. Startup verifies local assets
and rolls back acquired resources if any provider fails. Models remain resident
until shutdown, although Whisper.cpp launches a bounded subprocess per inference.
This command does not download weights or install native runtimes.

The default listener is `127.0.0.1:8769`. For a non-loopback listener, pass
`--api-key-env CAPICOVE_MODEL_HTTP_KEY` and set that variable through your normal
secret-management mechanism. Use TLS termination for remote access. Incoming
service authentication and upstream VLM credentials are separate.

`GET /v1/models` lists registered aliases, capabilities and endpoints. Requests
select the registered alias, not a filesystem path. Each model permits one active
request; another request for it receives 429. At most four requests may be in the
shared upload/dispatch stage. Health and discovery do not perform inference.

| Capability | Endpoint | Model selection |
| --- | --- | --- |
| Text embedding | `/v1/embeddings` | JSON `model` |
| Reranking | `/v1/rerank` | JSON `model` |
| Audio transcription | `/v1/audio/transcriptions` | Multipart `model` |
| VLM/text generation | `/v1/chat/completions` | JSON `model` |
| Exact token counting | `/v1/count_tokens` | JSON `model` |
| SigLIP2 text embedding | `/v1/embeddings/text` | JSON `model` |
| SigLIP2 image/similarity | Existing image routes | Query `model` |
| OCR/document parsing | `/v1/ocr`, `/v1/documents/parse` | Query `model` |

For existing raw image/PDF routes, the selector is optional when exactly one
matching model is served. Existing upload formats and PDF page-selection query
parameters still apply. The dispatcher removes its model selector before passing
requests to these capability applications. Legacy serving commands remain
available; use the configured entry point for a shared authenticated listener.

## Register Whisper.cpp

Use already provisioned Whisper.cpp and Silero files. Keep weights outside the
source checkout. Compute each file's SHA-256 and the manifest digest:

```python
import hashlib
import json
from pathlib import Path

root = Path("/absolute/path/whisper-models")
files = []
for name in ("ggml-tiny.bin", "ggml-silero-v6.2.0.bin"):
    with (root / name).open("rb") as stream:
        files.append((name, hashlib.file_digest(stream, "sha256").hexdigest()))
print(dict(files))
print(
    hashlib.sha256(
        json.dumps(sorted(files), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
)
```

Place these hashes in the following profile. Paths are operator configuration;
clients cannot replace model files or select executables.

```toml
schema_version = 2
max_instances = 1

[bindings]
speech = "speech"

[profiles.speech]
provider = "whisper-cpp"
capability = "audio_transcription"
root = "/absolute/path/whisper-models"
manifest_sha256 = "REPLACE_WITH_MANIFEST_SHA256"
device = "cpu"

[profiles.speech.options]
model = "ggml-tiny.bin"
model_sha256 = "REPLACE_WITH_MODEL_SHA256"
vad_model = "ggml-silero-v6.2.0.bin"
vad_model_sha256 = "REPLACE_WITH_VAD_SHA256"
whisper_cli = "/absolute/path/whisper-cli"
whisper_vad_cli = "/absolute/path/whisper-vad-speech-segments"
ffmpeg = "/absolute/path/ffmpeg"
ffprobe = "/absolute/path/ffprobe"
backend_version = "whisper.cpp-v1.9.5"
max_duration_seconds = 1800
```

The decoded duration limit accepts 1–7200 seconds, default 1800. Normalization
stops after the configured limit plus one second and rejects overlong audio;
it never returns an apparently complete truncated transcript. The upload limit
is 25 MiB, intake timeout 30 seconds, and processing deadline 300 seconds.
Use a compatible FFmpeg build and verify it with your installed Whisper runtime.

```bash
curl http://127.0.0.1:8769/v1/audio/transcriptions \
  -F model=speech \
  -F file=@sample.wav \
  -F language=en \
  -F response_format=verbose_json
```

`language` defaults to `auto`. Supported response formats are `json` (text),
`text` (plain text), and `verbose_json` (language, duration, segments and additive
hash/lineage fields). Multiple audio streams, unknown parameters, duplicate fields,
multiple files and invalid content are rejected. No prompt, streaming, word-level
timestamps, URL download or audio translation is implied by this endpoint.

## Reference a registered VLM from knowledge generation

Register the upstream service as described in [VLM](vlm.md), with
`token_counting = "vllm"` and an explicit `model_revision`. The model must support
JSON-object responses. Set a declared revision even for a local vLLM deployment;
Capicove cannot attest remote GPU weights through Chat Completions.

Use this generation configuration with the existing knowledge `--generation`
option or knowledge configuration field:

```json
{
  "schema_version": 2,
  "models": "/absolute/path/models.toml",
  "generation_model": "qwen35-9b",
  "context_limit": 65536
}
```

The endpoint, credential reference and chat template are read from the selected
model profile. Knowledge host network opt-in and destination authorization still
apply. Text graph requests use the same VLM provider directly, without a request
through Capicove's local HTTP listener. Schema validation, evidence access and
receipt storage remain in the graph workflow.

Complete request counts include the exact messages and template. Chunk packing
uses upstream chat counts including framing, so packing is conservative. There is
no approximate fallback when `/tokenize` is unavailable. Input usage disagreement,
missing usage, tool calls or oversized output are rejected. Model/configuration
identity changes invalidate old receipts. Schema-1 generation configurations
remain supported for explicit migration; local Transformers generation is not
silently replaced.
