# Register and serve vision language models

The `openai-vlm` provider calls an existing OpenAI-compatible Chat Completions
endpoint. It works with operator-deployed vLLM models without adding a provider
for each model family. Capicove provides a Python port, a JSON CLI and an HTTP
service. Client inference requires `capicove[vlm]`; HTTP serving
requires `capicove[vlm,api]`; Torch, Transformers and model weights belong to the
upstream deployment.

## Register a deployed model

Save this as `models.toml`. Set `model` to the exact served model name reported by
the upstream `/v1/models`. Adjust modality and count limits to that deployment.
The profile name `qwen35-9b` becomes the model alias exposed by Capicove.

```toml
schema_version = 2
max_instances = 1

[profiles.qwen35-9b]
transport = "http"
provider = "openai-vlm"
capability = "multimodal_generation"
endpoint = "http://127.0.0.1:8000/v1"
model = "Qwen/Qwen3.5-9B"
allow_network = true
# Environment variable reference, not a credential.
api_key_env = "CAPICOVE_QWEN_API_KEY" # pragma: allowlist secret
modalities = ["image", "video"]
max_images = 8
max_videos = 1
max_output_tokens = 4096
timeout_seconds = 120
token_counting = "vllm"

[profiles.qwen35-9b.chat_template_kwargs]
enable_thinking = false

[bindings]
"assistant.vision" = "qwen35-9b"
```

Set the referenced environment variable through your secret manager or an
interactive prompt. Omit `api_key_env` for an upstream without authentication.
Keys are never read from configuration literals or passed as command arguments.
HTTP profiles require `allow_network = true`, including loopback connections.
Remote endpoints require HTTPS; loopback HTTP is supported. Proxy environment
variables, redirects and automatic retries are disabled.

To register another model, add another profile using `provider = "openai-vlm"`,
its endpoint, served model name, supported modalities and limits; add a binding.
No Python changes are needed. Each HTTP profile has its own template options.
Only set `enable_thinking` when supported by that model's chat template. Existing
version-1 local configurations remain valid; version 2 can also contain unchanged
local profiles with their verified roots and manifests.

```bash
capicove models vlm list --config /absolute/models.toml
```

## Python inference

```python
from pathlib import Path

from capicove.model import ModelFactory, load_configuration
from capicove.model.vlm import VlmMedia, VlmMessage, VlmRequest, VlmText

request = VlmRequest(
    messages=(
        VlmMessage(
            "user",
            (
                VlmText("Describe this image."),
                VlmMedia(Path("picture.png").read_bytes(), "image/png"),
            ),
        ),
    ),
    max_tokens=256,
)
with ModelFactory(load_configuration(Path("/absolute/models.toml"))) as factory:
    with factory.vlm("assistant.vision") as model:
        result = model.generate(request)
        print(result.text, result.finish_reason, result.usage)
```

Other Capicove capabilities can consume the public `VlmProvider` port and inject
`factory.vlm(binding)`. Keep borrowed providers inside their lease. Registration
does not automatically replace existing OCR, captioning or graph adapters.

## JSON CLI and HTTP

Create a portable request with embedded image bytes:

```bash
python - <<'PY'
import base64
import json
from pathlib import Path

image = base64.b64encode(Path("picture.png").read_bytes()).decode("ascii")
request = {
    "model": "qwen35-9b",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + image,
        }},
    ]}],
    "max_tokens": 256,
}
Path("request.json").write_text(json.dumps(request))
PY

capicove models vlm generate --config /absolute/models.toml \
  --binding assistant.vision --input request.json

capicove models vlm serve --config /absolute/models.toml \
  --host 127.0.0.1 --port 8769
```

All bound VLM profiles are listed by the listener:

```bash
curl -sS http://127.0.0.1:8769/v1/models
curl -sS http://127.0.0.1:8769/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
```

An OpenAI SDK client can use `base_url="http://127.0.0.1:8769/v1"` and the profile
alias as `model`. The unambiguous upstream served name is also accepted. If several
profiles share one upstream name, use their distinct profile aliases. To expose
the listener beyond loopback, set `--api-key-env CAPICOVE_VLM_HTTP_KEY`. The incoming
service key and upstream key are independent.

## Supported input and response behavior

- Text, single/multiple images, and multi-turn text/image conversations.
- PNG, JPEG, WebP and GIF images, with base64 data URLs and optional `detail`
  values of `auto`, `low` or `high`. `VlmMedia` also accepts
  MP4, QuickTime and WebM video bytes. Video uses vLLM's `video_url` extension,
  enabled only when the profile declares `video`. Actual decoding and model
  modality support belong to the upstream deployment.
- Standard roles `system`, `developer`, `user`, `assistant`, and `tool`; media is user content.
  Text-only calls are useful for health qualification and follow-up questions.
- Sampling options `temperature`, `top_p`, `seed`, up to four `stop` sequences,
  `max_tokens` or `max_completion_tokens`, and `response_format` of `text` or
  `json_object` when supported upstream.
- One nonstreaming choice. Function tools support `tools`, `tool_choice`,
  `parallel_tool_calls`, assistant `tool_calls`, and `tool_call_id` results.
  Tool execution belongs to the caller. Streaming, arbitrary upstream body overrides,
  filesystem URLs and remote media URLs are rejected explicitly. Unsupported
  fields return 400. Template options are operator configuration.
- Raw requests are limited to 32 MiB, each decoded media item to 20 MiB, and
  upstream responses to 2 MiB. There are at most 64 messages and 64 parts per
  message. Model profiles further bound output tokens and media counts. Context
  capacity and decoded media resources are enforced by the deployed model;
  Capicove does not truncate inputs or approximate multimodal token counts.

`usage` contains upstream-reported prompt/completion counts, or `null` when absent.
`finish_reason="length"` preserves truncation. Reasoning, when returned, is exposed
as `reasoning_content`. Valid empty or whitespace-only output retains its finish
reason and usage; a null text field becomes an empty string. A custom `stop` can
end output immediately or interrupt a JSON object. Complete JSON validation is
applied to `json_object` responses ending in `stop` only when no custom stop
sequences were requested. Callers using custom stops must handle partial JSON.
The additive `capicove` object records the effective
profile identity, exact outbound request SHA-256 and ordered media SHA-256 values.
Optional `model_revision` is an operator declaration, not server attestation.
Changing assets on the upstream server requires updating that declaration.

The listener admits one request at a time and returns 429 while busy. Parsing,
media encoding, generation and response serialization run off the event loop so
health remains responsive. `/health` reports listener readiness and explicitly says
`upstream_checked=false`; use an actual generation request to qualify the model.
Upstream authentication, redirects, malformed responses and wrong model IDs fail
with sanitized errors. Timeouts return 504 and do not retry automatically. Closing
a socket cannot guarantee cancellation of generation already running upstream.
A total deadline covers DNS, TCP, TLS, headers and response reads. The optional
aiohttp/c-ares client owns connection and DNS cancellation; Capicove does not
maintain custom resolver threads or socket subclasses. TLS retains certificate
and hostname verification. Incomplete HTTP bodies, compressed responses and
invalid Unicode are rejected at the upstream boundary. Synchronous Python calls
must run in a worker when invoked from an async application.

## Token counts and agent tool conversations

`POST /v1/count_tokens` accepts the same model and message body as generation.
With `token_counting = "vllm"`, it calls the upstream `/tokenize` API, including
images, videos, tools and configured chat-template settings. It returns
`input_tokens`, `context_limit`, `source`, `request_sha256` and `profile_identity`.
This endpoint is a Capicove extension. An unconfigured or unavailable backend
returns 501; no approximate fallback is used. Compare counts with generation
usage when qualifying another model or server configuration.

```bash
capicove models vlm count-tokens --config /absolute/models.toml \
  --binding assistant.vision --input request.json
curl -sS http://127.0.0.1:8769/v1/count_tokens \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Python callers use `provider.count_tokens(request)`. Define function tools with
`VlmTool` and select a specific function with `VlmToolChoice(name)`, or parse
standard OpenAI JSON using `parse_chat_request`. `VlmResult.tool_calls` preserves
call IDs, function names and arguments. A completed tool proposal uses
`finish_reason="tool_calls"`; a truncated proposal remains `length` and must not
be executed. Some vLLM named-function responses use `stop` with complete tool
calls; these are normalized to `tool_calls`, retaining the upstream value in
`capicove.upstream_finish_reason`. The host validates and executes approved functions, appends the
assistant message and one `tool` result per call, and sends the updated history.
Duplicate IDs, unmatched results and missing results fail before network access.
Assistant `reasoning_content` is preserved for multi-turn reasoning models.
Empty tool arrays are omitted so deployments do not need an empty-tools proxy.

## Reusable deployment and examples

The source distribution includes `scripts/vlm/deploy.py`, `qwen35-9b.toml`,
`models.toml`, `client.py`, and a README with commands. Acquisition pins a model
commit and verifies upstream checksums; startup verifies the receipt and forces
offline model loading. The Qwen recipe records the locally qualified development
build of vLLM; supplying that trusted runtime is a separate installation step.

`client.py` supports single images, repeated images, video, preflight counts and a
bounded example tool loop. Its `request` mode emits JSON suitable for curl. Use
`capicove models vlm serve` for the Capicove HTTP service and `deploy.py serve`
for the GPU runtime. To register another deployed model, copy the profile and
change its endpoint, served name, modalities, limits and template options.

Protocol references: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [vLLM multimodal input](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

See [registered model services](model-services.md) for the shared HTTP listener,
ASR uploads, and knowledge generation through a registered VLM.
