# Reusable vLLM deployment and VLM examples

This directory provides explicit model acquisition, offline startup, registration,
and image/video/tool clients. The provider supports other OpenAI-compatible VLMs;
copy the configuration and change the model identity and runtime options.

## Runtime

Use a dedicated Python 3.12 vLLM environment. The Qwen recipe records the locally
qualified runtime `0.32.0rc1.dev41+g426f3bd36` and upstream model commit
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. This is a development build, so it is not
assumed to be available from PyPI. Install the corresponding trusted wheel and its
platform dependencies in that environment, or qualify a different vLLM version
and update `vllm_version` in a copied recipe. The launcher checks that version.
It does not install or upgrade CUDA, Torch, or vLLM during serving.

The example uses BF16 and a 32k context. Adjust GPU memory utilization, context
length, concurrency and media limits for the selected GPU. Qwen-specific reasoning
and tool parsers belong in this recipe. The conservative Qwen settings disable
prefix caching and async scheduling, matching the locally qualified deployment.
The provider itself contains no Qwen-specific model name or parser.

## Acquire and verify weights

Install the Hugging Face CLI in the acquisition environment. Downloading is a
separate, explicit network operation. The output directory must be outside the
Capicove source repository.

```bash
python scripts/vlm/deploy.py download \
  --root /absolute/model-storage/qwen35-9b --allow-network
python scripts/vlm/deploy.py verify --root /absolute/model-storage/qwen35-9b
```

Acquisition uses `hf download` at the pinned commit and `hf cache verify` against
the upstream inventory, then writes `capicove-assets.json` with SHA-256 hashes,
repository, revision, license, format, conversion provenance and runtime version.
A repeated download verifies an existing matching receipt. A changed revision
requires another directory. Serving verifies all hashes and rejects extra files.
Keep the receipt with the weights and protect both from unauthorized changes.

## Start vLLM and Capicove

Set `CAPICOVE_VLM_API_KEY` using a secret manager or an interactive prompt. In the
vLLM environment:

```bash
read -rsp 'VLM API key: ' CAPICOVE_VLM_API_KEY; export CAPICOVE_VLM_API_KEY
python scripts/vlm/deploy.py serve --root /absolute/model-storage/qwen35-9b
```

`--dry-run` verifies assets and prints the exact launch arguments without starting
a process. Credentials are passed through the vLLM environment, never process
arguments. Serving forces Hugging Face/Transformers offline mode and disables
usage telemetry. The process runs in the foreground; a service supervisor can
own its lifecycle. Its readiness endpoint is `http://127.0.0.1:8001/health`.

In the Capicove environment, install `capicove[vlm,api]`, set the same upstream key,
and start the HTTP adapter:

```bash
capicove models vlm list --config scripts/vlm/models.toml
capicove models vlm serve --config scripts/vlm/models.toml --port 8769
```

For an existing vLLM deployment, edit `endpoint` and `model` in a copied
`models.toml`; `model` must match `/v1/models`. No model download or vLLM restart
is needed. `token_counting = "vllm"` enables that server's `/tokenize` extension.
For another compatible server, omit it unless that extension is supported.

## Single image, multiple images, video, counts and tools

Run clients from an environment with `capicove[vlm]` installed:

```bash
python scripts/vlm/client.py generate --image picture.png --prompt 'Describe it.'
python scripts/vlm/client.py generate --image first.png --image second.png --prompt 'Compare these images.'
python scripts/vlm/client.py generate --video clip.mp4 --prompt 'Describe the sequence.'
python scripts/vlm/client.py count --image picture.png
python scripts/vlm/client.py tools
```

The example tool loop only executes the allowlisted `multiply` function, validates
its integer arguments, and limits the conversation to four model turns. Provider
responses never execute tools themselves. Applications should apply their own
schema validation, authorization, deadlines and side-effect policies.

Generate a portable JSON body and call with curl:

```bash
python scripts/vlm/client.py request --image picture.png > request.json
curl --fail-with-body -sS http://127.0.0.1:8769/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
curl --fail-with-body -sS http://127.0.0.1:8769/v1/count_tokens \
  -H 'Content-Type: application/json' --data-binary @request.json
capicove models vlm count-tokens --config scripts/vlm/models.toml \
  --binding assistant.vision --input request.json
```

Use repeated `--image` arguments or `--video` with `request` to produce the other
payloads. Media is embedded as base64; file paths are read only by the client.
There is no automatic video transcoding or remote URL fetching. Resize or shorten
large media explicitly before submitting it. The public service accepts JSON;
it does not expose multipart uploads or streaming responses.

Token counting uses the upstream chat template, registered template options,
tool definitions, and multimodal preprocessing. It is a preflight result; model
or template changes on the server can invalidate it. Qualify a new deployment
by comparing `input_tokens` with generation `usage.prompt_tokens`, including
images, video, tool schemas and tool-result history. Backend-specific changes
such as excluding schemas for `tool_choice=none` must match the tokenizer path;
Capicove does not infer those server settings or fall back to text estimates.
