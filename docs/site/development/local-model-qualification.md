# Local model qualification and repair

This procedure exercises the actual Hugging Face adapters and standalone video
CLI with existing, verified assets. It never downloads models or substitutes
inference functions. General regression tests and real-model qualification are
separate gates. Missing model configuration is a skip, not qualification success.

## Supported entry points

- Bare `capicove image run` composes rule tools. Model-backed image tools still
  require explicit Python/MCP provider composition. Installing weights alone does
  not enable them in the bare CLI.
- Standalone `capicove video filter` supports the local model criteria below.
  Its input directories and criteria-file directory are authorized read roots.
  Keep criteria in a dedicated parent of the model directories; do not broaden
  host permissions or modify a verified model directory to accommodate it.
- The existing model factory supports BGE, SigLIP2 and Paddle profiles. Video host
  mode retains its model-filter restrictions. No new factory or remote gateway
  is required for these repairs.
- `capicove doctor` reports installation metadata, not successful model inference.
  Use `setup doctor`, capability-specific diagnostics, and real-model tests for
  the configured workload.

## Scoring and compatibility

The repository lock selects Transformers 5.15.0. Qualification of that environment
does not establish that every version admitted by optional dependency ranges is
supported. Tensor and structured `pooler_output` feature returns are covered by
regression tests; a full historical SDK matrix is not claimed.

Image classifiers and caption pipelines load model/processor objects explicitly
with `local_files_only=True` and `trust_remote_code=False`. CLIP image-pair scoring
uses the projected image features and cosine similarity. Its preprocessing identity
includes `clip-image-tower-cosine-v1`. Captioning uses an empty BLIP prefix and
embedded chat images; its identity includes `unconditional-embedded-chat-v1`.
Existing asset receipts are unchanged. Recompute derived scores/captions when
comparing against the old preprocessing behavior.

Video aesthetics accepts either one regression logit per frame (preserving its
existing raw scale), or a classifier with exactly one case-insensitive `aesthetic`
label. Classification returns the softmax probability of that named class, with
the selected logit retained as `raw_value` and its index in result details.
Ambiguous labels, unequal output rows and non-finite scores fail closed. Do not
reuse a regression threshold for a probability without calibration.

Watermark's default image threshold is **0.8**, not 0.5. The earlier evaluation
used the default while describing it as 0.5. Small authored samples establish
integration behavior, not general detection accuracy. A nonempty difference
caption establishes inference completion, not that the described change is true.

## Repeat the real-model checks

Provision assets separately with exact revision/digest verification and the
required license authorization. Use an operator-owned JSON configuration:

```json
{
  "model_root": "/absolute/qualification/models",
  "criteria_root": "/absolute/qualification",
  "data_root": "/absolute/evaluation/data",
  "device": "cpu",
  "digests": {
    "nsfw-falconsai": "REPLACE_WITH_TRUSTED_FILE_MANIFEST_SHA256"
  }
}
```

The digests map must include `nsfw-falconsai`, `watermark-siglip2`, `clip-vit-b32`,
`blip-caption-base`, `smolvlm-256m-instruct`, `aesthetics-cafe`, and `blip-itm-coco`.
Place the writable criteria root above the verified model directories. Tests
create uniquely named temporary criteria there and remove them after CLI use.
The fixture directory is the authored full-function evaluation dataset: its
`manifest.json` contains `photos` with `key`/`path` entries including `cat` and
`coffee`; `video/` contains `testsrc_10s.mp4`, `text_overlay_4s.mp4`, and
`static_6s.mp4`. Keep this data local. Absent assets, bad digests, or absent fixture
files fail when qualification is explicitly selected.

```bash
umask 022
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  CAPICOVE_MODEL_QUALIFICATION=/absolute/qualification.json \
  uv run --no-sync pytest --no-cov -v tests/model/test_local_huggingface.py
```

Install the appropriate optional SDK dependencies before this command. `--no-sync`
preserves the explicitly prepared local environment, including optional GPU SDKs.
Use `device: cuda` only in an independently qualified GPU environment. The tests
exercise image APIs and five standalone video CLI cases; they do not claim that
bare image CLI or every transport exposes these model capabilities.

## Host-specific environment findings

On the 2026-10-19 Linux host, system FFmpeg 4.4 lacks the required `fd:` protocol.
The previously provisioned FFmpeg/ffprobe 7.0.2 binaries were copied into `.venv/bin`
without replacing system binaries. Activate that environment, use `uv run`, or
pass explicit executable paths. Check `capicove audio capabilities --component clip`.

Torch 2.13.0+cu130 loads cuDNN 9.20 from its wheel but can discover the system's
9.23 optional tensor-IR library. Clearing `LD_LIBRARY_PATH`, or pointing only at
the wheel directory, did not fix this host. A real CUDA convolution succeeded
with a consistent system 9.23 library set using:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  uv run --no-sync python -c \
  'import torch; print(torch.nn.Conv2d(3,8,3).cuda()(torch.zeros((1,3,32,32),device="cuda")).shape)'
```

These paths are host-specific, not application defaults. Apply them only to the
selected local GPU command. The application does not globally disable cuDNN or
change library search paths. CPU BLIP/SmolVLM qualification was interrupted after
305 seconds, so that attempt is not evidence of successful CPU caption completion.

## Adversarial review

Regression checks cover channel order and contiguous Torch input, structured and
Tensor CLIP outputs, NaN/zero/malformed embeddings, classification ambiguity and
non-finite logits, invalid caption payloads, prompt-only chat output, tampered
model receipts, and retry after a failed post-load CLIP verification. Model
handles are published only after that verification succeeds. No automatic
network download, trust-remote-code opt-in, host permission widening, model-weight
modification, or new public contract was introduced.

The original batch evaluation selected a nonexistent `data/images` directory and
accepted zero inputs. The corrected local evaluation requires twelve actual,
content-distinct images and verifies twelve cache hits on resume. Identical
content can legitimately produce cache hits during the first run; that is not a
batch executor defect.

## Verification results on this host (2026-10-19)

| Check | Result |
| --- | --- |
| `make quality` | Ruff and mypy passed |
| `make test` | 6,031 passed, 35 skipped; 85.32% coverage |
| `make build` | Wheel and source distribution built |
| Real HF qualification | 10 passed in 125.69 seconds; image inference on GPU, standalone video CLI on CPU |
| Native video model criteria | NSFW, watermark, aesthetics and both CLIP cases passed without inference patches |
| Wiki / workspace retrieval / image rules | 7/7, 6/6, 15/15 |
| BGE-M3 | Authored retrieval top-1/MRR 1.0; wrong digest and overlength input rejected |
| PaddleOCR / video OCR | 5/5 image cases, mean CER 0; 2/2 video cases |
| Book pipeline | 3 chapter submissions; published successfully; 9/9 original lines recoverable |
| SigLIP2 loopback HTTP | Ranking top-1/MRR 1.0; normalized embeddings; empty request returned 422 |
| Batch with real inputs | 12 images processed and retained; 12/12 cache hits on resume |
| Memory / MCP / HTTP / trace export | 11/11, 11/11, 14/14, 4/4 |

The full regression skips include platform-specific cases and the explicitly
configured real-model suite, which was run separately above. The 35 skips are
not counted as passing tests. Book summary submissions use deterministic authored
content; no external generative summarizer is qualified by that journey. Watermark
and aesthetics quality, semantic difference-caption accuracy, other operating
systems, and unprovisioned optional backends remain outside these claims.

Temporary raw evidence is under `/tmp/capicove-requalification/work/results`.
Gate logs are `/tmp/capicove-{quality-final,test-final,build-final,docs}.log`;
real-model evidence is `/tmp/capicove-real-models-qualified.log`. These temporary
paths are local debugging artifacts, not durable release attestations. The
original external evaluation report and its fixtures were not edited.
