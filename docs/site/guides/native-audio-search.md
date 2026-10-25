# Native audio search (preview)

Search recordings by sound, not only by transcript. The `audio-local-v1` profile
cuts each imported audio stream into exact 30-second windows (25-second hop),
embeds the waveform directly with one shared text/audio model, and returns
timestamped audio segments with verifiable evidence. No ASR or speech detection
runs, so music, alarms and environmental sounds are indexed like speech.

!!! warning "Preview"
    Functional real-model inference is qualified on Linux x86-64 CPU. Retrieval
    quality on representative media is **not** qualified; scores are ordering
    values, not probabilities. The repository acceptance record is
    [known limitations](../reference/known-limitations.md).

## 1. Provision the model and runtime

The profile uses the EmbeddingGemma 2 Q8_0 GGUF pair published by
[`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
(revision `58faf90940886df2d62be555c8765d00dac362c5`, converted from
`google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`,
Apache-2.0) and a CPU build of llama.cpp release `b11765`
(`9df0b73c707900b7277a46b5ae52ff7e7e567d13`). Capicove never downloads them.

| File | SHA-256 |
| --- | --- |
| `embeddinggemma-2-Q8_0.gguf` | `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135` |
| `mmproj-embeddinggemma-2-Q8_0.gguf` | `c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f` |

Build only the server, without network features:

```bash
cmake -S llama.cpp -B llama.cpp/build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build-cpu --target llama-server
```

## 2. Write the receipt and model configuration

```bash
python scripts/embeddinggemma_cpu.py receipt \
  --model /models/eg2/embeddinggemma-2-Q8_0.gguf \
  --mmproj /models/eg2/mmproj-embeddinggemma-2-Q8_0.gguf \
  --llama-server /opt/llama.cpp/build-cpu/bin/llama-server \
  --llama-cpp-source /opt/llama.cpp --llama-cpp-release b11765 \
  --gguf-repository ggml-org/embeddinggemma-2-GGUF \
  --gguf-revision 58faf90940886df2d62be555c8765d00dac362c5 \
  --upstream-repository google/embeddinggemma-2 \
  --upstream-revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --license apache-2.0 --threads 4 \
  --output /models/eg2/receipt.json --model-config-output /models/eg2/models.json
```

The receipt checks the GGUF headers (architecture, Q8_0 quantization, 768-dimensional
output, audio encoder present), hashes every file and the server binary, and records
the llama.cpp revision and build options. The generated `models.json` binds
`unified` to provider `embeddinggemma2-llamacpp`. The provider re-verifies every
hash before it starts `llama-server --offline` on a private Unix socket.

## 3. Qualify CPU inference on your host

```bash
python scripts/embeddinggemma_cpu.py benchmark \
  --receipt /models/eg2/receipt.json --scratch /tmp/eg2-bench \
  --repeats 5 --batch-windows 16 --output /models/eg2/benchmark.json
```

Add `--audio FILE` to time a real recording's first window instead of the built-in
synthetic signal. Keep the results file: loading the model alone does not qualify
inference.

## 4. Configure the workspace

Add a `multimodal` capability to a version-2 workspace host file:

```json
"capabilities": {
  "multimodal": {
    "schema_version": 1,
    "profile": "audio-local-v1",
    "source_roots": {"media": "/srv/recordings"},
    "index_root": "/srv/workspace/multimodal/index",
    "staging_root": "/srv/workspace/multimodal/staging",
    "corpus_ref": "corpus:recordings",
    "model_config": "/models/eg2/models.json",
    "model_binding": "unified",
    "ffmpeg": "/usr/bin/ffmpeg",
    "ffprobe": "/usr/bin/ffprobe",
    "max_source_bytes": 1073741824
  }
}
```

Index and staging roots must be inside `workspace_root` and disjoint from each
other and from source roots. Enable `capicove.search.multimodal` and
`capicove.multimodal.get_evidence` with the `evidence:search` and `evidence:read`
permissions. Import, build and activation require a `write` or `admin` profile.

## 5. Import, build, activate, search

```bash
capicove retrieval import --runtime-config host.json --root media --relative-path dog.wav
capicove retrieval build --runtime-config host.json
capicove retrieval activate --runtime-config host.json --generation ug-...
capicove retrieval search --runtime-config host.json --query "a dog barking"
capicove retrieval search --runtime-config host.json --audio-query /srv/recordings/clip.wav
capicove retrieval search --runtime-config host.json --audio-query /srv/recordings/long.wav \
  --start-ms 60000 --end-ms 75000
capicove retrieval get-evidence --runtime-config host.json --evidence-id mev-...
```

- Sources with several audio streams need `--audio-stream-index`.
- An audio query without a range must fit one 30-second window; longer files need
  `--start-ms`/`--end-ms`. Nothing is silently truncated.
- Imports append to an inventory; each build consumes every non-retired import.
  `capicove retrieval forget --source-ref media:dog.wav` affects the next build.
- A build never changes search results until you activate it. Rebuilding unchanged
  inputs returns the same generation ID and reuses cached vectors.
- Evidence returns the source reference, stream, exact sample interval, millisecond
  range and PCM digest, never a fabricated transcript or a private path.

The same Tools are served over MCP by `capicove mcp serve --config host.json` as
`capicove_search_multimodal` and `capicove_multimodal_get_evidence`.

## Unified text, image, audio and video profile

Set `"profile": "unified-local-v1"` to index Markdown notes (`.md`, `.markdown`),
PNG/JPEG images, audio and video (`.mp4`, `.mov`, `.mkv`) in one shared space. Text queries combine the shared dense
lane with exact-word lexical matching over notes, so rare identifiers such as
invoice numbers still rank first. Image and audio-clip queries use the dense lane
only. Results list notes, images and audio windows together:

```bash
capicove retrieval search --runtime-config host.json --query "invoice ZX-4471"
capicove retrieval search --runtime-config host.json --image-query /srv/media/red.png --target image
```

Restrict results with repeated `--target text_chunk|image|audio_segment|keyframe`
(MCP: `targets`, `media_types`, `text_origins`, `time_range`).

Each video contributes one keyframe every 10 seconds from its first video stream
(scaled to at most 896 pixels per side; results carry `timestamp_ms`) and 30-second
windows of its soundtrack as `audio_segment` results. A video with one audio stream
uses it; a silent video yields keyframes only; a video with several audio streams
needs `--audio-stream-index`. Videos longer than one hour (360 keyframes) are
rejected rather than sampled sparsely. Keyframe and soundtrack evidence is decoded
from the bytes preserved at import, never from the original path.
