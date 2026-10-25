# Native knowledge base

Use one local workspace to import evidence, update its searchable generation and
ask questions with verified citations. Capicove composes its existing capabilities;
no LightRAG service, graph database or external knowledge plugin is required.

The native knowledge workspace currently requires Linux and its secure `/proc`
file-descriptor paths. Windows/macOS support for other Capicove capabilities does
not imply support for this workspace. Models and native executables remain
explicitly provisioned assets.

## Start with local text

```bash
capicove kb init /srv/knowledge
# Put authorized UTF-8 documents in /srv/knowledge/inputs.
capicove kb update --config /srv/knowledge/knowledge.json --root inputs notes.md
capicove kb ask --config /srv/knowledge/knowledge.json "What do the notes say?"
capicove kb status --config /srv/knowledge/knowledge.json
```

The parent directory must exist; initialization requires a new destination. Output
is JSON. Text-only installation provides lexical search and `evidence_only`
responses. It does not claim that a model generated an answer. Empty results say
`no_evidence`. Nothing downloads or contacts a model implicitly.

## Select the existing models

For dense retrieval, graph construction and compatible generation, install the
`knowledge` extra. For OCR/PDF preparation also install the Paddle processor and
CPU runtime. HTTP/MCP are separate optional extras:

```bash
python -m pip install 'capicove[knowledge,mcp,api]' \
  capicove-processor-paddleocr 'paddlepaddle>=3.3,<4'
```

These commands install Python runtimes, not weights. For controlled/offline
delivery, select `knowledge`, `ocr-book` and, for native media, `audio-asr` and `media`
in the [deployment workflow](deployment.md), then install its verified wheel
bundle. Keep weights, credentials, source media and generated stores outside the
source checkout and release archives.

Reuse a [model factory configuration](model-factory.md) with these bindings:

| Binding | Capability/provider | Purpose |
| --- | --- | --- |
| `knowledge.embedding` | `text_embedding` / `bge-m3` | Dense text search and graph identity vectors |
| `knowledge.ocr` | `image_ocr` / `paddleocr` | Images, scanned PDF pages and video frames |
| `knowledge.parser` | `document_parser` / `paddle-document` | Text-origin PDF layout and blocks |

Each profile requires its existing absolute local root and reviewed manifest
SHA-256. Choose CPU or CUDA explicitly for BGE; Paddle uses CPU. Model names are
aliases, not automatically downloaded repositories. The compatible generation
profile selects any qualified OpenAI-compatible endpoint/model and its matching
verified local tokenizer; there is no binding to a particular text model.

```bash
capicove kb init /srv/knowledge-full \
  --models /srv/assets/models.json \
  --generation /srv/assets/generation.json \
  --audio /srv/assets/native.json \
  --graph --allow-network \
  --source-root papers=/srv/authorized-papers
capicove kb doctor --config /srv/knowledge-full/knowledge.json
capicove kb update --config /srv/knowledge-full/knowledge.json \
  --root papers handbook.pdf
```

`--generation` requires `--allow-network`, and the referenced generation profile
must independently opt in. Supply its credential through the configured
`CAPICOVE_...` environment variable; never put the value in CLI arguments or
configuration. Selected original evidence is sent to that endpoint for answers
and, with `--graph`, for extraction. Doctor checks configuration and pinned native
assets; `configured_runtime_unprobed` is deliberately distinct from a successful
actual import or answer.

`native.json` reuses the deployment file inventory: `files` contains `model`,
`vad_model`, `whisper_cli`, `whisper_vad_cli`, `ffmpeg`, and `ffprobe`, each with
absolute `path` and reviewed `sha256`; `backend_version` records the Whisper
runtime version. No downloads occur. Existing optional acceptance-fixture metadata
is accepted but is not ingested automatically.

## Supported evidence and coverage

| Source | Preparation | Citation location |
| --- | --- | --- |
| `.txt`, `.md`, `.markdown` | Exact UTF-8 text | Character range |
| PNG/JPEG/WebP/BMP/TIFF | Existing image OCR | Image and OCR character range |
| PDF without raster images | Existing Paddle document parser | Zero-based page and block |
| Scanned/mixed PDF | Bounded rendering and existing image OCR | Zero-based page and rendered-page lineage |
| WAV/MP3/FLAC/M4A/OGG/AAC | Existing Whisper + Silero VAD | Original audio start/end seconds |
| MP4/MOV/MKV/WebM | Frame OCR every 30 seconds plus the sole audio track | Sample time or audio start/end seconds |

Video visual coverage is sampled OCR, not continuous scene understanding. A video
must have exactly one video track and at most one audio track, and last at most
one hour. Audio-only inputs require exactly one audio track; ambiguous selections
fail instead of silently ignoring a track. Images without recognized text are
not searchable through this text-based knowledge entry; use the separate visual
retrieval capability for pixel-semantic search.

PDFs are limited to 200 pages and 12 million rendered pixels per page. Other
bounds are 128 MiB/input, two million prepared characters/source, 1,000 active
sources, 10,000 chunks and 32 disjoint source roots. These are local-workspace
bounds, not claims of distributed service capacity. OCR/ASR quality remains model
and language dependent; inspect the original when accuracy matters.

## Update, rebuild and revoke

`update` prepares the selected source, rebuilds dense representations for the
candidate corpus and optionally runs the complete typed graph workflow. It
switches the active corpus only after validation and a revision check. With graph
mode enabled, rejected extraction records or incomplete reports produce
`partial`, a nonzero CLI exit status and `activated: false`. The previous complete
generation remains recorded; changed source hashes make its old evidence
unavailable until a successful update. There is no automatic downgrade to an
incomplete graph.

```bash
capicove kb rebuild --config /srv/knowledge-full/knowledge.json
capicove kb forget --config /srv/knowledge-full/knowledge.json \
  --root papers handbook.pdf
```

Rebuild reprocesses all active sources after a model/profile change. Forget
immediately revokes the source and invalidates graph context. It retains immutable
local history and source copies; it is not physical erasure. An answer in progress
is discarded if the knowledge revision changes. Missing, changed, symlinked or
hard-linked sources fail closed. Already delivered data cannot be recalled.

The graph retains the existing closed ontology. Native extraction uses a compact
quote-based response, and the host binds unique exact quotes to coordinates. A
relationship can declare an otherwise unlisted CONCEPT endpoint only when its
literal name is uniquely present in the cited source. Native community summaries
use short citation aliases; the host supplies task identities and mandatory
limitations before the existing report validator runs. Original provider responses,
source hashes, graph receipts and all typed graph tables remain preserved.
General reference sources never become personal self reports automatically.

## Ask and inspect evidence

```bash
capicove kb search --config /srv/knowledge-full/knowledge.json "main findings"
capicove kb ask --config /srv/knowledge-full/knowledge.json "What supports the finding?"
capicove kb evidence --config /srv/knowledge-full/knowledge.json 'kb:RETURNED_ID'
```

Search combines lexical and configured dense ranks; an active graph contributes
entity and community-report context. Answers contain individual `claims` and
citations with an `evidence_id` and exact `quote`. The evidence command resolves
that ID to the current authorized original path, content hash and page/time
location. Unknown IDs, invented quotations, omitted citations, truncated model
output and changed source authority are rejected. `insufficient_evidence` means
the model returned no supported claim. Valid citations prove traceability, not
semantic entailment of every generated paraphrase.

Initialization also writes `host.json`. Serve the same query-only tools with:

```bash
capicove mcp catalog --config /srv/knowledge-full/host.json
capicove mcp serve --config /srv/knowledge-full/host.json
capicove http serve --config /srv/knowledge-full/host.json
```

The host exposes `capicove.knowledge.status`, `.search`, `.ask`, and `.evidence`.
Mutation stays in the operator CLI. Host-owned workspace, actor, source roots and
endpoint allowlist cannot be supplied by a tool caller. This is a private
single-owner workspace, not a multi-tenant access-control service. If generation
configuration changes, restart the host to revalidate its endpoint authority.
