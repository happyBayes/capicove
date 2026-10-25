# Video retrieval

Status: **experimental, not qualified for production support**. Local engineering
acceptance passed; production support review remains open. No reviewed production
qualification summary is available. The local authored-data, ANN, index recovery and real-service
checks have engineering evidence; that evidence does not establish real-media
semantic accuracy, million-vector service capacity, or a supported public profile.
See the current [models and service profiles](../../getting-started/runtime-profiles.md)
for the real-service coverage.

## What the implementation does

Text or authorized image queries return keyframe or video-segment metadata.
Video-clip queries are unsupported. Offline processing prepares verified previews,
visual representations and temporal evidence. Online search does not decode or
recognize target video.

Video OCR uses the public image-OCR capability through injection. It does not own
another OCR engine or import PaddleOCR/PDF parser internals. A frame observation
is not evidence that the same text persisted throughout its scene. Shared OCR
provider identity and temporal association policy participate in cache identity.

The local runtime composes a unit catalog, immutable generation sets, visual
indexes and optional ASR/OCR text lanes. NumPy is the bounded exact reference;
FAISS is an optional ANN plugin. Weighted reciprocal-rank fusion combines ranks,
not raw scores. Queries pin one generation; unavailable optional lanes are reported
as degradation. Corrupt required indexes fail the request; malformed optional
candidate batches are discarded and marked as degraded. Compact responses carry references;
full evidence requires a separate permission-checked read.

Tool, CLI, MCP and HTTP adapters share the same search operations. The default
application does not silently provision models, indexes or a workspace. A caller
must explicitly inject a configured runtime and authorized resource resolvers.

## Runnable local MVP

`capicove.runtime.video_search.LocalVideoSearchRuntime` now connects preparation,
physical index publication, atomic generation activation, leased search and the
existing Tool/CLI/MCP/HTTP adapters. Close it after use to drain query workers and
release index resources. Reopening the same workspace restores the active snapshot.
The provisioned text backend remains owned by the caller.

For an operator-run text query, install the Elasticsearch retriever plugin and
start the existing verified local SigLIP2 service and a TLS Elasticsearch instance.
Then run from the repository root:

These commands are integration recipes: replace every path and endpoint with
your provisioned service values. Port 8000 below is the model service, not the
MkDocs preview. If `make docs-serve` already occupies port 8000, configure the
model service on another loopback port and use that port in both commands.

```bash
uv run python scripts/run_video_retrieval.py \
  --workspace /absolute/path/video-workspace \
  --model-endpoint http://127.0.0.1:8000 \
  --es-endpoint https://localhost:9200 \
  --es-ca /absolute/path/http-ca.crt \
  --build-video /absolute/path/authorized-video.mp4 \
  --query "a person at a whiteboard"
```

The build flag authorizes only the selected source path. Omit `--build-video` on
subsequent runs to query the persisted snapshot. Each build replaces the active
**single-video snapshot**; it does not append videos to a larger corpus. A failed
build preserves the prior active snapshot but may leave unpublished index members
for operator cleanup. The command emits build warnings to stderr and the canonical
query response to stdout. It never downloads models. ASR remains disabled and is
reported as `asr_not_configured`. Builds default to Paddle image OCR; supply
`--ocr-model-root /absolute/path/paddleocr/book-balanced` after installing the
optional plugin and its pinned models. Use `--ocr-provider none` to build without
OCR and receive `ocr_not_configured`. Applications can still inject the existing
public ASR and `ImageOcr` services into the runtime constructor.
An unavailable configured text lane is reported as query degradation.

To expand a visual hit returned by that query, reuse the same workspace and
provider configuration in a fresh process. Replace the placeholder below with
the hit's exact `evidence_id`:

```bash
uv run python scripts/run_video_retrieval.py \
  --workspace /absolute/path/video-workspace \
  --model-endpoint http://127.0.0.1:8000 \
  --es-endpoint https://localhost:9200 \
  --es-ca /absolute/path/http-ca.crt \
  --get-evidence resource:sha256:REPLACE_WITH_RECEIPT_HASH \
  --section metadata --section preview --section lineage
```

Omitting `--section` reads metadata only. `--query` and `--get-evidence` are
mutually exclusive; reading evidence cannot also rebuild the workspace. The
operator explicitly authorizes this local CLI to read the selected workspace.
The runtime leases the current generation, verifies the stored receipt and unit
binding, and returns opaque references rather than filesystem paths or media
bytes. A receipt that is absent from the current generation or has been altered
fails closed. Old snapshot evidence is not automatically available after a
replacement build. This bootstrap expands visual metadata, previews and lineage;
ASR/OCR text and query-score expansion require separately composed readers and
remain unavailable here. No successful empty response substitutes for them.

Applications must supply `authorize_evidence(context, evidence_id, resource_ref)`
at runtime construction; omission denies every evidence read. Authorization is
checked again for each resource and each invocation. The CLI context identifies
the local operator, not a remote authenticated user. MCP evidence registration
uses `runtime.register_evidence_mcp(server, context_factory=...)`; HTTP evidence
is opt-in through `runtime.http_app(evidence_context_factory=...)`. These context
factories are host-owned and must represent authenticated callers; Tool arguments
cannot grant access. The HTTP evidence route remains absent without that opt-in.

The ordinary `capicove search` and `capicove get-evidence` commands still require
explicit runtime injection. They intentionally do not infer providers, start
services or download assets. The bootstrap above is the executable local
composition, not a claim that a bare installation is fully provisioned.

When a host supplies the Search handler, the CLI accepts the canonical text and
video filters directly. For example, `capicove search "lecture" --target
video_segment --language en --start-ms 1000 --end-ms 9000 --corpus-ref course-a
--text-origin asr` narrows a configured query without selecting a provider.
`--target text_chunk` selects text-only results and cannot be combined with a
video target. Repeat `--corpus-ref`, `--artifact-ref`, `--media-type`, or
`--text-origin` to supply multiple values; repeated identical values are
deduplicated. A temporal filter requires both endpoints, and `--language` applies
only to a text query.

Applications can call `runtime.cli(...)`, `runtime.register_mcp(server)` or
`runtime.http_app()` over the same instance. The HTTP app is a local development
adapter and requires a separately configured host to serve it. Image queries use
the same model adapter with an explicitly authorized image resolver.

For this MVP, keyframes and segments share one model/preprocessing identity.
Segments pool normalized keyframe vectors; a query is encoded once and reused by
both visual lanes. The keyframe lane intentionally contributes temporal evidence
to segment-only queries. This is a bounded MVP choice, not the only long-term
model architecture: pooling loses event order and can dilute brief actions.
A temporal encoder requires its own reviewed space and fusion policy.

## Local model process

`LocalSiglip2QueryEncoder` reuses the existing local SigLIP2 API. The composition
root verifies the pinned model manifest before starting the model process and
binds its preprocessing identity to the generation. The adapter accepts only
numeric loopback HTTP origins, disables proxies and redirects, enforces deadlines
and response limits, and verifies returned model and vector identities.

Separate model and FAISS processes avoid conflicting native OpenMP runtimes on
macOS. Do not enable `KMP_DUPLICATE_LIB_OK`. Loopback-only transport is not remote
authentication; do not expose these development services to a public network.

## Qualification and limitations

For the local MVP, acceptance covers offline preparation, exact retrieval and
generation recovery, callable query transports, evidence lineage, and independent
OCR/ASR degradation. OCR recognition accuracy is not an MVP release gate. Stage 4
semantic accuracy, million-keyframe capacity, memory budgets and latency targets
belong to enterprise qualification and do not block the functional MVP. See the
Issue #39 scope decision (archived development review).

The authored metadata corpus checks ranking, fusion and temporal association.
Synthetic vectors check ANN behavior and resource use. Real local model/HTTP
measurements check transport and runtime composition. These are different evidence
types: none may silently substitute for another.

No reviewed production qualification summary is published here. Do not advertise
a capacity, latency or semantic-accuracy guarantee from small fixtures, synthetic
timing, or an unreviewed local run. See [Testing](../../development/testing.md)
for deterministic and opt-in checks. The repository's
`docs/development/video-retrieval-qualification.md` records the reproducible
procedure and the exact evidence required before changing this status.
