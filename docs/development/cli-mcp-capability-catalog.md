# CLI/MCP capability inventory — Issue #68

This is Task 1's implementation inventory, not a maintainer-approved expansion of
the public catalog. RFC-0009/ADR-0015 (memory) and RFC-0010/ADR-0016 (Wiki) were
accepted on 2026-09-28; their new CLI/MCP compositions and the remaining delivery
waves are not completed by this inventory. The common transport reuses the
accepted ADR-0003 boundary.
For implementation order, code ownership, and acceptance evidence, see the
remaining functional-closure plan (archived development reference).

## 2026-09-30 source-tree registry reconciliation

The evaluated commit is `9533a2a` with uncommitted changes; this is not a
release-candidate SHA. Running the current environment's `capicove --help`
exposes exactly the 14 top-level names listed below. `mcp --help` exposes
`serve` and `catalog`; `setup --help` exposes `init` and `doctor`; `batch
--help` exposes only `run`. `book --help` exposes the four result commands
listed in the Book section below alongside its existing workflow/chunk
commands. These commands were observed from the source environment, not from
a clean installed wheel.

The core distribution declares two console scripts in `pyproject.toml`:
`capicove` and the separate, explicit
`capicove-download-image-models`. The latter is a model-management helper,
not a common-host Tool or an implicit startup download. The five optional
plugin distributions in `plugins/` declare exactly three `capicove.retrievers`
entries (`faiss`, `opensearch`, `elasticsearch`), one
`capicove.execution_backends` entry (`spark`), and one
`capicove.document_parsers` entry (`paddleocr`). This is a declaration
inventory, not proof that those extras are installed or qualified.

## Executable profile in this slice

`capicove mcp serve --config /absolute/host.json` hosts the version-1 `image`
or local `memory` profile, or an explicitly selected version-2 combination of
those supported Tools. `capicove setup init WORKSPACE` creates a read-only
offline version-2 starter profile, and `setup doctor --config PATH` checks its
selected local Tool readiness. This is a PR01 development slice, not completion
of the full workspace-host contract.
`capicove mcp catalog --config /absolute/host.json` performs composition checks
and returns deterministic JSON without starting stdio or opening a trace database.
Neither command discovers implicit configuration or downloads a provider.

No Tool is enabled by default. `enabled_tools` must list exact canonical IDs;
the MCP name is `capicove_` plus the canonical suffix with `.` and `-` replaced
by `_`. All image Tools below have contract version 1. Their CLI entry is
`capicove image run --tool <catalog name>`.

| Canonical ID | CLI catalog name | New host status |
| --- | --- | --- |
| `capicove.image.filter.aspect-ratio` | `image_aspect_ratio_filter` | Supported when selected |
| `capicove.image.filter.shape` | `image_shape_filter` | Supported when selected |
| `capicove.image.filter.size` | `image_size_filter` | Supported when selected |
| `capicove.image.filter.subplot` | `image_subplot_filter` | Supported when selected |
| `capicove.image.phash` | `image_phash` | Supported when selected |
| `capicove.image.lsh-search` | `image_lsh_search` | Supported when selected |
| `capicove.image.remove-background` | `image_remove_background_mapper` | Provider composition deferred |
| `capicove.image.diff.areas` | `imgdiff_difference_area_generator_mapper` | Provider composition deferred |
| `capicove.image.diff.caption` | `imgdiff_difference_caption_generator_mapper` | Provider composition deferred |
| `capicove.image.filter.aesthetics` | `image_aesthetics_filter` | Provider composition deferred |
| `capicove.image.filter.nsfw` | `image_nsfw_filter` | Provider composition deferred |
| `capicove.image.filter.text-matching` | `image_text_matching_filter` | Provider composition deferred |
| `capicove.image.filter.text-similarity` | `image_text_similarity_filter` | Provider composition deferred |
| `capicove.image.filter.watermark` | `image_watermark_filter` | Provider composition deferred |
| `capicove.image.filter.pair-similarity` | `image_pair_similarity_filter` | Provider composition deferred |
| `capicove.image.perceptual-deduplicate` | `image_perceptual_deduplicator` | Provider composition deferred |
| `capicove.image.perceptual-deduplicate-ray` | `ray_image_perceptual_deduplicator` | Provider composition deferred |
| `capicove.image.ocr` | `image_ocr` | Provider composition deferred |

There are 18 image Tool specs, not 19. The supported six use their existing
default deterministic providers. The startup operator owns the absolute
`read_roots`, identity, and permissions. They require `image:read`, do not write
source files, and do not access a network. Existing inline-image limits remain
part of the canonical Tool contract. Optional tracing is an explicit local write.
Unsupported Tools fail startup instead of silently disappearing from the requested
catalog. Profile-specific provider settings belong to later Task 7 work.

## Temporal memory: accepted v1 Tools, selected host available

All 11 canonical Tools have handlers and registration adapters. The MCP host now
composes an explicitly selected subset against one local repository with
host-owned workspace, scopes, and permissions. Real stdio remember/recall and
foreign-scope rejection are tested on Linux. Content-free tracing and Windows
runtime qualification remain open; the wider Task 4 acceptance is not complete.

| Canonical ID | Existing CLI operation | Eventual classification |
| --- | --- | --- |
| `capicove.memory.remember_episode` | `memory remember` | Explicit write profile |
| `capicove.memory.assert_fact` | `memory assert-fact` | Explicit write profile |
| `capicove.memory.recall` | `memory recall` | Read profile |
| `capicove.memory.get_episode` | `memory get-episode` | Read profile |
| `capicove.memory.get_fact_history` | `memory history` | Read profile |
| `capicove.memory.retry_extraction` | `memory retry` | Explicit write profile |
| `capicove.memory.archive_memory` | `memory archive` | Explicit write profile |
| `capicove.memory.erase_scope` | `memory erase` | Admin-only destructive opt-in |
| `capicove.memory.doctor` | `memory doctor` | Read profile |
| `capicove.memory.work_extractions` | `memory work` | Explicit write profile |
| `capicove.memory.resolve_entity` | `memory resolve-entity` | Explicit write profile |

The existing memory service remains responsible for scopes, source evidence,
extraction, ontology, permissions, and storage. This slice does not revise it.

## Remaining transport work

At this working-tree checkpoint, the actual `capicove --help` registry has 17
top-level groups/commands: `doctor`, `audio`, `video`, `image`, `models`,
`search`, `get-evidence`, `source`, `index`, `batch`, `book`, `traces`, `wiki`,
`memory`, `mcp`, `http`, and `setup`. `batch` exposes only `run`; durable jobs use `capicove jobs` (RFC-0016 amendment 1).
`mcp` exposes `serve` and `catalog`; `setup` exposes `init` and `doctor`.
`book` exposes the legacy workflow commands plus the four result commands
listed below. `search` and `get-evidence` accept `--runtime-config`, and
`source import` / `index build` implement the accepted RFC-0018 operator
surface (2026-10-04). A Linux clean-wheel journey passes
(`tests/e2e/test_retrieval_wheel.py`); macOS and Windows qualification remain
open, and Windows fails closed by design (ADR-0024).

The table is a **source-tree inventory**, not a QA verdict. “Standalone” means
an existing command or capability-specific server; it does not mean that a
version-2 workspace configuration selects that capability. “Not composed” is
unavailable through the common host even when the Python API works. Freeze
required QA child IDs and actual OS/profile results separately in the
integration checklist (internal record; public support status is documented in the capability guide).

| Capability | Current CLI / other surface | Version-2 common host | Contract gate and next task | QA families |
| --- | --- | --- | --- | --- |
| Search / Evidence | `source import`, `index build`, and `search` / `get-evidence --runtime-config` run the offline FTS5 journey; `capicove http serve` exposes them over loopback HTTP. | `capicove.search` and `capicove.get_evidence` compose through `capabilities.retrieval` on POSIX. | RFC-0018/ADR-0024 accepted; native macOS/Windows wheel qualification remains in Task 02. | `RET-*`, `SRCH-*`, `EVD-*` |
| Image | `image run` has six default local providers; 18 canonical specs exist. | Six deterministic Tools are selectable; the other 12 are unavailable. | Accept selected-provider configuration before Task 03 qualification. | `IMG-*`, `WSH-*` |
| Video split/filter/deduplication | Standalone CLI plus `--runtime-config` host mode; retrieval APIs exist. | Split, metadata/motion filter, and the new read-only `capicove.video.deduplicate` compose through `capabilities.video` (RFC-0021, 2026-10-06). | Model-based filter criteria, scan/delete Tools, and Video Search remain deferred; native macOS qualification open. | `VSP-*`, `VFL-*`, `DED-*`, `VRQ-*` |
| Audio transcription, intake, and clipping | Standalone transcribe/clip CLI plus `--runtime-config` host mode and `audio import`. | `capicove.audio.import`, `capicove.audio.transcribe` v2, and `capicove.audio.clip` v1 compose through `capabilities.audio`; audio config v2 adds a clip output root (RFC-0005, RFC-0022). | Clipping uses real FFmpeg qualification; real ASR qualification remains blocked until pinned Whisper/VAD assets are provisioned; ASR-origin search deferred. | `AUD-*`, `WSH-*` |
| Wiki / Memory | Wiki operator CLI exists; Memory has standalone CLI and 11 canonical specs. | Selected Memory Tools compose on Linux; Wiki is not composed. | Accept Wiki Tool/config IDs; finish Memory tracing and native storage qualification in Task 05. | `WIKI-*`, `MEM-*`, `WSH-*` |
| Book workflow / chunks | Standalone `book chunk-tasks`, `chunk-status`, `chunk-submit`, and `chunk-publish` commands and workspace-bound `book chunk-mcp` bindings exist. The local result ledger and cited Wiki publisher have focused tests. | Not composed. | RFC-0015/ADR-0021 are accepted; finish recovery/security, selected-host authority, real supported-PDF wheel, and native OS qualification in Task 06. | `BOOK-*`, `BRP-*` |
| Batch / Trace / Models | Standalone `batch run`, `traces`, `models install` remain. `capicove jobs plan\|enqueue\|status\|cancel\|worker\|resume`; `traces … --runtime-config`; `models plan\|verify\|install --runtime-config`. | Ten RFC-0016 Tools compose in the version-2 host (MCP, HTTP, configured CLI) from `capabilities.batch/trace/model`; batch video ops reuse `capabilities.video` pins (2026-10-09). | Amendment 1 accepted 2026-10-09. Native macOS run and real model install qualification remain open. | `BAT-*`, `TRC-*`, `JOB-*`, `MDL-*`, `OPS-*` |
| Data plans / local Spark | Python service and optional Spark plugin only. | Not composed. | Accept public run identity, Tools, and config; Task 08. | `DAT-*` |
| Document graph | Public Python application service only. | Not composed. | Accept run-discovery and transport contract; Task 09. | `GRL-*` |
| Skill evolution | Python clustering and promotion policy only. | Not composed. | Accept bounded Tool/authority contract; Task 10. | `EVC-*` |

The current optional plugin entry points are `faiss`, `opensearch`, and
`elasticsearch` in `capicove.retrievers`; `spark` in
`capicove.execution_backends`; and `paddleocr` in
`capicove.document_parsers`. Their distributions live under `plugins/` and
remain opt-in. Plugin discovery does not supply a core offline text index,
authorize a workspace Tool, or prove an installed-wheel journey. Before each
PR's QA run, reconcile these names and versions with the evaluated wheel and
record any additional or missing entry point in the run inventory.

The four accepted Book result IDs are `capicove.book.chunk-tasks`,
`capicove.book.chunk-status`, `capicove.book.chunk-submit`, and
`capicove.book.chunk-publish`. The standalone Book MCP includes the two reads
by default and permits writes only when selected at startup with a private
workspace root. Their presence does not imply version-2 common-host selection
or the `BRP-*` installed-wheel journey. Keep the legacy reference-only
`chunks(tasks=true)` behavior distinct from submitted results.

`capicove http serve` (RFC-0020, 2026-10-06) is the launchable loopback HTTP
transport for a version-2 workspace: it serves the same selected bindings as MCP
on `127.0.0.1` behind a per-start bearer token, through an isolated worker
process. The older factories below remain Python-only.

The optional HTTP inventory is separate from CLI/MCP: `create_search_app`
registers `POST /v1/search` and, only with an explicit Evidence handler plus
context factory, `POST /v1/evidence`. The SigLIP2 inference app registers
`GET /health`, `POST /v1/similarity`, `POST /v1/embeddings/image`, and
`POST /v1/embeddings/text` after a verified local model is composed. Neither
factory starts a listener or supplies the missing version-2 workspace host
configuration. Task 02 must compare Search/Evidence results across the
configured CLI, MCP, and applicable HTTP adapter against independent fixtures;
the SigLIP2 routes have their own model-serving qualification.

The current draft host catalog excludes model installation, duplicate deletion,
Wiki migration, restricted trace export, and memory erasure. Later administrative
Tools require their exact accepted contract and configured authority. The draft
does not add a remote listener, authentication scheme, or implicit model download.

## Verification and release boundary

The first slice is covered by common-host, config, CLI, compatibility, real stdio,
and installed-wheel tests. Full repository gates remain required before technical
completion. Fixture and transport results are not evidence of model quality or
production capacity. See the [operator guide](../site/guides/mcp-stdio.md).
