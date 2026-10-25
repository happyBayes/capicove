# Remote MCP capability coverage and production gates

- Status: Proposed design inventory; no remote capability is qualified by this file.
- Date: 2026-10-22
- Parent: [RFC-0028](../rfcs/0028-remote-mcp-service.md)
- Baseline: `d63f946d8e265f5c7f0e335e2ba4ec306dfd6c2c` plus the inspected working tree.
- Review baseline: `bbec8d4200b6cbc80aab5b05684c23791296eefb` with unrelated
  working-tree changes, inspected again on 2026-10-22.
- Evidence: source inspection of CLI registration, Tool specs, common-host factories
  and public capability services. No new Linux or model qualification was run.

## Scope and interpretation

Each row below is mandatory for the final all-capability outcome. "Existing"
describes a local implementation or entry point, not a remote production result.
New operation names are proposed contract families, not claims of registered Tools.
An operation may be covered by a typed submit/status/result lifecycle instead of
a second synchronous implementation, but its input/output schema, authorization
and acceptance evidence must remain individually addressable.

The final release inventory must expand each family into individual operations
and supported provider variants. Compare that inventory to CLI registrations,
Tool registries, public application APIs, documented model surfaces and installed
plugin entry points. Fail coverage CI for an unclassified new public operation.
Do not use a hard-coded historical Tool count or count catalog entries as tests.

## Capability matrix

Authority classes: R = read; W = scoped write; A = approved administrative effect.
Every remote call also requires the relevant application/workspace binding. Names
under proposed namespaces require accepted schemas before implementation.

| ID | Capability and current source | Existing surface | Required remote work | Authority and decisive acceptance |
| --- | --- | --- | --- | --- |
| C01 | Image: `tools/image.py`, `runtime/image_mcp.py` | 18 Tool specs; six deterministic common-host providers | Compose all 18 with explicit model/provider profiles and artifact inputs/outputs | R/W; execute every selected provider against real qualified fixtures, including OCR and background removal |
| C02 | Image retrieval: `tools/image_search.py`, `mcp/image_search.py` | Dedicated image search binding, outside common host | Compose intent-routed reverse search with pinned index/model/source authority | R; independent relevance fixtures, citations, forbidden-source tests and missing-provider rejection |
| C03 | Video: `tools/video_split.py`, `video_filter.py`, `video_deduplicate.py`, `runtime/video_mcp.py` | Common-host split, selected filters and read-only deduplication | Add artifact handoff, all supported filter providers/criteria and long-operation dispatch; directory/action workflow is required in C25 | R/W; real FFmpeg/decoder outputs, temporal boundaries, child cleanup and no accidental source deletion |
| C04 | Video retrieval: `search.py`, `processing/video/retrieval/` | Separate search/provider APIs | Compose supported video search/evidence operations with pinned encoders and authorized evidence ranges | R; source/time-span evidence checks and actual model/index profile qualification |
| C05 | Audio: `tools/audio_import.py`, `audio_transcribe.py`, `audio_clip.py` | Three configured common-host Tools | Remote upload -> import -> transcribe/clip -> authenticated output retrieval | R/W; real Whisper/VAD/FFmpeg, selected streams, sample boundaries and lineage |
| C06 | Sources/indexes: `cli/commands/local_retrieval.py`, `runtime/local_retrieval.py` | Source import/forget and index build are operator CLI | Typed `source`/`index` mutation operations; artifact references and atomic generation activation | W/A; import -> build -> query -> forget; stale readers/results cannot leak retired evidence |
| C07 | Text/multimodal retrieval: `tools/search.py`, `tools/evidence.py`, `runtime/retrieval_mcp.py` | Search and evidence common-host Tools | Remote grant-aware retrieval, reranking, provider/index selection fixed by operator | R; FTS5 and selected FAISS/OpenSearch/Elasticsearch profiles; denied evidence excluded before result release |
| C08 | Native knowledge: `runtime/native_knowledge.py`, `runtime/knowledge_mcp.py`, `cli/commands/knowledge_base.py` | status/search/ask/evidence common-host; init/doctor/rebuild/update/forget CLI | Add bounded initialization in preallocated roots and mutation operations; amend query-only boundary | R/W/A; complete multimodal ingestion and cited answer, live source revocation and transactional rebuild |
| C09 | Temporal Memory: `tools/memory_tools.py`, `knowledge/service.py`, `runtime/memory_mcp.py` | 11 common-host specs; additional public knowledge operations | Cover all specs plus supported entity aliases/merges, fact reads and projection maintenance through reviewed contracts | R/W/A; cross-scope denial, idempotent writes, erase approval and content-free audit; do not silently enable unqualified memory tracing |
| C10 | Agent context: `agent/context_assembly.py`, `agent/memory_service.py`, `agent/memory_policy.py` | Public Python composition | Bounded context assembly and governed memory-write operations, reusing existing policies | R/W; token budgets, citations, denied evidence, no uncontrolled model/tool loop |
| C11 | Document parsing: `documents/`, `model/factory.py`, Paddle parser plugin | Parser ports/providers consumed by Book/knowledge | Explicit parse operation and parser receipts; input artifact, configured parser/profile, immutable outputs | R/W; real supported PDFs including malformed files, page coordinates, hybrid-routing exclusions and no implicit downloads |
| C12 | Book workflow: `books/service.py`, `tools/book.py` | prepare/next-task/submit/status/publish specs and CLI | Common-host composition, structure review, bounded remote task/result schemas, durable preparation | R/W/A; upload PDF -> prepare -> review/tasks -> submit -> publish with citations and restart recovery |
| C13 | Book chunks: `tools/book_chunks.py`, `book_chunk_results.py`, `books/chunk_service.py`, `chunk_result_service.py` | chunk/chunk-preview/chunks/context plus four chunk-result Tools; standalone MCP | Compose all operations with workspace authority and versioned output schemas; preserve legacy local encoding | R/W/A; parent-context bounds, run identity, append-only results, repeated submissions and cited publication |
| C14 | Wiki: `wiki/service.py`, `cli/commands/wiki.py` | init/scan/ingest/lint/build/doctor/query/migrate-v2 CLI | Add typed Tools and safe Markdown artifact interchange; plan/apply migration split | R/W/A; ingest -> lint -> build -> query; path attacks, deterministic projection and migration rollback evidence |
| C15 | Document graph: `graph_build/service.py` | Public build/resume/cancel service | Approved graph plan/build/status/result/resume/cancel operations; retain stage/profile/source semantics | R/W; real provider build, checkpoint mismatch, cancellation and evidence-grounded graph results |
| C16 | Data plans: `processing/planner.py`, `processing/local.py`, public processing models/ports | Python planner, snapshots, run store and publication | Typed plan validation/execute/status/result/cancel where backend supports it; snapshot publication/retraction controls | R/W/A; schema-preserving reads/transforms, fencing, failed publication recovery; no SQL/UDF/shell input |
| C17 | Spark: `plugins/execution-spark/` | Explicit local backend and reviewed operator set | Expose through C16, with registered backend/profile and isolated Java runtime | R/W; actual Spark operators, nested schema conversion and crash cleanup; active cancellation is a new contract/implementation requirement, not existing support |
| C18 | Batch and jobs: `tools/batch_jobs.py`, `batch/job_service.py`, `cli/commands/jobs.py` | plan/enqueue/status/cancel Tools; operator worker/resume lifecycle | Generalized remote operations and per-application quotas/ownership; reconcile underlying domain job IDs | R/W; multi-app fairness, duplicate admission, crash/fence/retry and audited resume |
| C19 | Traces/export: `tools/operator_traces.py`, `traces/`, `export/` | inspect/validate/export Tools and dataset export APIs | Resource-scoped trace access, approved restricted exports, artifact retrieval and schema/version receipts | R/W/A; redaction, private content canaries, cross-app export denial, hashes and dataset schema checks |
| C20 | Model administration: `tools/operator_models.py`, `runtime/model_mcp.py`, `cli/commands/models.py` | plan/verify/install Tools plus registry/diagnostics operations | Approved exact-plan acquisition; expose supported inventory/status/verification by opaque profile IDs | R/A; pinned revision/hash/license, allowlisted files/destinations, interruption recovery and no automatic download |
| C21 | Model inference: `model/factory.py`, provider ports, `cli/commands/model_serve.py`, `model_server.py`, `vlm.py`, `reranker.py`, `api/` | Factory interfaces and dedicated inference HTTP surfaces, not generic common-host inference | Bounded text/image embedding, similarity/reranking, OCR/parser/transcription/VLM generation, exact token counting and model discovery; preserve supported structured/tool-call data; reuse domain Tools when equivalent | R/W; actual configured model/provider output, batch/token/image limits, cancellation, no secret/path leakage |
| C22 | Skill evolution: `cli/commands/evolution.py`, `evolution/registry.py`, evaluation/generation services | init/status/active/distill/generate/evaluate/run/feedback/review/rollback CLI | Full typed MCP lifecycle, campaign/provider selection by trusted registry, supervised review and activation | R/W/A; grounded evaluation evidence, stale-report denial, authenticated reviewer, compare-and-swap activation/rollback; no self-approval |
| C23 | Plugin/runtime diagnostics: `plugins/`, `cli/commands/doctor.py`, `setup.py` | Operator diagnostics, explicit plugin registry and setup | Bounded capability/profile readiness Tool; scrub host paths/secrets; preallocated initialization under admin only | R/A; installed distribution/version/provider receipt; missing selected capability fails readiness |
| C24 | Remote infrastructure (new) | No existing public remote artifact/operation/auth service | Resumable upload/status/abort/commit and artifact reads, operation Tools, remote policy, audit, approval and isolated workers in RFC-0028 | R/W/A; two-application end-to-end workflow, adversarial isolation and production recovery |
| C25 | Video directory and duplicate actions: `cli/commands/video_deduplicate.py`, `processing/video/deduplication/actions.py` | Directory scan, manifest export, move and delete exist only on the standalone path | Bounded registered collection discovery; immutable scan plan; approved move/delete with explicit destination alias and manifest artifact | R/W/A; keeper preservation, stale source/hash rejection, path-race protection, partial failure receipts and interrupted action reconciliation |
| C26 | Index generations: `retrieval/generation_service.py`, `image_generations.py`, `video_generations.py`, `video_publication.py` | Text create/validate/activate/reconcile; image generation contracts; video register/activate/read leases and prepared-video publication | Typed build/validate/publish/activate/inspect/reconcile and safe cleanup operations over public services; no raw backend/index/path selection | R/W/A; old generation remains active on failure, active-reader leases prevent eviction, exact manifest/member hashes and source authorization survive restart |

C01 includes the following exact existing canonical IDs; these are not all
currently common-host enabled:

```text
capicove.image.remove-background
capicove.image.diff.areas
capicove.image.diff.caption
capicove.image.filter.aesthetics
capicove.image.filter.aspect-ratio
capicove.image.filter.nsfw
capicove.image.filter.shape
capicove.image.filter.size
capicove.image.filter.subplot
capicove.image.filter.text-matching
capicove.image.filter.text-similarity
capicove.image.filter.watermark
capicove.image.filter.pair-similarity
capicove.image.phash
capicove.image.lsh-search
capicove.image.perceptual-deduplicate
capicove.image.perceptual-deduplicate-ray
capicove.image.ocr
```

C09's existing Tool operations are remember_episode, assert_fact, recall,
get_episode, get_fact_history, retry_extraction, archive_memory, erase_scope,
doctor, work_extractions and resolve_entity. Additional supported public service
operations must receive explicit coverage or a documented internal-only rationale.

C13's four result operations are chunk-tasks, chunk-status, chunk-submit and
chunk-publish. Their existence does not establish safe common-host composition.

## Mandatory operation expansion discovered in review

This is a minimum enumeration, in addition to the image and memory lists above.
Each semicolon-separated operation/mode becomes an individual registry row;
grouping them here does not permit a single aggregate pass result.

| Family | Operations/modes that must not disappear under a family label | Current source anchor |
| --- | --- | --- |
| C03/C25 | Explicit-file grouping; registered directory scan; manifest export; duplicate move; duplicate delete; keeper selection and partial action receipts | `cli/commands/video_deduplicate.py`, `processing/video/deduplication/` |
| C04/C26 | Video preparation (keyframes, ASR/OCR units and temporal relations); prepared-video publication; generation register/validate/activate/resolve; read lease/eviction safety | `processing/video/retrieval/`, `retrieval/video_publication.py`, `video_generations.py` |
| C06/C26 | Source import; forget; offline build; text generation create/validate/activate/reconcile; image manifest staging/validation/publication/activation where implemented | `cli/commands/local_retrieval.py`, `retrieval/generation_service.py`, `image_generations.py`, public generation ports |
| C08 | Bounded initialize; status/doctor; rebuild; update; forget; search; ask; evidence | `cli/commands/knowledge_base.py`, `runtime/native_knowledge.py` |
| C09/C10 | Current facts; entity alias/merge and resolution checks; extraction retry/work; governed memory admission; context assembly; projection maintenance | `knowledge/service.py`, `agent/memory_service.py`, `agent/context_assembly.py` |
| C11 | Document parse with page selection; page/block/coordinate/receipt retrieval; rejected/partial document semantics | `documents/service.py`, `api/documents.py` |
| C12 | Prepare; summary tasks/list/next; structure task listing/submission; summary submission; status; publish | `books/service.py:BookService` |
| C13 | Chunk preview/create/read; parent context; chunk task/status/submit/publish; publication manifest | `books/chunk_service.py`, `chunk_result_service.py`, `tools/book_chunks.py` |
| C14 | Initialize; scan; ingest; lint; build; doctor; query; migration preview; approved migration apply | `cli/commands/wiki.py` |
| C15 | Build; resume; cancel; run discovery/status/result via accepted public accessors; preserve graph stage/profile and immutable snapshot identity | `graph_build/service.py:GraphBuildService` and graph build ports |
| C16/C17 | Plan/validate; execute; run status/result; snapshot lookup/validation/publication/retraction; each registered local/Spark operator and nested schema mode | `processing/`, `plugins/execution-spark/` |
| C18/C19 | Plan/enqueue/status/cancel/resume; scoped trace inspect/validate; each supported redaction/media-copy export mode; artifact retrieval | `tools/batch_jobs.py`, `cli/commands/jobs.py`, `tools/operator_traces.py`, `export/` |
| C20 | Registry/model inventory; profile plan/verify/install; configured model metadata; legacy image-download helper mapped to approved acquisition | `cli/commands/models.py`, `model/serving.py`, `model/cli.py`, `model/provisioning.py` |
| C21 | Text embeddings (supported encoding formats); text rerank; image and text embeddings; image-text similarity; OCR; document parse; audio transcription; VLM generate; VLM count-tokens; bound model list/lookup | `model/serving.py:ROUTES`, `api/model_server.py`, `api/bge_m3.py`, `api/siglip2.py`, `cli/commands/vlm.py` |
| C21 | VLM text/JSON response mode; bounded function-tool definitions/selection and returned calls; exact usage/token accounting | `model/vlm.py`, `model/vlm_tools.py`, `model/vlm_service.py` |
| C22 | Initialize; status; active; distill; generate; evaluate; run; feedback; review show/approve/reject; rollback | `cli/commands/evolution.py`, `evolution/learning.py`, `evaluation.py` |
| C23 | Runtime diagnostics; configured capability/catalog readiness; selected profile/model verification; explicit compatible-LLM readiness probe | `cli/commands/doctor.py`, `setup.py`, `runtime/` |

VLM returned function calls are inert structured output, not authority to execute
those functions. Existing generation is nonstreaming; MCP SSE progress does not
turn it into token-streaming model inference. Do not infer unimplemented sparse/
multi-vector embedding modes from a model family name. Provider profile tests
must list actual supported request parameters and reject unsupported variants.

Duplicate actions operate only on operator-designated mutable input collections,
never on canonical immutable evidence storage or another application's artifacts.
Approval binds keeper/eliminated identities, source versions/hashes and destination
alias. The existing path-based implementation is not evidence of remote-safe
TOCTOU handling: the remote action service must hold safe handles/recheck identity,
avoid overwrites, record each effect and reconcile partial execution. The current
read-only dedup Tool remains unchanged.

Index codecs, locks, read leases, model instance acquire/close, service start/stop,
CLI formatting flags, and pure evaluation metric helpers are implementation or
lifecycle primitives, not automatically individual Tools. Their externally useful
results must be covered by the enclosing typed operation and its tests. A public
symbol scanner proposes inventory entries; capability owners classify them.
Interfaces without implementations are marked `contract_only`, never presented as
already supported behavior. They cannot be used to excuse a missing implemented
business operation.

The provider cross-product must include the five repository plugin distributions:
FAISS, OpenSearch, Elasticsearch, Spark, and PaddleOCR (including its parser and
model-installer entry points), plus built-in provider/installer selections in
the frozen model registry. GPU/Ray variants such as perceptual deduplication need
their own runtime evidence. A CPU profile may be published as a subset, but does
not close the final full-profile request while its GPU/Ray/provider rows are open.

## Operation-level release registry

R0 produces a machine-readable registry with one row per operation/provider
combination. This document fixes the required fields and closure rule; it does not
invent qualification results for unimplemented adapters.

| Field | Meaning |
| --- | --- |
| `capability_id`, `operation_id` | Matrix family and stable business operation identity |
| `source_entrypoints` | Exact public API, CLI and existing Tool references |
| `classification` | business_tool, admin_tool, transport_lifecycle, internal_helper |
| `classification_rationale` | Required for every non-Tool classification; reviewed owner |
| `canonical_tool_id`, `contract_version`, `mcp_name` | Accepted Tool mapping; absent until contract is accepted |
| `input_schema_digest`, `output_schema_digest` | Exact registered schemas, including remote reference wrappers |
| `provider_profile`, `runtime_requirements` | Package/model/service/OS/hardware pins, not merely a provider name |
| `permissions`, `resource_policy`, `approval_policy` | Exact invocation and resource controls |
| `execution_mode`, `deadline`, `idempotency_policy` | sync or durable; cancellation/recovery semantics |
| `test_ids`, `acceptance_receipt` | Contract, isolation, installed-wheel, real-provider and recovery evidence |
| `state`, `blocking_reason` | inventoried, contract_accepted, implemented, qualified, human_accepted; truthful blockers |

For a release claiming full coverage, every business/admin row must be qualified
and approved for at least its documented supported provider profile, and every
provider advertised by that release must pass its own row. All remaining rows
need accepted lifecycle/internal classifications. No undocumented exclusion,
silent downgrade, mock provider or skipped real-runtime test counts as closure.
A business operation cannot be reclassified as internal merely because its
remote adapter is hard to implement. Removal from the supported baseline needs
a separately accepted scope change, migration note and explicit requester approval.
Profiles may be delivered incrementally, but the final requested full profile
cannot be declared complete while a mandatory family remains blocked.

## Acceptance journeys

1. Two independently registered applications initialize MCP sessions. Each sees
   only its grant-filtered catalog. Replaying another application's session,
   operation ID, evidence ID, artifact reference or pagination cursor is denied.
2. Upload real authorized image/audio/video/PDF fixtures through MCP; validate
   hashes; process them through the selected capability; retrieve results through
   MCP without a server-local path or an unprotected download URL.
3. Import evidence, build retrieval/knowledge/graph snapshots, ask/search, revoke
   a source while a query is running, and verify that stale evidence is not
   released. Check citations independently from generated answers.
4. Prepare a Book, review/submit chunk results, publish a Wiki projection and
   query it. Run scoped memory writes/recall and demonstrate another application's
   denial against the same apparent object ID.
5. Execute a data plan through local and Spark profiles, export a redacted trace
   dataset, then run a supervised Skill candidate/evaluation/review workflow.
   A background application cannot approve its own candidate or impersonate a
   human reviewer through arguments.
6. Execute each model/profile contract with real pinned assets, including explicit
   approved installation where required. Missing assets produce truthful failures
   and cannot be replaced by automatic network acquisition.
7. Interrupt a long operation before/after admission and publication. Retry its
   idempotency key, verify fencing and actual artifact state, cancel/restart, and
   recover results without duplicated side effects or false success.
8. Rotate identity keys, revoke application grants, fill disk, kill workers,
   restart the gateway, drain/upgrade/rollback and restore from backup. Record
   affected work, recovery time, lost-work bounds and client reconnection behavior.

## Delivery and evidence gates

| Gate | Required proof | Completion claim allowed |
| --- | --- | --- |
| G0: Design | Maintainer acceptance of RFC-0028/ADR-0032 and capability contract amendments | Design accepted only |
| G1: Protocol | Real external Python/TypeScript clients, exact protocol/SDK pins, OAuth and session tests | Remote transport preview |
| G2: Isolation | AUTH/ISO/CTX/NET/IO/ADM tests, independent security review and private-data canaries | Security boundary qualified for tested profile |
| G3: Capability | Every registry row, real providers, canonical parity/lineage and reviewed non-business classifications | Full capability profile qualified |
| G4: Reliability | JOB/OPS tests, 24-hour mixed-workload soak, representative load, backup/restore/rollback drill | Operational profile qualified |
| G5: Release | Repository gates, reproducible wheel/image/SBOM, deployment/client runbooks, no unresolved critical/high findings, maintainer acceptance | Production-ready for the named profile |

SLO acceptance uses raw counts and p50/p95/p99 distributions, not average latency
alone. Report failures, overload rejections, unavailable providers and skipped
tests. A signed release receipt includes G0-G5 status and the complete operation
registry. Live deployment and human acceptance are separate events after build
qualification, with exact source -> wheel -> image -> deployed configuration lineage.

## Measurable release acceptance

G0-G5 use a frozen clean source revision and its built wheel/image, not a changing
working tree. All required test IDs must have results; `skipped`, `not_run`, mocks
or missing licensed assets are blockers for the full profile. Scope changes are
explicit approval decisions, never a waiver hidden in a test marker.

| Gate/test population | Required threshold and receipt |
| --- | --- |
| Complete inventory | 100% of implemented business/admin operations and advertised provider variants mapped; zero unexplained CLI/API/Tool/plugin deltas; exact input/output schema digests |
| Positive and negative capability coverage | Every row has an authorized success, denied resource/principal, malformed input, provider-unavailable case, output/schema/lineage checks and declared timeout/recovery behavior |
| Identity compatibility | Two independently configured OAuth/OIDC providers or independent standards-conformant implementations; JWT/JWKS and opaque introspection; app credentials and delegated PKCE flows; actual admin MFA/step-up |
| Client compatibility | External Python and TypeScript clients at pinned versions; initialize/list/call/pagination/SSE/reconnect/cancel; fresh token, downgrade/rotation, error handling and large-artifact flow |
| Principal isolation | At least two tenants, two apps per tenant, two users in one app, one service principal, one shared-resource grant and one admin reviewer; zero unauthorized reads/writes/egress/cache hits |
| Target dispatch | Invoke forbidden targets through direct Tool calls, operation.submit, resume and nested payloads; all rejected before provider/effect invocation |
| Revocation races | Deterministic barriers before retrieval, provider send, worker start, commit, chunk release, cached result and approval reservation; zero old-epoch releases/commits after enforcement acknowledgement |
| Idempotency/approval races | At least 100 simultaneous repeats of the same key/plan, including crash/restart; one admitted effect identity; changed payload conflicts; cross-user same keys cannot share results |
| Protocol and boundary fuzzing | Fixed seeded malformed JSON/schema/header/Unicode/URL/cursor corpus; request smuggling and body/decompression bounds at the actual edge; zero process escape or secret canary leak |
| Nominal performance | Recorded hardware, datasets and operation mix; at least 10,000 catalog/status requests at 20 concurrent clients across 10 applications, paced below nominal limits, after warmup; p95 under 300 ms and gateway overhead p95 under 100 ms; end-to-end auth latency also reported |
| Mixed-workload soak | 24 hours across all enabled worker classes, including at least one real long-running job per class; zero unhandled crashes, lost accepted operations, cross-principal leaks or unreconciled publication failures |
| Resource stability | Resident memory growth from post-warmup first-hour median to final-hour median at most 10% under repeated fixed load; descriptor/process counts return to baseline plus declared pool size; quotas never overshoot accepted reservation bounds |
| Overload/fairness | Drive each configured limit and twice admitted capacity for 15 minutes; bounded queues/memory, explicit 429/503, no auth bypass; small read/status calls stay within twice nominal p95; a tenant cannot exhaust another's reserved capacity |
| Recovery | Inject kill/OOM/disk-full/provider/identity/audit failures at every durable state transition, at least three repeats per fault; no duplicate effects, leaked worker descendants or fabricated success; unknown effects remain blocked for reconciliation |
| Backup/restore | Recover from a backup no older than one hour onto a fresh isolated Linux host within four hours; verify payload hashes, current revocations, approvals, key access and operation ownership; no erased data becomes served |
| Release/security | Frozen Linux x86-64/Python 3.11 installed-wheel run plus each advertised architecture/GPU profile; full repository checks, dependency/license/SBOM and independent security review; zero unresolved critical/high findings |

Nominal gateway/server error rate must be below 0.1%; expected authorization
denials and deliberate fault/overload tests are reported separately with explicit
denominators, never silently removed. Performance reports include p50/p95/p99,
sample counts, throughput, queue wait, CPU/RAM/GPU/disk and cold-start figures.
Provider quality thresholds use capability-specific independently labeled fixtures,
fixed budgets and confidence intervals where applicable; transport success is not
semantic quality. G0 freezes those thresholds before model qualification, rather
than choosing them after seeing results. Real paid/network calls need explicit
fixture-egress approval and recorded cost limits.

The 99.9% monthly availability objective is measured with external authenticated
probes and scheduled representative workflow probes, including maintenance and
identity/storage failures. Health alone cannot establish availability. A 24-hour
pre-release soak cannot certify a monthly SLO; report readiness and later observed
SLO attainment separately. No contractual node-loss availability claim is permitted
for the single-node profile. Missing HA qualification blocks deployments that
require it, even if the single-node capability suite passes.

Run recovery tests for audit persistence itself: atomic admission plus a local
audit outbox; eventual append to the audit sink; no unlogged committed mutation.
When the sink is full/unavailable, privileged reads/exports and mutations fail
closed, while liveness, cancellation and private recovery control stay operable.
Audit records are sequenced, access-controlled and tamper-evident with protected
checkpoint digests; do not claim resistance to a compromised host administrator.
Operation/result expiry and resource deletion tests must include backups and
derived caches, not only deletion of the visible input file.

## Review findings and design corrections, 2026-10-22

These are design findings, not claims of exploitable deployed code. All listed
corrections are incorporated in the three proposal documents; implementation and
runtime proof remain required.

| ID | Priority | Finding | Correction and required proof |
| --- | --- | --- | --- |
| D01 | P1 | Family labels omitted directory duplicate move/delete and index generation lifecycle | C25/C26 plus operation expansion; real approved-action and generation/failure journeys |
| D02 | P1 | Model coverage omitted exact token counting, unified discovery and supported function-call/JSON modes | C21 explicit route/mode inventory; inert model calls, no false token-streaming or embedding-mode claims |
| D03 | P1 | App/workspace keys did not isolate two delegated users of one app | Full AuthorityKey for caches, sessions, jobs, artifacts, idempotency and worker mounts; same-app multi-user negative tests |
| D04 | P1 | Generic operation submission could become a permission bypass | Require wrapper plus target permissions/resource/egress/approval; deny recursive/admin target laundering |
| D05 | P1 | Final-output filtering did not prevent unauthorized retrieval/context or model egress | Pre-retrieval/pre-provider resource policy and derived-data ACL intersection; egress canary tests |
| D06 | P1 | Policy check and later publication lacked atomic revocation semantics | Workspace-owned epoch gate, explicit revocation acknowledgement/failure, local CLI coordination and fault barriers |
| D07 | P1 | Direct mutations and approval retries lacked a shared crash-safe admission rule | Durable mutation wrappers, subject-scoped keys, atomic approval reservation/idempotency/admission and effect reconciliation |
| D08 | P1 | Token lifetime depended on AS defaults; replay/claim mapping and admin step-up were underspecified | Enforced lifetime, issuer/client enrollment mapping, bounded introspection, MFA, explicit bearer replay limitation |
| D09 | P2 | Protocol error precedence, schema naming and resource-link capability were ambiguous | Hidden-Tool error precedence, pinned catalog, distinct versioned wrapper names, Tool-only artifact retrieval |
| D10 | P1 | Production gates lacked populations, thresholds and no-skip rules | Measurable G0-G5 matrix, independent identity/client implementations, per-operation/provider and fault tests |
| D11 | P2 | Four-hour single-node recovery conflicted with a 99.9% monthly claim | Explicit SLO budget/limitation and HA requirement for guaranteed node-loss availability |
| D12 | P1 | Aggregate resource/cost/retention and deletion-from-backup constraints were incomplete | Global/tenant/app reservations, decoded-content bounds, encrypted storage, retention states and restore revocation tests |

A further implementation review must verify these invariants in code. This review
does not convert Proposed contracts into accepted changes or claim that all
possible design defects have been eliminated.

## Implementation review packages

Detailed TDD plans are written after acceptance of each package, with tests before
production behavior. Use these exact existing anchors to avoid parallel domain
implementations:

- Identity/context: `contracts/tools.py`, `mcp/bindings.py`, `runtime/mcp_host.py`,
  `runtime/knowledge_mcp.py`; preserve local callers while adding explicit context.
- Transport: `mcp/server.py`, `cli/commands/mcp.py`; add optional remote modules and
  lifecycle tests instead of changing the meaning of `serve`.
- Worker/jobs: public `batch` services and `api/workspace_server.py` lifecycle;
  extract public runtime primitives, then add persisted operation ownership/fencing.
- Capability completion: existing `runtime/*_mcp.py` and public capability APIs;
  new config/Tool contracts first, then factory and real-provider journeys.
- Deployment: `scripts/deploy_release.py`, installed-wheel tests and deployment
  guides; add isolated remote profile, secret-free templates and exact manifests.

Required new test suites are `tests/contract/remote/`, `tests/unit/runtime/remote/`,
`tests/integration/mcp/remote/`, `tests/e2e/test_remote_mcp_wheel.py`, and explicit
Linux qualification suites. These paths are proposed implementation targets, not
existing tests. Package review must resolve schema and security amendments before
code is treated as accepted. No commit, push, deployment or external transfer is
part of this design-only task.
