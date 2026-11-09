# RFC-0034: Offline retrieval evaluation harness, pipeline profiles and optional reranking

- Status: Proposed (drafted 2026-10-29 under requester direction; not accepted)
- Date: 2026-10-29
- Owners: Capicove maintainers
- Decision: [ADR-0038](../adr/0038-offline-evaluation-and-pipeline-profiles.md) (Proposed)
- Ledger: `docs/development/goal-status-2026-10-29.md`
- Feature status: **experimental** (`capicove eval`, pipeline profiles, reranking)

## Authorization and status

The 2026-10-29 requester goal directed implementing an offline evaluation harness,
moving hardcoded retrieval settings into profiles and adding optional reranking,
with deterministic fakes for every model-related test. That direction authorizes
this draft and the experimental implementation in the working tree. It does not
accept the public contracts, persistent formats or CLI surfaces below. Maintainer
acceptance is required before any of them is treated as stable. No commit, push,
release, model download or real-model evaluation is part of this record.

**Real-model quality validation has not been run.** Every reranker and dense-lane
number produced so far comes from deterministic fakes and measures ranking
mechanics only.

## Problem

Six retrieval stacks (offline FTS5 text, the native knowledge base, unified
shared-space search, code search, reverse-image research and video fusion) each
had their own ad hoc evaluation script or none. The KB, offline text and unified
paths hardcoded RRF constants, candidate depths and limits, so a change could not
be recorded, compared or reproduced. No search path used the existing
`TextReranker` port.

## Proposal

### 1. Evaluation datasets (persistent format)

`capicove.retrieval-eval.v1` is a closed JSON document:

| Field | Meaning |
| --- | --- |
| `schema`, `dataset_id`, `frozen_at`, `provenance` | identity and provenance |
| `stacks` | stacks the set supports (`offline-text`, `kb`, `unified`, `image`, `graphrag-local`, `graphrag-global`) |
| `corpus` | authored `documents` (`id`, `path`, `text`, optional `graph` annotations) and/or `images` (`id`, `path`, `spec`) |
| `queries[]` | `id`, `category`, `text` (nullable), `relevant[]`, optional `payload` |
| `relevant[]` | `label`, `document`, `contains` (exact quote or null), `grade` 0–3 |

Labels name a document and an exact quote, never a chunk ID, so chunker changes do
not move ground truth. Existing frozen sets are read unchanged:
`capicove.code-search-eval.v1` (labels by path and symbol) and the digest-pinned
video fixture directory. Dataset identity is the SHA-256 of the exact bytes.

### 2. Run, score and comparison records (persistent formats)

- `capicove.retrieval-eval-run.v1`: stack, adapter version, dataset identity,
  profile identity (`id`, `version`, `fingerprint`), index or snapshot identity,
  model identity (`kind` = `none` or `deterministic-fake`, and always
  `real_model_validation: "not_run"` until a real-model run exists), options,
  deterministic `results` with `results_sha256`, and separate non-deterministic
  `timing` and `environment`.
- `capicove.retrieval-eval-score.v1`: macro MRR, nDCG@k, recall@k,
  single-credit recall@k, complete-support@k, candidate-pool recall,
  `context_chars@k` where items have sizes, per-category and per-query values,
  no-answer false positives and latency statistics (count, mean, nearest-rank
  p50/p95, max).
- `capicove.retrieval-eval-comparison.v1`: requires identical dataset bytes and
  query IDs; per metric it reports means, delta, a seeded paired bootstrap 95%
  interval (2,000 resamples, reusing `summarize_paired`), wins/losses/ties and an
  exact two-sided sign-test p-value; it lists queries whose rankings changed.

MRR and nDCG reuse `capicove.retrieval.evaluation` with each item crediting at
most one not-yet-credited label (highest grade first). Recall and
complete-support credit every label an item satisfies, because one chunk can hold
several labelled facts. Candidate-pool recall uses the deepest list the public
interface returns (KB 20, offline text and unified 100, code 30, video and image
the full fused list, GraphRAG `max_evidence`); for reranked runs it bounds what
reranking can recover.

### 3. CLI (new surface)

```text
capicove eval run --stack STACK --dataset PATH --output PATH [--k N]
                  [--profile PATH] [--reranker none|deterministic-fake|bge]
                  [--reranker-model DIR]
                  [--fake-dense] [--corpus-repo DIR --git ABS_PATH] [--workdir DIR]
capicove eval score RUN [--output PATH]
capicove eval compare BASELINE CANDIDATE [--output PATH]
```

Runs build in a private scratch directory and write their output exclusively
(never overwrite). `--reranker bge` loads the verified local BGE assets from
`--reranker-model` through the existing `BgeReranker` (hash-checked, offline) and
fails before any query when they are missing; a run with a local model is marked
`real_model_validation: "measured_unaccepted"`. No such run has been made. The code stack exports `src/` and `scripts/` of the dataset's
commit from a local repository's object store with `git archive`; network
transports are disabled for every Git call.

### 4. Pipeline profiles (persistent format and public API)

`capicove.retrieval-pipeline-profile.v1` is a closed document per stack with
`profile_id`, `version` (1), stack-specific settings and a `rerank` object; its
fingerprint is the SHA-256 of its canonical JSON. Defaults reproduce the previous
constants exactly:

| Profile | Settings |
| --- | --- |
| `kb-default-v1` | `rrf_k` 60 over zero-based ranks, lexical and dense weight 1.0, `default_limit` 8, `max_limit` 20, `graph_context_limit` 8 |
| `offline-text-default-v1` | `rrf_k` 60, `lexical_candidate_count` 100 |
| `unified-default-v1` | `rrf_k` 60, `candidate_depth` 100, dense and lexical weight 1.0, `max_lexical_tokens` 64 |

Validation is strict: exact field sets, exact integer/boolean/number types (no
booleans as integers, no NaN or infinity), bounds, duplicate JSON keys rejected,
64 KiB file limit. Python entry points gain keyword-only parameters:
`KnowledgeBase(profile=, reranker=, text_encoder=)`,
`open_local_retrieval(pipeline_profile=, reranker=)`,
`open_unified_retrieval(pipeline_profile=, reranker=)` and
`TextSearchService(rerank_policy=, reranker=, rerank_texts=)`.
`capicove kb search|ask --profile PATH` loads a KB profile.

Unified generation identity (`_preprocessing_identity`) is unchanged; query-time
profiles are recorded by evaluation runs and, when reranking is enabled, in KB
results.

### 5. Optional post-fusion reranking

`rerank` = `enabled` (default `false`), `model_id` (default
`BAAI/bge-reranker-v2-m3`), `binding` (model binding, default
`retrieval.reranker`), `candidate_depth` (1–128, default 20) and
`on_unavailable` (`fail` default, or `degrade`).

The fused head of `candidate_depth` candidates is reranked; later candidates keep
their fused order. A missing reranker, a model-identity mismatch, a reranker
exception or a non-permutation output either fails the search (`fail`) or keeps
the fused order with an explicit marker (`degrade`). It is never skipped silently:

- KB results carry `rerank.status` (`applied`, `degraded`, `no_candidates`) and
  the profile identity whenever reranking is enabled; `fail` raises
  `reranker_unavailable`.
- Offline text and unified searches report `degraded: true` with the existing
  warning code `retrieval_lane_unavailable`; under `fail`, offline text returns the
  Tool error `unavailable` (`provider_unavailable`) and unified returns the failure
  code `provider_unavailable`. Adding a dedicated `rerank_unavailable` warning code
  would change the published output schema of versioned Tools that share the
  warning enum (for example `capicove.search.image@1`), so it is proposed here for
  acceptance and was not implemented.
- Unified reranks only text-chunk results of text queries; other targets keep
  their positions.

Offline text reranks full chunk text read from the pinned generation; unified
reranks stored chunk text; KB reranks chunk quotes. No path truncates text for a
reranker; the BGE adapter itself rejects overlong pairs.

## Compatibility

Default behavior and rankings are unchanged. The saved default baselines were
captured before any tracked file changed, and the same rankings reproduce byte
for byte after the change (`tests/integration/evaluation/test_stack_baselines.py`,
plus a HEAD-export cross-check recorded in the ledger). KB default output gains no
keys. No canonical enum or Tool schema changes.

## Not proposed

- Turning reranking on by default, or any claim about BGE quality on these sets.
- Profiles for code, image or video search (they keep their existing policies).
- Real-model evaluation: **not run**; it requires authorized model assets.

## Acceptance needed

1. Dataset, run, score, comparison and profile formats.
2. The `capicove eval` CLI and `kb --profile` options.
3. Reranking semantics, the fail default, and whether to add a dedicated
   `rerank_unavailable` warning code (a Tool schema version change).
4. A real-model evaluation plan (BGE reranker; embedding models) with authorized
   assets, before any quality claim.
