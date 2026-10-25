# Video retrieval qualification procedure

Repository: https://github.com/happyBayes/capicove

Status: qualification procedure, not a production acceptance receipt. Freeze the
source, corpus, runtime and model identities before collecting release evidence.
Engineering measurements and public support qualification are separate decisions.

## Separate evidence families

| Gate | Fixture and measurement | Required result |
| --- | --- | --- |
| Authored quality | 30 frozen metadata judgments, 10 scenario classes, real RRF and temporal assembly | Text/keyframe recall >= 0.90; segment recall >= 0.85; temporal IoU@1 >= 0.50 |
| ANN recall | Frozen 10k exact-vs-native query set; production-dimension representative subset | Recall@20 >= 0.95 |
| Large ANN | Million production-dimension synthetic vectors and >= 1000 known-neighbor probes | Recovery recorded separately from random-query Recall@20 |
| Opened visual memory | Native keyframe and segment indexes, cold process/reopen | <= 8 GiB |
| Serving memory | Model, visual indexes, catalog, pools and transport resident memory | <= 16 GiB; steady growth <= 5% |
| HTTP | Real model/index/catalog/text backend, warm 100, measured 1000 at concurrency 1 and 4 | Zero failed/degraded requests; c4 P95 < 500 ms; RTT <= 20 ms; compact top-20 <= 16384 bytes |
| Integrity | Corruption, read-only catalog changes, leases, activation, recovery and retirement tests | Fail closed; no mixed-generation or partial activation |

The capacity plan explicitly specifies **authored metadata**, not acquisition of
a new real-video semantic corpus. `VideoEvaluationReport.passed` intentionally
retains a stricter real-semantic-evidence requirement for public support claims.
An authored engineering test checks its quality metrics, not that aggregate flag.
Do not relabel authored data as real semantic evidence to obtain a green report.

## Deterministic checks

Run from the repository root with the locked development environment:

```bash
make quality
make test
make build
make docs
uv run pytest tests/evaluation/video_retrieval -q
```

The JSONL fixture loader fixes byte-level SHA-256 identities, closed field sets,
unit relations, finite scores and contiguous ranks. Judgment labels are authored
independently of execution. Tests cover visual ablation and reject raw-score
addition as a substitute for the configured RRF weights (OCR weight 0.5).
Fixtures contain no source media, copied transcript or private user material.

## Native ANN checks

Provision the optional FAISS package explicitly; do not install into the model
server process. Keep all generated output in a fresh ignored or temporary directory.
The benchmark refuses to replace an existing output directory.

```bash
PYTHONPATH="$PWD/plugins/retriever-faiss/src:$PWD/plugins/retriever-faiss/tests/unit:$PWD/src" \
  CAPICOVE_TEST_REAL_FAISS=1 .venv/bin/python -m pytest plugins/retriever-faiss/tests -q --no-cov
PYTHONPATH="$PWD/plugins/retriever-faiss/src:$PWD/src" \
  .venv/bin/python scripts/benchmark_video_ann.py --help
PYTHONPATH="$PWD/plugins/retriever-faiss/src:$PWD/src" \
  .venv/bin/python scripts/benchmark_video_ann_reopen.py --help
```

Freeze the seed, dimension, normalization, dtype, thread count, HNSW M,
construction effort, query effort and vector/query hashes. For the recorded local
run: 768 dimensions, seed 20260915, four native threads; query effort 1024.
The initial effort 128 failed million-scale known-neighbor recovery. Preserve
that failed result; never lower the 0.95 recall threshold. Changing startup query
effort is allowed after validation without modifying the immutable graph files.

Full exact search over a million high-dimensional vectors is not falsely reported
as measured. Record representative-subset exact recall and million known-neighbor
recovery in separate fields. Verify that the subset really matches the frozen
corpus prefix. Neither field measures model semantic relevance.

## Real local serving checks

Use the existing model-management verifier and SigLIP2 API, with a pinned public
bundle, verified receipt, offline environment and numeric loopback binding.
Provision a local TLS Elasticsearch instance with privileges restricted to the
test index prefix `capicove-video-*`. Never disable TLS or expose it externally.

The native integration matrix uses explicit environment opt-ins:

```bash
PYTHONPATH="$PWD/plugins/retriever-elasticsearch/src:$PWD/plugins/retriever-faiss/src:$PWD/src" \
  CAPICOVE_TEST_REAL_ELASTICSEARCH=1 \
  CAPICOVE_VIDEO_ES_ENDPOINT=https://localhost:19321 \
  CAPICOVE_VIDEO_ES_CA=/absolute/path/to/localhost.crt \
  CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT=http://127.0.0.1:19422 \
  .venv/bin/python -m pytest plugins/retriever-elasticsearch/tests/integration/test_video_v3.py -q --no-cov
```

For the million full-serving profile also set these explicitly before invoking
the same test with `-k 'faiss and siglip2'`:

- `CAPICOVE_TEST_VIDEO_CAPACITY_VECTORS`: frozen million-by-768 NumPy memmap.
- `CAPICOVE_TEST_VIDEO_CAPACITY_WORKDIR`: fresh absolute output directory.
- `CAPICOVE_TEST_VIDEO_HTTP_REPORT`: new absolute report file, never overwritten.
- `CAPICOVE_TEST_VIDEO_MODEL_PID`: exact verified local model-server PID.

This builds one generation with the full unit catalog and relations, both visual
indexes, physical text members and real HTTP service. Model processes and local
test servers must be stopped after measurements. Preserve raw reports locally;
do not commit indexes, vectors, previews, source media or model weights.

Memory reports must distinguish process lifetime peak, sampled current RSS,
opened-index measurements and model-process measurements. A late model snapshot
is not a continuous whole-system peak. Do not claim a complete memory gate from
a standalone plugin RSS value or a small catalog.

### Cold reopen, warm baseline and lease transition

After constructing the full fixture, run `scripts/benchmark_video_runtime_reopen.py`
in a new process, with both plugins on `PYTHONPATH` and the same explicit local
endpoint/CA environment. Required arguments are `--root`, a fresh `--report`,
the verified `--model-pid`, `--text-pid`, and `--model-manifest-sha256` used when
starting the model service. This command reuses immutable payloads; it registers
separate ANN-profile test generation sets in that test catalog.

The first full run exposed a cold-query timeout: activation validated payloads
without preloading the verified index cache. Activation now opens verified indexes
before serving begins. Native vector verification reconstructs batches of at most
512 rows, preserving the original canonical digest. Never raise the online
deadline or omit validation to hide this failure.

The cold probe samples search/model/text RSS every 100 ms beginning before native
opens, records the sum of individual peaks, and separately records the activated
post-GC process RSS as a conservative upper bound for opened visual memory.
HTTP stable-growth baselines are captured after each 100-request warmup. Native
startup/transient allocation peaks are not mislabeled as steady opened-index memory.

It holds a query lease across same-profile activation, verifies that the old set
cannot be evicted until release, and records transition RSS. This transition
shares immutable physical members; it is explicitly not a measurement of two
different million-vector graphs resident together.

### Completed-report gates

`tests/performance/video_retrieval/test_qualified_reports.py` validates the locally
produced evidence; it does not regenerate, download, or fabricate measurements.
Set `CAPICOVE_VIDEO_QUALIFICATION=1` and these absolute report paths:

- `CAPICOVE_VIDEO_ANN_SCALE_REPORT`
- `CAPICOVE_VIDEO_ANN_REOPEN_REPORT`
- `CAPICOVE_VIDEO_ANN_SUBSET_REPORT`
- `CAPICOVE_VIDEO_RUNTIME_REPORT`

Then run `uv run pytest tests/performance/video_retrieval -m performance -q`.
The gates check actual subset vector equality, recall thresholds, memory limits,
generation/profile matching, the HTTP report hash, both concurrency settings,
sample counts, zero errors/degradation, compact size and latency. Missing reports
fail an opted-in run; the default suite explicitly skips these native-evidence
gates instead of generating large assets in ordinary CI.

## Review and publication gate

Before submission run both correctness and failure-mode reviews, inspect all
staged paths and object sizes, refresh exact base/head, and run the full gates
after the final edit. Review any new warning; do not suppress it for a green log.

Publish only a non-sensitive reviewed summary tied to exact model, preprocessing,
corpus/query, index, generation, fusion/temporal policy, host, OS, storage, RTT and
concurrency identities. Include all failed runs and changes that resolved them.
Human approval, RFC acceptance, commit, push, merge and release remain separate.
WorkBuddy has not been invoked: any private-code advisory review requires explicit
authorization for that exact invocation and scope.
