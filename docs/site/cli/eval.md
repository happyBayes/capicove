# Retrieval evaluation (experimental)

`capicove eval` runs a frozen query set through one retrieval stack offline,
scores the run and compares two runs query by query. It is an experimental
preview (RFC-0034, Proposed): the formats and options may change.

```text
capicove eval run --stack STACK --dataset PATH --output RUN.json [--k 10]
                  [--profile PROFILE.json] [--reranker none|deterministic-fake|bge]
                  [--reranker-model DIR]
                  [--fake-dense] [--corpus-repo DIR --git /usr/bin/git] [--workdir DIR]
capicove eval score RUN.json [--output SCORE.json]
capicove eval compare BASELINE.json CANDIDATE.json [--output COMPARISON.json]
```

Stacks: `offline-text`, `kb`, `unified`, `code`, `image`, `video`,
`graphrag-local` and `graphrag-global`.

The graph fixture uses a scripted generator and a deterministic connected
component partitioner so results can be replayed across Python patch versions.
It measures query behavior over a verified snapshot. Normal knowledge-base
graph builds continue to use the configured native Louvain implementation.

From a source checkout, run an authored multi-hop fixture with either mode:

```bash
capicove eval run --stack graphrag-local \
  --dataset docs/development/eval/multihop-graph-eval-v1.json --output local.json
capicove eval run --stack graphrag-global \
  --dataset docs/development/eval/multihop-graph-eval-v1.json --output global.json
capicove eval compare local.json global.json
```

The recorded code-search baselines pin a historical corpus commit. Replaying
them requires a local repository that contains that commit. The public source
history does not include it. To evaluate another repository, use a dataset
whose corpus field pins a commit available in the repository you supply.

- **Offline only.** Each run builds its index in a private scratch directory. No
  network access, telemetry or model download happens. The `code` stack exports
  the dataset commit's `src/` and `scripts/` trees from a local Git repository.
- **Identities.** Every run records the dataset SHA-256, the profile ID, version
  and fingerprint, and the index or graph snapshot identity. Rankings carry a
  `results_sha256`, so default-profile baselines can be compared byte for byte.
- **Fakes are labelled.** Stacks that need a model use deterministic fakes. Their
  identities start with `capicove/deterministic-fake`, and every record states
  `real_model_validation: "not_run"`. `--reranker deterministic-fake` exercises
  reranking mechanics; it is not the BGE reranker. `--reranker bge` loads verified
  local BGE assets from `--reranker-model` and fails with exit status 2 when they
  are missing; such a run is marked `measured_unaccepted`, never accepted quality.
- **Outputs are never replaced.** `run`, and `score`/`compare` with `--output`,
  create new files and fail if the path exists. Failures exit with status 2.

Scores include MRR, nDCG@k, recall@k, complete-support@k (all relevant facts in
the top k), candidate-pool recall, no-answer false positives and latency
percentiles. Comparisons report paired deltas with a seeded bootstrap interval,
wins/losses/ties and an exact sign-test p-value.

The `kb search` and `kb ask` commands accept `--profile PROFILE.json` with a
`capicove.retrieval-pipeline-profile.v1` document. `kb graph-query` runs the
experimental GraphRAG local or global query over the active knowledge graph.
