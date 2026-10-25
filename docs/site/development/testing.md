# Testing

Capicove separates unit, contract, integration, end-to-end, model, and GPU tests.
The default suite remains deterministic and avoids network access. Optional-media
and model tests use explicit markers and locally provisioned fixtures.

```bash
make test
uv run pytest tests/unit
uv run pytest -m integration
```

Documentation uses a real strict build rather than tests that search configuration
text:

```bash
make docs
```

The strict build fails on unresolved links, missing navigation pages, malformed
configuration, and mkdocstrings resolution errors.

## Video retrieval qualification

`tests/evaluation/video_retrieval` runs a frozen, project-authored metadata corpus
without models or network. It covers ten query/scenario classes, the configured
fusion weights, temporal assembly, visual ablation, fixture integrity and
independently specified judgments. It is not a real-media model accuracy claim.

Native FAISS tests and the Elasticsearch/model integration matrix are opt-in.
Million-vector capacity and full HTTP measurements write raw results outside Git.
They require explicitly provisioned local services and frozen model/index/corpus
identities. See [Video retrieval](../guides/video/retrieval.md) for the current
experimental status, and `docs/development/video-retrieval-qualification.md` in
the repository for exact procedure, thresholds and publication rules.
