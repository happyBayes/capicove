# Local image models

Model download is a separate, explicit operation. Capicove runtime adapters never
silently fetch weights.

```bash
python -m pip install "capicove[models-cpu]"
capicove-download-image-models --help
```

Use `models-gpu` for a supported GPU runtime and `models-dev` for export,
quantization, or PyTorch-backed development. Authentication for private or gated
Hugging Face repositories is performed by the developer before download. Review
licenses and model provenance before installing weights.

For the official SigLIP2 preset, follow
[local SigLIP2 retrieval on CPU](siglip2-local-retrieval.md) for pinned download,
file verification, offline API startup, and a synthetic image query. The
[CPU device evaluation](../../development/siglip2-cpu-evaluation.md) records the
2026-10-12 source, HTTP, and installed-wheel results with qualification limits.

For native adapter qualification, video model criteria, score semantics, and
host troubleshooting, see the [local model qualification procedure](../../development/local-model-qualification.md).
