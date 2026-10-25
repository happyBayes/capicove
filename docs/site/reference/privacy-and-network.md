# Privacy and network behavior

Capicove is local-first. Base processing does not enable telemetry, hosted model
APIs, or automatic model downloads.

- Local paths require explicit authorization at capability boundaries.
- Source evidence is not overwritten by derived processing.
- Models and executables are selected explicitly and their lineage is recorded.
- Network access is opt-in and belongs to a separately documented adapter or
  administrative download operation.
- CLI JSON can contain local paths, hashes, transcript text, and model metadata;
  treat it according to the sensitivity of the source material.
- A standalone Wiki copies exact Markdown bytes into `raw/` and may retain
  sensitive text in pages, logs, and local indexes. Wiki operations use no network,
  but generated workspaces should not be committed or shared without review.

Review plugin and model provenance before installation. A local plugin or model
adapter runs with the permissions granted to the Capicove process.
