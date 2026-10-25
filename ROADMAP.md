# Roadmap

Capicove grows from local, inspectable knowledge into richer retrieval and
reviewed agent workflows. See the [changelog](CHANGELOG.md) for the first release
and the [capability matrix](docs/site/getting-started/capabilities.md) for current
workflow requirements.

## Coming soon

These capabilities are in development or available as previews in the source
checkout. Inclusion here is not a release date or a production-readiness promise.

| Capability | Direction | Current boundary |
| --- | --- | --- |
| **Local code search** | Identifier and path search with line-cited evidence; optional AST chunking, hybrid search and relation expansion | Preview; duplicate-content indexing needs a verified fix, and relation expansion remains opt-in. [Guide](docs/site/guides/code-search.md) |
| **Unified multimodal retrieval** | Search notes, images, audio and video keyframes through text, image or audio queries | Preview; duplicate-content indexing and representative real-media quality remain release gates. [Guide](docs/site/guides/native-audio-search.md) |
| **Native audio search** | Find sounds directly from waveforms, with timestamped evidence and no transcription requirement | Preview; Linux CPU inference has been qualified, but representative retrieval quality has not. [Guide](docs/site/guides/native-audio-search.md) |
| **Remote MCP** | Authenticated access to selected workspace tools over Streamable HTTP | Controlled preview; identity-provider interoperability and independent security qualification remain open. Keep artifact and approval/control features disabled pending verified fixes. [Guide](docs/site/guides/remote-mcp.md) |
| **Reviewed Wiki maintenance** | Source-cited proposals, previews, review, publication and recovery | Experimental v3; older readers cannot open migrated workspaces. Semantic quality and production capacity need acceptance. [Guide](docs/site/guides/wiki-maintenance.md) |

The [known limitations](docs/site/reference/known-limitations.md) are authoritative
for preview restrictions. Implementation, runtime qualification and release
acceptance are separate milestones.

## Further exploration

- Multimodal reranking and richer cross-media retrieval.
- Freshness automation, contradiction handling and knowledge discovery.
- Trace-derived workflows, standard operating procedures and skill packs.
- Additional execution backends and local domain indexes.

Scope and sequencing depend on reviewed proposals, reproducible evaluations and
community feedback. Public contracts and security boundaries follow the RFC
process described in [Contributing](CONTRIBUTING.md).

## Design commitments

- Preserve immutable source evidence, versions and lineage.
- Keep network access, model acquisition and telemetry explicit opt-ins.
- Keep knowledge publication and skill activation under explicit review.
- Keep infrastructure replaceable and optional dependencies optional.

Internet-scale crawling, foundation-model training, safety-critical robot control
and mandatory cloud services are outside the v0.x scope.
