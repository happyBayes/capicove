# ADR-0029: Compose a native evidence-backed knowledge workspace

- Status: Accepted for operator-requested implementation
- Date: 2026-10-19
- RFC: [RFC-0025](../rfcs/0025-native-knowledge-workspace.md)

Use one private local corpus configuration and transactional publication boundary
to compose the existing model, media, graph, retrieval and generation capabilities.
Do not add an external knowledge service or a second model registry.

Source authority is live; immutable storage is not authorization. Changes or
revocation invalidate reads, derived graph context and in-flight answers. Publish
only complete validated candidate generations. Preserve failed/partial outcomes
and exact source/page/time lineage separately from active state.

Keep mutation in the operator CLI and expose query-only operations through the
existing HTTP/MCP transport. A compatible generation endpoint remains an explicit
operator choice, independent of a particular vendor or text model. Citation
validation checks exact authorized evidence rather than trusting generated IDs.

The tradeoff is a bounded single-owner implementation with full corpus rebuilding
and existing ontology limits. Larger-scale incremental indexing or multi-tenant
policy requires a separate design; neither is silently simulated here.
