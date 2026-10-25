# ADR-0028: Explicit deployment and compatible generation

- Status: Accepted for implementation under RFC-0024
- Date: 2026-10-19
- Supersedes: No local-only provider contract

The operator needs a deployable package and an existing self-hosted chat service.
Adding a separate model-specific SDK or weakening offline local-model loading
would duplicate infrastructure and obscure the network boundary.

Implement the existing graph generation ports with a separate explicit
OpenAI-compatible profile, verified local tokenization and environment credential
references. Preserve strict graph output validation, current authorization and
durable receipts. Use existing package/model/native manifests for delivery and
report real acceptance separately from installation.

This adds an opt-in network adapter; it does not enable network access in existing
workspaces. Exact remote weight identity remains an operator assertion. An
independent maintainer review and publication authorization remain separate.

See [RFC-0024](../rfcs/0024-explicit-deployment-and-compatible-generation.md) for
configuration, operational limits and validation requirements.
