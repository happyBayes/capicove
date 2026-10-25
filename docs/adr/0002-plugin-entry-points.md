# 0002: Discover plugins through Python entry points

- Status: Accepted
- Date: 2026-08-08
- Owners: `@TransActionCode`

## Context

Capicove must support community-provided source adapters, processors, embedding
providers, retrievers, rerankers, execution backends, web providers, and skill
packs. Requiring edits to a core registry would serialize contributions and couple
optional dependencies to the core release.

## Decision

Plugins are independently distributed packages named `capicove-<category>-<name>`.
They register providers through `importlib.metadata` entry-point groups beginning
with `capicove.`. The core plugin registry owns discovery, metadata validation,
compatibility checks, diagnostics, and loading.

Import and discovery must be side-effect free. Provider activation occurs only
after explicit selection by configuration or an application operation.

## Consequences

- Third parties can publish plugins without changing the core repository.
- Optional and platform-specific dependencies remain isolated.
- The project must publish stable plugin contracts and reusable contract tests.
- Installed plugins execute with user privileges unless an execution backend adds
  isolation, so documentation and trust signals are essential.

## Alternatives considered

- A hard-coded registry was rejected because every plugin would require a core
  release.
- Namespace-package scanning was rejected because discovery and distribution
  metadata are less explicit.
- Loading arbitrary file paths was rejected as the default because it weakens
  provenance and supply-chain controls.
