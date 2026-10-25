# Agent Instructions for Capicove

These instructions apply to the entire repository.

## Product and naming

- The project, distribution, and import package are `capicove`.
- CLI commands start with `capicove`.
- Plugin distributions start with `capicove-`; entry-point groups start with
  `capicove.`.
- Environment variables start with `CAPICOVE_`.
- Use English for code, comments, schemas, documentation, commits, issues, and pull
  requests.

## Coding style

- Write all code comments and docstrings in English.
- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  and [PEP 8](https://peps.python.org/pep-0008/) coding conventions (formatting,
  naming, imports, docstrings, and readability). Prefer the repository's configured
  Ruff / formatter settings when they encode these rules.

## Architecture

- Keep canonical contracts independent from databases, web frameworks, MCP SDKs,
  model SDKs, GPU frameworks, and execution backends.
- Depend on public contracts, never another capability's private implementation.
- Keep optional dependencies out of module import paths until the relevant feature
  is used.
- Preserve immutable source evidence, content hashes, versions, and lineage.
- Network access, external APIs, and telemetry must be explicit opt-ins.
- Public contracts, persistent formats, plugin interfaces, and security boundaries
  require an accepted RFC and ADR when appropriate.

## Model and dataset assets

- Model and dataset transfers require explicit authorization for the exact
  repository, revision, file scope, license and data usage rights.
- Pin immutable revisions, verify SHA-256 hashes, and retain provenance receipts.
- Never commit weights, private media, credentials or generated indexes.
- Runtime paths remain offline; network acquisition is a separate opt-in action.
- If `.internal/agent-policy.md` exists in an internal checkout, read and follow
  those additional local rules. Local policy files must not enter public exports.
- Internal checkouts also retain
  `docs/development/huggingface-private-collaboration-handbook.md`; when present,
  follow its shared transfer policy. Public snapshots omit that internal handbook.

## Development

- Superpowers is the default agent-development methodology for this project. When
  the Superpowers plugin is available, use its applicable workflow skills for
  brainstorming, planning, test-driven development, debugging, review, and branch
  completion. Project-specific instructions and explicit user directions take
  precedence.
- Add a failing test before production behavior, then implement the smallest
  passing change.
- Run `make quality`, `make test`, and `make build` before declaring work complete.
- Preserve unrelated user changes and never modify `reference/`.
- Do not commit source data, generated indexes, model weights, credentials, or user
  content.
- Do not initialize Git, commit, push, publish, or alter GitHub settings unless the
  user explicitly requests that action.

See `CONTRIBUTING.md`, `GOVERNANCE.md`, `docs/architecture/`, and
`docs/development/` for the authoritative project workflow.
