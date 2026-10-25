# Capicove Governance

Capicove is maintained at `happyBayes/capicove`. These rules define
contribution review, change control and release authority.

## Roles

### Contributors

Contributors report issues, improve documentation, write tests, and submit code
through reviewed pull requests.

### Reviewers

Reviewers have demonstrated knowledge of one or more modules. They assess
correctness, tests, compatibility, privacy, security, and documentation. Reviewer
approval alone does not grant merge or release authority.

### Maintainers

Maintainers are listed in `.github/CODEOWNERS`. They triage work, approve RFCs,
merge pull requests, manage releases, and enforce project policies.

### Security maintainers

Security maintainers receive private vulnerability reports and coordinate fixes
and disclosures. Repository administrators coordinate the security response until a dedicated
security team is established.

## Decision process

Routine fixes and documentation changes require one approving review from a
reviewer or maintainer responsible for the affected area.

The following changes require an accepted RFC and one approving review from a
maintainer:

- canonical contracts or serialized schemas;
- persistent storage formats or compatibility guarantees;
- plugin entry-point groups or plugin contracts;
- security, privacy, consent, or trust boundaries;
- cross-module dependency direction;
- removal or incompatible change of a public API, CLI, or MCP tool;
- license or governance changes.

When consensus cannot be reached, maintainers document the competing options and
the repository administrator makes the final decision. The decision and rationale
must be recorded in the RFC or an ADR.

## Merge policy

- Normal changes reach `dev` through pull requests; `main` receives verified release
  and hotfix pull requests only. The authoritative branch and PR workflow is in
  [docs/development/workflow.md](docs/development/workflow.md).
- Required CI checks must pass.
- Review conversations must be resolved.
- Code-owner review is required for owned paths.
- The project uses squash merges and a linear history.
- Force pushes and deletion of `main` or `dev` are prohibited.

Security fixes may use a private fork or security advisory. Review details can be
restricted until coordinated disclosure, but the final remediation must still be
reviewed and tested.

## Releases

Capicove follows Semantic Versioning. During `0.x`, minor releases may contain
breaking changes only when they are described in release notes with a migration
path. Persistent evidence and lineage formats receive stricter compatibility
review than internal Python APIs.

Only maintainers may create release tags or publish packages. Release automation
must use short-lived trusted credentials or GitHub trusted publishing.

## Amendments

Governance changes use the RFC process and require one approval from a repository
administrator. The governance history remains visible in the repository.
