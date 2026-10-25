# Contributing to Capicove

Thank you for helping build Capicove. Contributions use reviewed issues,
explicit capability boundaries, and pull requests.

## Communication language

Use English for code, APIs, schemas, comments, documentation, commit messages,
issues, and pull requests. This keeps the project accessible to an international
community.

## Before starting work

1. Search existing issues and pull requests.
2. Create or claim an issue with a clear, independently testable outcome.
3. Confirm the proposed module owner in
   [module boundaries](docs/architecture/module-boundaries.md).
4. Open an RFC before changing a public contract, persistent format, plugin
   interface, security boundary, or cross-module dependency direction.

Use short-lived branches created from `dev` and named with one of these prefixes:

```text
feat/<issue>-<description>
fix/<issue>-<description>
docs/<issue>-<description>
chore/<issue>-<description>
```

Do not work directly on `main` or `dev`. See the [Git workflow](docs/development/workflow.md)
for the required PR target and release process.

## Development environment

Install Python 3.11 or later and uv, then run:

```bash
uv sync --group dev
uv run pre-commit install
uv run capicove doctor
```

The committed `uv.lock` defines the contributor and CI environment. Do not hand-edit
it; regenerate it with uv after dependency changes.

## Development rules

- Add behavior through a failing test, then implement the smallest passing change.
- Keep the core package free of unnecessary runtime dependencies.
- Import optional dependencies only inside the feature that needs them.
- Do not access another capability's private implementation.
- Preserve immutable source evidence and end-to-end lineage.
- Do not enable network access or telemetry by default.
- Never commit source data, generated indexes, model weights, credentials, or
  private user content.
- Include license, privacy, and consent implications when adding a source adapter.
- Add an ADR when a decision changes long-lived architecture constraints.

See [testing guidance](docs/development/testing.md) and the
[development workflow](docs/development/workflow.md) for details.

## Quality checks

Run the same gates used by CI before opening a pull request:

```bash
make quality
make test
make build
```

For a focused test during development:

```bash
uv run pytest tests/unit/test_cli.py -q
```

## Secret scanning false positives

The `detect-secrets` hook runs locally and in CI. Its entropy detectors can flag
public model revisions, content hashes, and synthetic test values. Check the
value's origin and use before treating a finding as a false positive.

For a confirmed false positive in a comment-capable file, add a reason and an
inline annotation to the exact line:

```toml
# Public model repository commit SHA, not a credential.
revision = "0123456789abcdef0123456789abcdef01234567" # pragma: allowlist secret
```

Use the same annotation for deliberately fake credentials needed by security
tests, with a comment explaining that they are synthetic. Keep each annotated
line limited to the non-secret value: the annotation suppresses findings on that
line. Review the annotation again whenever the value changes.

Keep test directories and model manifests in the scan. Do not disable entropy
detectors, raise their thresholds globally, or exclude all hexadecimal values to
resolve individual false positives. For files that cannot contain comments, use
a separately reviewed, narrowly scoped baseline if needed; do not automatically
baseline every finding. See the
[detect-secrets documentation](https://github.com/Yelp/detect-secrets/tree/v1.5.0#inline-allowlisting).

Verify the working tree with the same secret scan used in CI:

```bash
uv run pre-commit run detect-secrets --all-files
```

If pre-commit reports that stashed changes conflicted with hook auto-fixes, run
the formatting hooks on the intended files before staging. Review their changes,
stage only the intended changes, and retry the commit. `--all-files` scans the
working tree without the staged-only stash workflow; it does not update the
staged copy of a file.

## Commit and pull request conventions

Use Conventional Commit prefixes:

```text
feat: add a user-visible capability
fix: correct user-visible behavior
docs: change documentation only
test: add or improve tests
refactor: change structure without changing behavior
build: change packaging or dependencies
ci: change automation
chore: perform repository maintenance
```

Each pull request must:

- address one coherent outcome;
- link its issue or RFC;
- explain user and architecture impact;
- include tests for behavior changes;
- update relevant documentation;
- declare dependency, privacy, security, and compatibility impact;
- pass all required checks;
- receive the approvals required by [GOVERNANCE.md](GOVERNANCE.md).

The repository uses squash merging. The pull request title becomes the final commit
message and must follow the same convention.

## Reporting security issues

Never disclose a suspected vulnerability in an issue or discussion. Follow
[SECURITY.md](SECURITY.md).
