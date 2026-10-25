# Development Workflow

## Branch model

Capicove uses a reviewed integration workflow. `main` is always releasable;
`dev` is the shared integration branch for the next release. Neither branch accepts
direct pushes.

| Branch | Purpose | Created from | May merge into |
| --- | --- | --- | --- |
| `main` | Production-ready, tagged release history | N/A | N/A |
| `dev` | Verified integration for the next release | `main`, then maintained continuously | `main` |
| `feat/<issue>-<description>` | One new capability | `dev` | `dev` |
| `fix/<issue>-<description>` | One non-urgent defect fix | `dev` | `dev` |
| `docs/<issue>-<description>` or `chore/<issue>-<description>` | Scoped documentation or maintenance | `dev` | `dev` |
| `hotfix/<issue>-<description>` | Urgent production correction | `main` | `main`, then `dev` |
| `release/<version>` | Optional release stabilization when `dev` needs a freeze | `dev` | `main`, then `dev` |

Branch names use lowercase kebab-case descriptions. Example:
`feat/123-add-json-import`.

## Commits

Make small, buildable commits that each describe one logical change. Do not mix
format-only changes, unrelated refactors, generated artifacts, or secrets with a
feature or fix.

Use Conventional Commits, in English:

```text
<type>(<optional scope>): <imperative summary>
```

Allowed types are `feat`, `fix`, `docs`, `test`, `refactor`, `build`, `ci`, and
`chore`. Keep the summary at 72 characters or fewer, without a trailing period.
Reference the issue in the body or footer when useful, for example `Refs: #123`.
Before asking for review, squash or reorder noisy WIP commits locally when that makes
the PR easier to review. The merge itself is always a squash merge, so the PR title
must be a valid Conventional Commit and becomes the canonical `dev` or `main`
history entry.

## Daily feature and fix flow

1. Create or claim an issue with acceptance criteria and identify the required
   reviewer. Open an RFC first for the changes listed in `CONTRIBUTING.md`.
2. Update local `dev`, then create a short-lived branch from it:

   ```bash
   git switch dev
   git pull --ff-only origin dev
   git switch -c feat/123-add-json-import
   ```

3. Develop with focused commits and the required tests. Push the branch and open a
   Draft PR early so a non-author reviewer can see the contract and failing-test
   direction.
4. When ready, rebase onto the current `origin/dev`, resolve conflicts locally, and
   rerun the required checks:

   ```bash
   git fetch origin
   git rebase origin/dev
   make quality
   make test
   make build
   ```

5. Open or mark the PR ready with **base branch `dev`**. Link its issue/RFC, complete
   the PR template, request a non-author reviewer's review, and address every thread.
6. After required CI passes and one non-author approval is present, the maintainer
   squash-merges the PR into `dev` and deletes the remote feature branch. The author
   updates their local `dev` before beginning the next task.

Do not force-push shared branches. A force-push is permitted only to the author's
own PR branch after rebase; notify the reviewer because prior review comments may
need rechecking.

## Promoting `dev` to `main`

Promote only a verified, coherent release candidate. The release PR is from `dev`
to `main` (or from `release/<version>` to `main` when a stabilization branch is
used); it is never a direct push.

1. Freeze unrelated merges into the release source branch and verify all intended
   issues are included.
2. Rebase the release source on `origin/main` when necessary, resolve conflicts,
   and run `make quality`, `make test`, and `make build` from a clean checkout.
3. Open a PR with **base branch `main`**, title it `chore(release): <version>` (or a
   valid release-specific Conventional Commit), and summarize release scope,
   migration, compatibility, privacy, and rollback impact.
4. Maintainers review the release PR. It requires two approvals, including the
   maintainer and a non-author reviewer, plus all required CI checks and resolved
   conversations.
5. The maintainer squash-merges it into `main`, creates the version tag and release
   only after the release checks pass, then restores normal `dev` work. If a
   `release/<version>` branch was used, merge it back into `dev` by PR so release
   fixes are not lost.

## Hotfixes

Create an urgent production fix from the current `main`, not `dev`. Open a PR from
`hotfix/<issue>-<description>` to `main`, run the full gates, and obtain the same
review required for the risk level. Immediately open a second PR from `main` to
`dev` (or cherry-pick the squash commit into a dedicated PR) so the fix is included
in the next release. Record any exception to the normal review timing in the PR.

## GitHub protection settings

Repository administrators configure these rules in GitHub; documentation alone does
not enforce them.

| Setting | `main` | `dev` |
| --- | --- | --- |
| Require pull request before merging | Yes | Yes |
| Required approvals | 2 | 1 non-author |
| Dismiss stale approvals on new commits | Yes | Yes |
| Require review from Code Owners | Yes | Yes |
| Require resolved conversations | Yes | Yes |
| Required status checks | `lint`, `typecheck`, `unit`, `package`, `security` | `lint`, `typecheck`, `unit`, `package`, `security` |
| Allow squash merge | Yes, only merge method | Yes, only merge method |
| Allow force push or deletion | No | No |

Restrict direct pushes, branch deletion, and bypass permission to repository
administrators only. Administrators must follow the same PR rules except for a
documented emergency security response.

## Work selection

Every change begins with an issue. Use an RFC issue before implementing changes to
canonical contracts, persistent formats, plugin interfaces, trust boundaries, or
module dependency direction. Normal feature work branches from `dev`; see the
branch model above.

For the current CLI/MCP full-function program, start with the
remaining-functional-closure engineer handoff (internal record; public support status is documented in the capability guide)
and its linked task-by-task plan and QA matrix. Record contract, implementation,
and qualification states separately for each PR.

Keep branches short-lived and rebase them on `dev` before final review. Do not mix
unrelated refactors with behavior changes.

## Implementation cycle

1. Describe the observable behavior and acceptance criteria in the issue.
2. Write the smallest failing test for one behavior.
3. Run it and confirm it fails for the missing behavior.
4. Implement the smallest passing change.
5. Run the focused test, then the full quality and test gates.
6. Update user, contributor, architecture, and migration documentation.
7. Open a pull request using the repository template.

## Review

Reviewers assess correctness, failure modes, module boundaries, compatibility,
dependency cost, data integrity, privacy, security, observability, tests, and
documentation.

Authors respond to review with code or explicit reasoning. Resolve conversations
only after the concern is addressed or the reviewer agrees with the documented
decision.

## Merge and release

The project uses squash merging. The pull request title follows Conventional
Commits and becomes the final commit message. Maintainers create version tags and
releases after all release checks pass.

No automated publication is enabled during private foundation work.
