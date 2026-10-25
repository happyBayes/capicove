# Maintaining public source snapshots

The public project is [happyBayes/capicove](https://github.com/happyBayes/capicove).
An internal development repository can retain planning, review records and
restricted assets while publishing a reviewed file snapshot with independent
Git history. A branch derived from internal development still carries that
history; pushing it to a public remote does not sanitize earlier commits.

## File boundaries

| Material | Storage and publication rule |
| --- | --- |
| Source, tests, plugins, RFCs, ADRs and public documentation | Versioned and included after review |
| Existing internal plans, product requirements, reports and research sources | Kept in the internal repository, excluded by export attributes |
| New local notes, credentials and machine policy | `.internal/` or `docs/internal/`; ignored by Git and excluded from exports |
| Shared internal records that need version history | Keep in the internal repository's excluded documentation directories; never force-add credentials or local policy |
| Public development contracts and authored evaluation corpora | Explicitly allowlisted in `.public-export.toml` and `.gitattributes` |
| Design references and unused brand variants | Excluded; only the README logo and authored Wiki result illustration are allowlisted root assets |
| Model weights, user media, generated indexes and caches | Provisioned separately; excluded from source publication |

`.gitignore` prevents accidental addition of new local files. It does not untrack
existing files. `.gitattributes` controls `git archive` without deleting the
internal originals. `.public-export.toml` independently checks the exported
inventory; new files under `docs/development/` are private unless allowlisted.
Keep the two export configurations aligned when promoting a document.

Public tests read the public graph ledger, deferred PDF adaptation design and
reverse-image lifecycle requirements under `docs/architecture/`. They do not
require the internal implementation plans. The remote MCP JSON manifests retain
their existing paths because they are versioned contracts used by coverage tests.

## Inspect uncommitted changes

From the repository root:

```bash
python scripts/public_snapshot.py --working-tree
```

This inspects edited files and new non-ignored files using current export
attributes, without staging, committing or creating a new Git repository.
It checks excluded paths, archive members and relative document links. Draft
inspection cannot write a publication archive. `--working-attributes` checks
draft exclusion rules against the selected committed tree instead.

## Export a reviewed commit

Run the repository gates and review the complete changes before committing through
the normal contribution workflow:

```bash
make quality
make test
make docs-check
make build
```

Once the reviewed source and export configuration are committed and the checkout
is clean, choose the exact source commit and a new output path:

```bash
python scripts/public_snapshot.py --ref REVIEWED_COMMIT \
  --output /tmp/capicove-public-source.tar
```

The exporter writes the tar archive and `capicove-public-source.manifest.json`
containing a SHA-256 inventory. It does not create a repository, commit, push,
change visibility, upload packages or deploy documentation. It uses files from
the selected commit, including that commit's export attributes.

Extract the archive into a new directory and rerun tests, packaging, documentation
and secret scanning there before a maintainer creates the public root commit.
Never copy `.git`, push internal branches or tags, or use `git push --mirror`.
For later updates, apply the reviewed snapshot changes to the existing public
history; do not repeatedly replace public history with new root commits.

The path and link checks are not secret, copyright, license or security audits.
Retain the project license and copyright notices. Review logo redistribution
rights, model licenses, authored fixtures and scanner suppressions separately.
Never treat excluding an internal defect report as fixing the defect; preserve
the user-facing restrictions in [known limitations](../reference/known-limitations.md).

## Configure accounts before publication

The configured public repository is `happyBayes/capicove`, and the documentation
project slug is `capicove`. Package metadata, support links and the MkDocs editor
links use the public destination. The intended documentation address is
`https://capicove.readthedocs.io/`; local configuration does not establish that
the hosted site is active.

Maintainers must verify repository ownership, private vulnerability reporting,
branch protection, Read the Docs integration and PyPI Trusted Publishing before
announcing availability. Keep `CAPICOVE_PUBLISH_REPOSITORY` unset in internal
repositories; set it to `happyBayes/capicove` only in the authorized publishing
repository. See [releasing](releasing.md) and [Read the Docs](read-the-docs.md).
