# Documenting features

Documentation is part of a feature's definition of done.

For every user-visible feature pull request:

1. Add or update a task-oriented page under `docs/site/guides/`.
2. Update `docs/site/cli/` when command syntax or behavior changes.
3. Add English Google-style docstrings to supported public Python interfaces.
4. Export stable symbols deliberately from the capability package.
5. Add new Markdown pages to the explicit navigation in `mkdocs.yml`.
6. Link the guide to its CLI and API references.
7. Run `make docs` before requesting review.

Use `make docs-serve` for local live preview. Files outside `docs/site/` are not
published, so internal design records must be rewritten into developer-facing
guidance when they become part of the product contract.

## Guide template and acceptance

Start every guide with its audience, support status, prerequisites, network/model
requirements, and input/output contract. Then use this sequence:

1. Prepare a maintained synthetic fixture or name the required host objects.
2. Show a complete command/program, or label it **integration example** and list
   every value the reader must supply. Never present undeclared host variables as
   a standalone demo.
3. Show expected result fields and how to distinguish success, warnings and errors.
4. Explain created artifacts, retry/recovery and cleanup.
5. Link the CLI, API and next supported task.

Keep the [capability matrix](../getting-started/capabilities.md) synchronized with
the registered CLI. API pages must render intended public symbols, constructor
parameters, defaults and return/error behavior. Use explicit symbol directives
for compatibility re-exports and undocumented dataclasses, then inspect HTML.

Run `make docs-check` for maintained executable examples, command discovery and
rendered link/symbol checks. A strict build alone does not verify usability.
New runnable examples can use a `docs-smoke` marker and a dedicated test oracle;
do not automatically execute arbitrary code blocks or network commands.

Heading tokens live in `docs/site/stylesheets/extra.css`. Preserve Material's
navigation, keyboard controls and responsive layout. H1 uses weight 800 and H2
700; both must meet at least 4.5:1 contrast against each theme's background.
Check a narrow viewport, keyboard navigation, search and both themes after
changing the stylesheet.
