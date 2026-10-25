# Documentation site development

During development, preview the documentation locally without importing the
project into Read the Docs:

```text
Edit Markdown or Python docstrings
        ↓
make docs-serve
        ↓
Preview at http://127.0.0.1:8000/
        ↓
make docs
        ↓
Commit the changes
```

## Read the Docs administration

Repository configuration prepares the build, but a Read the Docs administrator
must connect the service:

1. Import `happyBayes/capicove` into Read the Docs with the project slug `capicove`.
2. Use `main` for the latest public development documentation; select a verified
   release tag as the stable default when a release is published.
3. Enable pull-request builds and verify the repository webhook.
4. Confirm a build uses `.readthedocs.yaml` and `mkdocs.yml` from the selected ref.
5. Activate version tags when public package releases begin.

`main` is continuously updated public documentation. Pull-request builds are previews.
Release tags become immutable versioned documentation. Do not configure a separate
`mike` or GitHub Pages deployment because Read the Docs owns version routing.

If a build fails before MkDocs starts, inspect repository access, webhook, and
environment setup. If MkDocs starts and fails, reproduce it locally with
`make docs`.
