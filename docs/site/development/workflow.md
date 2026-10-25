# Development workflow

Feature work starts from an accepted issue or RFC when public contracts, persistent
formats, plugin interfaces, or security boundaries change. Add a failing test,
implement the smallest passing behavior, and update user and developer
documentation in the same pull request.

Before review, run:

```bash
make quality
make test
make docs
make build
```

Normal features and fixes target `dev`. Promotion from `dev` to `main` follows the
repository release workflow and requires all release gates.
