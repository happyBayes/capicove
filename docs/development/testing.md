# Testing Strategy

Tests protect user-visible behavior, canonical compatibility, evidence integrity,
and extension contracts. Tests should name the regression they catch and exercise
real code rather than asserting mock behavior.

Audio Tool development and release evaluation follow the three-level strategy,
bound metric profiles, capability declarations, and component/Agent responsibility
split in [Audio Tool Development and Acceptance Standard](audio-tool-acceptance.md).

After feature modules are merged into `dev`, QA executes the fillable case catalog
in integration QA test-point checklist (internal record; public support status is documented in the capability guide). That
document is the manual and qualification overlay; it does not replace the
pull-request pytest layers below. Freeze the evaluated commit, supported capability
matrix, required case IDs, fixture/model/configuration hashes, and result evidence
before a QA wave. Required failed, blocked, or skipped cases prevent a pass;
no unresolved P0/P1 defect may remain in the claimed scope. Missing dependencies
for a supported feature are blocked coverage, not an out-of-scope exception.

The catalog includes standalone Wiki, PDF/book workflows, book chunks, and parser
model installation. Deterministic parser tests establish workflow behavior, not
real OCR quality. Component, integrated user-journey, model-quality, and capacity
results are separate verdicts. CLI/API/MCP agreement alone is not an independent
correctness oracle. See the catalog for regression mappings and a local Wiki smoke
recipe; unconfigured unified search remains blocked.

## Test layers

| Layer | Purpose | Pull request default |
| --- | --- | --- |
| Unit | Deterministic behavior within one module | Yes |
| Contract | Public and plugin compatibility | Yes |
| Integration | Filesystem, SQLite, and component boundaries | Selected local cases |
| End-to-end | CLI, API, MCP, and full vertical slices | Selected |
| Capacity and memory qualification | Streaming volume and memory limits, without coverage | Separate `qualification` CI job |
| Network/GPU/Ray and full model qualification | External or specialized infrastructure | Scheduled/manual |
| Spark plugin | Local execution and installed-wheel compatibility | Dedicated `spark-plugin` job in `.github/workflows/ci.yml` with JDK 17 |

## Test layout

Unit tests mirror implemented source packages. For example,
`src/capicove/cli/app.py` is exercised from `tests/unit/cli/test_app.py`, and
`src/capicove/runtime/diagnostics.py` is exercised from
`tests/unit/runtime/test_diagnostics.py`.

`tests/contract`, `tests/integration`, `tests/e2e`, and `tests/fixtures` are created
with their first real test or fixture. Do not use `.gitkeep` or placeholder tests to
preserve an empty test tier.

- Contract tests exercise public schemas, ports, plugin metadata, and compatibility.
- Integration tests use real local adapters and temporary storage at module seams.
- End-to-end tests invoke an installed CLI, API, or MCP user journey.
- Fixtures are minimal, non-sensitive, and include origin and license information
  when they are not authored by the project.

## Test rules

- Write the failing test before production behavior.
- Derive expected values independently from the implementation.
- Prefer temporary directories and local deterministic fixtures.
- Do not call external services in the pull-request fast path.
- Do not include copyrighted, personal, confidential, or credential-bearing
  fixtures.
- Record the license and origin of non-trivial fixtures.
- Test immutable versioning, hashes, idempotency, lineage, cancellation, and
  retry classification at the relevant boundaries.
- A missing optional dependency must not break unrelated imports or tests.

## Markers

The authoritative marker registry is `[tool.pytest.ini_options].markers` in
`pyproject.toml`: `unit`, `contract`, `integration`, `e2e`, `evaluation`, `cpu`,
`model_logic`, `model_onnx`, `model_small`, `model_full`, `performance`, `slow`,
`network`, `gpu`, `ray`, `spark`, `graph_slow`, and `qualification`. A test requiring specialized infrastructure
must declare its marker and document its prerequisites. `make test-fast` owns the
fast-path selection, including explicit test paths and marker exclusions. The
pre-push hook invokes that target directly. It includes unit, contract,
evaluation, and locally available model tests, plus selected batch, export,
MCP catalog, and Wiki integration checks. The `evaluation`, `cpu`,
`model_logic`, and `model_onnx` markers remain eligible because their registered
tests are deterministic and require no downloaded model or specialized
infrastructure.

## Commands

```bash
uv run pytest
uv run pytest tests/unit/cli/test_app.py -q
make test-fast
make test-regression
make test
make test-graph-slow
make test-qualification
```

Default `uv run pytest` does not collect coverage. The 85% package gate and
`term-missing` report run only from `make test` and the CI coverage job, because
branch coverage of `src/capicove` otherwise dominates pre-push and TDD cycles.
The pre-push hook invokes `make test-fast`, which also uses `--no-cov`. Use
`--no-cov` on a focused file run when you invoke coverage flags yourself.

Slow graph build checks are excluded from default pytest runs, `make test-fast`,
and `make test-regression`. The collection hook marks tests `slow` under a
`graph_build` test directory, plus slow cases in
`tests/e2e/test_document_graph_pipeline.py` and
`tests/e2e/test_graph_build_wheel.py`, as `graph_slow`; `--run-graph-slow`
selects them. `make test` and the CI coverage job pass that option, so the 85%
coverage gate measures all of `src/capicove`.

Run `make test-graph-slow` to execute only the graph checks, without coverage.
GitHub Actions also offers it through the manually dispatched **Manual graph
slow tests** workflow. For a focused graph check, add `--run-graph-slow` to
pytest.

`make test-regression` is the one-minute local regression run. It collects every
test path and excludes the same markers as `make test-fast`.

The one-minute limit is a target, not an enforced or guaranteed duration. Measure
the entire command on the development machine when changing its selection or
fixtures; moving checks out of this command does not make them redundant.

### Graph test responsibilities

- `tests/unit/graph_build/test_publication_orchestration.py` covers all 34
  table/boundary combinations with recording storage doubles. It checks publication
  order, replay arguments, stage sealing, and final result recording. Real recovery
  of SQLite and Parquet remains in `test_build_service.py`, before and after the
  first, middle, and last table, plus the separate durable checkpoint tests.
- `tests/unit/graph_build/test_snapshot_validation.py` owns cross-table rule
  matrices using authored immutable row objects. The integration catalog suite
  retains a real-Parquet invalid-row rejection and persistence, authorization,
  tampering, and transaction checks. Its module fixture publishes 17 tables once
  per xdist worker and shares their frozen manifest; every test has its own catalog
  database and access gate. Tests that mutate source files keep private publications.
  Shared constructors live in `tests/graph_build_fixtures.py`.
- Inference unit tests own structural repair/exhaustion matrices. Audit integration
  tests retain successful report repair, semantic failure, and cancellation outcomes
  through the concrete service. Service-to-adapter role checks run without storage.

### Capacity and memory qualification

The 100,000-record export case and both retrieval memory-limit cases carry
`qualification` and `slow`. They are deselected by default, including from the
coverage run; `--run-qualification` opts in. The 11-record export, source retirement,
and small multi-document build cases remain in routine regression.

`make test-qualification` runs these checks serially with `--no-cov`. The separate
`qualification` job in `ci.yml` runs on pull requests and pushes alongside the
unchanged 85% package coverage gate. Both results are needed: a passing regression
run does not qualify capacity or memory limits. The qualification cases retain
their original volumes, memory thresholds, and assertions. Do not combine this
target with coverage instrumentation, which distorts allocation and timing results.
The existing required `unit` check aggregates `coverage` and `qualification` and
fails if either job fails, is cancelled, or is skipped. This preserves the merge
gate without changing repository protection settings.

### Temporary storage and parallelism

`make test`, `make test-regression`, `make test-graph-slow`, and
`make test-qualification` run pytest
through `scripts/run_tests.py`, which behaves the same on Linux, macOS, and
Windows (CI runs it on all three in the `test-runner` job). You can call it
directly: `uv run python scripts/run_tests.py [options] -- [pytest arguments]`.

- **Location.** Each run gets one private directory, `capicove-pytest.*`,
  holding pytest's `--basetemp`, a `TMPDIR`/`TEMP`/`TMP` directory for
  subprocesses, and the shared installed-wheel environments. On Linux it uses
  `/dev/shm` when that has at least 8 GiB free, because the graph, memory, and
  book suites are fsync-bound and run about twice as fast there. Otherwise it
  uses the system temporary directory (`TMPDIR`, `TEMP`, or `TMP`). Choose a
  directory with `--tmp-parent DIR` or `make test PYTEST_TMP_PARENT=DIR`.
- **Budget.** The runner refuses to start with less than 2 GiB free
  (`--min-free-gib`). `--workers auto` uses one xdist worker per CPU, capped at
  one per GiB of available memory (where the OS reports it; macOS does not) and
  one per GiB of free disk above a 2 GiB reserve. Set `--workers N` or
  `PYTEST_WORKERS=N` to override.
- **Growth.** `tmp_path_retention_policy = "failed"` deletes a passing test's
  `tmp_path` as soon as it finishes, so usage tracks running tests rather than
  completed ones. Installed-wheel journeys share one wheel build and one
  environment per dependency selection through the session-scoped `wheel_cache`
  fixture (`tests/e2e/wheel_cache.py`); treat those environments as read-only,
  and use `wheel_cache.private_environment` in a test that changes its
  environment. Because pytest deletes `tmp_path` during teardown, scope
  patches of `os` or `shutil` functions with `monkeypatch.context()` so they are
  undone first. The runner places them on uv's cache filesystem when it can, so
  installs hardlink instead of copying.
- **Cleanup.** A passing run deletes its directory and reports anything it could
  not delete; the next run retries. A failed or interrupted run keeps its
  directory, prints its path, and keeps only failing tests' `tmp_path`. Only the
  newest failed run is kept (`--keep-failed`). The next run removes directories
  left by a crash or hard kill. Before deleting, the runner stops pytest's
  process group (xdist workers and anything a test left running); on Windows it
  stops the process tree when interrupted.

Free space is checked only at startup, so other programs filling the disk during
a run can still cause failures. Tests must stay independent of execution order
and of other workers, and must not write outside `tmp_path` or the temporary
directory.

Mark a test `slow` when it takes about one second or longer on a developer
machine, for example a full graph build, a crash-point or fencing matrix, a
corpus-scale "does not scan" check, or an installed-wheel journey. Unrelated slow
tests and slow graph build checks both run in `make test` and in the CI coverage
job. Prefer shrinking a test's data
over marking it when the smaller input still proves the property, as
`test_deep_execution_validation_is_iterative` does by lowering the recursion
limit instead of building a 1,050-level hierarchy.

Video processing tests generate a tiny synthetic ffmpeg fixture under pytest's
temporary directory. They never commit source video. An ignored local file under
`tests/datasets/` may provide supplementary developer verification but is not a CI
prerequisite. PyAV and PySceneDetect integration tests skip with the exact optional
extra to install when unavailable. Their private adapter implementations are omitted
from the base-environment coverage denominator and are exercised by the optional
`video-av` and `video-scene` integration runs; lazy availability behavior remains in
the default suite.

Video backend integration tests assert requested duration tolerance, audio/video
track preservation, timestamp alignment, scene range coverage, and usable optional
dependency metadata. Run the complete optional matrix with:

```bash
uv sync --extra video --extra mcp
uv run --extra video --extra mcp pytest --no-cov \
  tests/integration/processing/video -q
```

```bash
uv run pytest --no-cov tests/unit/processing/video -q
uv run pytest --no-cov tests/contract/processing/video -q
uv run pytest --no-cov tests/integration/processing/video -q
uv run pytest --no-cov tests/e2e/test_video_split_transport_parity.py -q
```

The initial coverage gate is 85 percent for the package. Coverage is evidence of
executed lines, not a replacement for meaningful assertions or contract tests.
