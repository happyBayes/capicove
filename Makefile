.DEFAULT_GOAL := help

.PHONY: help install format lint typecheck quality test test-fast test-regression test-graph-slow test-qualification docs docs-check docs-serve build clean

help:
	@echo "Capicove development commands"
	@echo "  make install    Install the development environment"
	@echo "  make format     Apply Ruff fixes and formatting"
	@echo "  make quality    Run formatting, lint, and type checks"
	@echo "  make test       Run regression tests, including slow graph checks, with coverage"
	@echo "  make test-fast  Run the pull-request fast path without coverage"
	@echo "  make test-regression  Run every non-slow test in parallel (one-minute budget)"
	@echo "  make test-graph-slow  Manually run slow graph build checks in parallel"
	@echo "  make test-qualification  Run capacity and memory gates without coverage"
	@echo "  make docs       Build the documentation with strict validation"
	@echo "  make docs-serve Preview the documentation with live reload"
	@echo "  make build      Build wheel and source distributions"
	@echo "  make clean      Remove generated build and test output"

install:
	uv sync --group dev

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run --extra mcp --extra graph-data mypy src tests

quality: lint typecheck

PYTEST_COV := --cov=src/capicove --cov-report=term-missing --cov-fail-under=85
PYTEST_FAST := -m "not slow and not network and not gpu and not ray and not spark and not model_small and not model_full and not performance"
PYTEST_FAST_PATHS := tests/unit tests/contract tests/evaluation tests/model \
	tests/integration/batch/test_sqlite_store.py \
	tests/integration/batch/test_executor.py \
	tests/integration/export/test_jsonl_export.py \
	tests/integration/mcp/test_catalog_server.py \
	tests/e2e/test_local_wiki_cli.py

# scripts/run_tests.py owns temporary storage on every OS: it picks /dev/shm on
# Linux only when it has enough space, sizes xdist workers from CPU, memory, and
# disk, deletes passing runs, and keeps the newest failed run as evidence. Set
# PYTEST_TMP_PARENT=DIR or PYTEST_WORKERS=N to override its choices.
TEST_RUNNER = uv run --extra graph-data --extra graph-community --extra code-ast python scripts/run_tests.py \
	$(if $(PYTEST_TMP_PARENT),--tmp-parent "$(PYTEST_TMP_PARENT)") \
	$(if $(PYTEST_WORKERS),--workers "$(PYTEST_WORKERS)")

# The coverage gate measures all of src/capicove and includes slow graph checks.
# Capacity and memory qualification runs separately without instrumentation.
test:
	$(TEST_RUNNER) -- $(PYTEST_COV) --run-graph-slow

test-fast:
	uv run --extra graph-data --extra graph-community pytest --no-cov -q $(PYTEST_FAST_PATHS) $(PYTEST_FAST)

test-regression:
	$(TEST_RUNNER) -- --no-cov -q $(PYTEST_FAST)

test-graph-slow:
	$(TEST_RUNNER) -- --no-cov -q --run-graph-slow -m graph_slow

# Run memory measurements serially to avoid competing qualification workloads.
test-qualification:
	$(TEST_RUNNER) --workers 1 -- --no-cov -q --run-qualification -m qualification

docs:
	uv run --group docs mkdocs build --strict

docs-serve:
	uv run --group docs mkdocs serve

docs-check: docs
	uv run --group docs python scripts/check_docs.py

build:
	uv build

clean:
	uv run python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.coverage', 'htmlcov', '.pytest_cache', '.mypy_cache', '.ruff_cache')]"
