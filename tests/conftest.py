"""Shared fixtures for deterministic execution of long-running test suites."""

from pathlib import Path

import pytest

from tests.deadline_fixtures import refresh_module_deadlines


_GRAPH_E2E_MODULES = frozenset(
    {"e2e/test_document_graph_pipeline.py", "e2e/test_graph_build_wheel.py"}
)
_RELATIVE_DEADLINE_MODULES = frozenset(
    {
        "e2e/test_reverse_image_runtime.py",
        "integration/retrieval/test_generation_lifecycle.py",
        "integration/retrieval/test_generation_publication.py",
        "integration/retrieval/test_personal_generation_build.py",
        "integration/retrieval/test_personal_generation_runtime.py",
        "integration/retrieval/test_sqlite_unit_catalog.py",
        "model/test_reverse_image_onnx_cpu.py",
        "model/test_reverse_image_onnx_runtime_vertical.py",
        "unit/processing/image/test_onnx_target.py",
        "unit/processing/image/test_representations.py",
        "unit/retrieval/test_onnx_image.py",
        "unit/retrieval/test_personal_image_research.py",
        "unit/retrieval/test_personal_numpy.py",
        "unit/retrieval/test_scale.py",
        "unit/retrieval/test_scale_oracle.py",
        "unit/retrieval/test_search_service.py",
        "unit/runtime/test_image_query.py",
        "unit/runtime/test_retrieval_build.py",
        "unit/runtime/test_retrieval_personal.py",
        "unit/runtime/test_retrieval_profiles.py",
    }
)
_RELATIVE_NOW_MODULES = frozenset(
    {
        "e2e/test_reverse_image_runtime.py",
        "integration/retrieval/test_generation_lifecycle.py",
        "integration/retrieval/test_generation_publication.py",
        "integration/retrieval/test_personal_generation_runtime.py",
        "model/test_reverse_image_onnx_runtime_vertical.py",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose opt-ins for slow graph checks and uninstrumented qualification."""
    parser.getgroup("capicove").addoption(
        "--run-graph-slow",
        action="store_true",
        default=False,
        help="Include slow graph build tests (excluded by default).",
    )
    parser.getgroup("capicove").addoption(
        "--run-qualification",
        action="store_true",
        default=False,
        help="Include capacity and memory qualification (run without coverage).",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Require explicit opt-ins for graph slow and capacity qualification checks."""
    run_graph_slow = config.getoption("--run-graph-slow")
    run_qualification = config.getoption("--run-qualification")
    tests_root = Path(__file__).parent
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker("qualification") is not None and not run_qualification:
            deselected.append(item)
            continue
        if not item.path.is_relative_to(tests_root):
            selected.append(item)
            continue
        relative_path = item.path.relative_to(tests_root)
        is_graph = (
            "graph_build" in relative_path.parts[:-1]
            or relative_path.as_posix() in _GRAPH_E2E_MODULES
        )
        if is_graph and item.get_closest_marker("slow") is not None:
            item.add_marker(pytest.mark.graph_slow)
            if not run_graph_slow:
                deselected.append(item)
                continue
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def _fresh_relative_deadlines(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent collection-time deadline constants expiring in a long suite."""
    module_path = Path(str(request.module.__file__)).relative_to(Path(__file__).parent)
    relative_path = module_path.as_posix()
    if relative_path in _RELATIVE_DEADLINE_MODULES:
        refresh_module_deadlines(
            request.module,
            monkeypatch,
            refresh_now=relative_path in _RELATIVE_NOW_MODULES,
        )
