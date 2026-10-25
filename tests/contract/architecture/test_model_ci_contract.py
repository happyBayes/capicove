"""Contract tests for optional model runtimes and model CI isolation."""

from __future__ import annotations

import tomllib
from pathlib import Path


_ROOT = Path(__file__).parents[3]


def _configuration() -> dict[str, object]:
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _workflow_job(workflow: str, job_name: str) -> str:
    _, separator, jobs = workflow.partition("\njobs:\n")
    assert separator, "workflow must define jobs"

    lines = jobs.splitlines(keepends=True)
    header = f"  {job_name}:\n"
    try:
        start = lines.index(header)
    except ValueError as error:
        raise AssertionError(f"workflow must define the {job_name} job") from error

    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":")
        ),
        len(lines),
    )
    return "".join(lines[start:end])


def test_model_runtime_extras_are_reusable_across_media_domains() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    assert set(("models-cpu", "models-gpu", "models-dev")) <= extras.keys()
    assert any(str(requirement).startswith("onnxruntime") for requirement in extras["models-cpu"])
    assert not any(str(requirement).startswith("torch") for requirement in extras["models-cpu"])
    assert any(
        str(requirement).startswith("onnxruntime-gpu") for requirement in extras["models-gpu"]
    )
    assert any(str(requirement).startswith("torch") for requirement in extras["models-dev"])


def test_retrieval_cpu_extra_contains_only_onnxruntime() -> None:
    configuration = _configuration()
    project = configuration["project"]
    assert isinstance(project, dict)
    extras = project["optional-dependencies"]
    assert isinstance(extras, dict)

    assert extras["retrieval-cpu"] == ["onnxruntime>=1.20,<2"]


def test_model_test_levels_are_registered() -> None:
    configuration = _configuration()
    tool = configuration["tool"]
    assert isinstance(tool, dict)
    pytest_options = tool["pytest"]["ini_options"]
    markers = pytest_options["markers"]
    marker_names = {str(marker).partition(":")[0] for marker in markers}

    assert {"model_logic", "model_onnx", "model_small", "model_full", "cpu", "gpu"} <= marker_names


def test_cpu_model_ci_installs_no_development_or_gpu_extra() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    coverage_job = _workflow_job(workflow, "coverage")
    cpu_job = _workflow_job(workflow, "cpu-model")
    assert "--no-cov" not in coverage_job
    assert "--extra models-cpu" in cpu_job
    assert "--group model-test" in cpu_job
    assert "--extra models-dev" not in cpu_job
    assert "--extra models-gpu" not in cpu_job
    assert "model_onnx" in cpu_job
    assert "--no-cov" in cpu_job
    assert "find_spec('torch') is None" in cpu_job


def test_required_unit_gate_includes_coverage_and_capacity_qualification() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    gate = _workflow_job(workflow, "unit")
    coverage = _workflow_job(workflow, "coverage")
    qualification = _workflow_job(workflow, "qualification")

    assert "needs: [coverage, qualification]" in gate
    assert "if: ${{ always() }}" in gate
    for name in ("coverage", "qualification"):
        assert f"needs.{name}.result" in gate
        assert f'test "$CAPICOVE_{name.upper()}_RESULT" = success' in gate
    assert "--cov-fail-under=85" in coverage
    assert "--run-graph-slow" in coverage
    assert "make test-qualification" in qualification
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    command = makefile.split("\ntest-qualification:\n", 1)[1].split("\n\n", 1)[0]
    assert "--no-cov" in command
    assert "--run-qualification -m qualification" in command


def test_reverse_image_onnx_ci_is_isolated_from_transformers_and_torch() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    reverse_image_job = _workflow_job(workflow, "reverse-image-onnx")
    assert "--extra retrieval-cpu" in reverse_image_job
    assert "--group model-test" in reverse_image_job
    assert "--extra models-cpu" not in reverse_image_job
    assert "--extra models-dev" not in reverse_image_job
    assert "--extra models-gpu" not in reverse_image_job
    assert "find_spec('torch') is None" in reverse_image_job
    assert "find_spec('transformers') is None" in reverse_image_job
    assert "test_reverse_image_onnx_cpu.py" in reverse_image_job
    assert "test_reverse_image_onnx_runtime_vertical.py" in reverse_image_job


def test_package_ci_smokes_public_reverse_image_wheel_surfaces() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    package_job = _workflow_job(workflow, "package")
    assert '"onnxruntime>=1.20,<2"' in package_job
    assert '"mcp>=1.7,<2"' in package_job
    assert "from capicove.retrieval import RetrievalService" in package_job
    assert "from capicove.runtime import ReverseImageRuntime" in package_job
    assert "from capicove.tools import SEARCH_SPEC" in package_job
    assert "from capicove.mcp.image_search import MCP_SEARCH_NAME" in package_job
    assert "SEARCH_SPEC.tool_id == 'capicove.search.image'" in package_job
    assert "MCP_SEARCH_NAME == 'capicove_search_image'" in package_job
