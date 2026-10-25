import ast
import subprocess
import sys
from pathlib import Path

import pytest


CONTRACT_ROOT = Path("src/capicove/contracts")
MODULE_BOUNDARIES = Path("docs/architecture/module-boundaries.md")
PACKAGE_ROOT = Path("src/capicove")
PROCESSING_IMAGE_ROOT = PACKAGE_ROOT / "processing" / "image"
RETRIEVAL_ROOT = PACKAGE_ROOT / "retrieval"
RETRIEVAL_RFC = Path("docs/rfcs/0002-multimodal-retrieval-contracts.md")
RETRIEVAL_ADR = Path("docs/adr/0008-multimodal-retrieval-boundaries.md")
SPARK_RFC = Path("docs/rfcs/0007-spark-data-processing-contracts.md")
SPARK_ADR = Path("docs/adr/0013-spark-execution-and-snapshot-boundaries.md")
SPARK_PLUGIN_ROOT = Path("plugins/execution-spark/src")
WIKI_RFC = Path("docs/rfcs/0010-local-wiki-projection-format.md")
WIKI_ADR = Path("docs/adr/0016-local-wiki-projection-boundaries.md")


@pytest.mark.contract
def test_video_package_does_not_eagerly_import_xxhash() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.processing.video; import sys; assert 'xxhash' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_canonical_contracts_import_only_stdlib_or_contract_modules() -> None:
    violations: list[str] = []
    for path in sorted(CONTRACT_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".", maxsplit=1)[0]
                if root not in sys.stdlib_module_names and not module.startswith(
                    "capicove.contracts"
                ):
                    violations.append(f"{path}: {module}")

    assert violations == []


@pytest.mark.contract
def test_retrieval_contract_import_does_not_load_optional_media_or_model_sdks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.contracts.retrieval; import sys; "
                "assert 'PIL' not in sys.modules; "
                "assert 'numpy' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_wiki_public_import_keeps_optional_and_runtime_dependencies_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.wiki; import sys; "
                "assert 'yaml' not in sys.modules; "
                "assert 'sqlite3' not in sys.modules; "
                "assert 'networkx' not in sys.modules; "
                "assert 'neo4j' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_wiki_schema_and_authority_boundaries_are_documented_for_review() -> None:
    assert WIKI_RFC.exists()
    assert WIKI_ADR.exists()
    rfc = WIKI_RFC.read_text(encoding="utf-8")
    adr = WIKI_ADR.read_text(encoding="utf-8")
    boundaries = MODULE_BOUNDARIES.read_text(encoding="utf-8")

    assert "- Status: Accepted" in rfc
    assert "- Status: Accepted" in adr
    assert "2026-09-28" in rfc
    assert "2026-09-28" in adr
    assert "not canonical Capicove truth" in rfc
    assert "Temporal Memory" in rfc
    assert "disposable" in adr
    assert "canonical truth" in boundaries
    for section in (
        "### Manifest: wiki.toml",
        "### Pages: pages/**/*.md",
        "### Ontology: graph/ontology.yaml",
        "### Facts: graph/facts.jsonl",
        "### Identity events: graph/identity-events.jsonl",
        "### Raw captures: raw/<digest>.md",
        "### Compatibility and extension fields",
    ):
        assert section in rfc
    for field in (
        "schema_version",
        "entityTypes",
        "valueType",
        "inputFacts",
        "effectiveAt",
        "source:sha256:",
    ):
        assert field in rfc


@pytest.mark.contract
def test_onnx_image_adapter_modules_do_not_eagerly_import_optional_model_sdks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.retrieval._adapters.onnx_image; "
                "import capicove.processing.image._adapters.onnx_target; import sys; "
                "assert 'onnxruntime' not in sys.modules; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_query_and_target_private_adapters_do_not_cross_capability_boundaries() -> None:
    violations: list[str] = []
    boundaries = (
        (RETRIEVAL_ROOT, "capicove.processing.image._adapters"),
        (PROCESSING_IMAGE_ROOT, "capicove.retrieval._adapters"),
    )
    for root, forbidden_prefix in boundaries:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    if module.startswith(forbidden_prefix):
                        violations.append(f"{path}: {module}")

    assert violations == []


@pytest.mark.contract
def test_retrieval_capability_does_not_import_runtime_composition() -> None:
    violations: list[str] = []
    for path in sorted(RETRIEVAL_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            violations.extend(
                f"{path}: {module}" for module in modules if module.startswith("capicove.runtime")
            )

    assert violations == []


@pytest.mark.contract
def test_retrieval_boundary_rules_are_documented() -> None:
    text = MODULE_BOUNDARIES.read_text(encoding="utf-8")

    assert "MediaRepresentationEncoder" in text
    assert "QueryEncoder" in text
    assert "IndexGenerationCatalog" in text
    assert "CLI and MCP" in text
    assert "one workspace and its sole index" in text


@pytest.mark.contract
def test_retrieval_change_control_accepts_single_node_mvp() -> None:
    assert RETRIEVAL_RFC.exists()
    assert RETRIEVAL_ADR.exists()

    rfc = RETRIEVAL_RFC.read_text(encoding="utf-8")
    adr = RETRIEVAL_ADR.read_text(encoding="utf-8")

    assert "- Status: Accepted" in rfc
    assert "- Status: Accepted" in adr
    assert "Option A" in adr
    assert "MVP" in adr
    assert "Option B" in adr
    assert "rejected" in adr.lower()
    assert "Option C" in adr
    assert "Post-MVP" in adr
    assert "federated retrieval" in adr.lower()


@pytest.mark.contract
def test_spark_processing_boundaries_are_accepted() -> None:
    assert SPARK_RFC.exists()
    assert SPARK_ADR.exists()
    assert "- Status: Accepted" in SPARK_RFC.read_text(encoding="utf-8")
    assert "- Status: Accepted" in SPARK_ADR.read_text(encoding="utf-8")


@pytest.mark.contract
def test_spark_remains_outside_canonical_contracts() -> None:
    text = MODULE_BOUNDARIES.read_text(encoding="utf-8")

    assert "committed DatasetSnapshotRef" in text
    assert "must not import PySpark" in text
    assert "must not call Ray" in text

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.contracts.data; import sys; "
                "assert 'pyspark' not in sys.modules; "
                "assert 'pyarrow' not in sys.modules; "
                "assert 'ray' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_spark_plugin_does_not_import_core_private_adapters() -> None:
    violations: list[str] = []
    for path in sorted(SPARK_PLUGIN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("capicove.processing._adapters"):
                    violations.append(f"{path}: {node.module}")

    assert violations == []


@pytest.mark.slow
@pytest.mark.contract
def test_capabilities_do_not_import_private_contract_modules() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.is_relative_to(CONTRACT_ROOT):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.startswith("capicove.contracts._"):
                    violations.append(f"{path}: {module}")

    assert violations == []
